import torch
from up.runner.base_runner import BaseRunner
from up.utils.general.registry_factory import MODEL_HELPER_REGISTRY, RUNNER_REGISTRY, MIMIC_REGISTRY
from up.utils.general.log_helper import default_logger as logger
from up.utils.env.gene_env import print_network
from up.utils.general.saver_helper import Saver

__all__ = ['KDRunner']


@RUNNER_REGISTRY.register('kd')
class KDRunner(BaseRunner):
    def __init__(self, config, work_dir='./', training=True):
        super(KDRunner, self).__init__(config, work_dir, training)

    def build_model(self):
        model_helper_type = self.config['runtime']['model_helper']['type']
        model_helper_kwargs = self.config['runtime']['model_helper']['kwargs']
        model_helper_ins = MODEL_HELPER_REGISTRY[model_helper_type]
        self.model = model_helper_ins(self.config['net'], **model_helper_kwargs)
        self.teacher_models = {}
        self.teacher_nums = 0
        for key in self.config.keys():
            if 'teacher' in key:
                self.teacher_nums += 1
                self.teacher_models.update({key: model_helper_ins(self.config[key],
                                                                  **model_helper_kwargs)})
        logger.info(f"The student model mimics from {str(self.teacher_nums)} teacher networks \
            which are {','.join(self.teacher_models.keys())}.")

        if self.device == 'cuda':
            self.model = self.model.cuda()
            for tmk in self.teacher_models:
                self.teacher_models[tmk] = self.teacher_models[tmk].cuda()
        if self.config['runtime']['special_bn_init']:
            self.special_bn_init()
        if self.fp16 and self.backend == 'linklink':
            self.model = self.model.half()
            for tmk in self.teacher_models:
                self.teacher_models[tmk] = self.teacher_models[tmk].half()

        self.mimic_configs, self.teacher_configs = self.prepare_mimic_configs(self.config['mimic'])
        for tmk in self.teacher_models:
            self.load_teacher_weight(tmk, self.teacher_models[tmk],
                                     self.teacher_configs[tmk].get('teacher_weight', None),
                                     self.teacher_configs[tmk].get('teacher_bn_mode', None))
        logger.info(print_network(self.model))
        for tmk in self.teacher_models:
            logger.info(print_network(self.teacher_models[tmk]))
        self.build_mimic()

    def prepare_mimic_configs(self, raw_configs):
        self.task_loss = True
        if 'task_loss' in raw_configs:
            self.task_loss = raw_configs.pop('task_loss')
        mimic_configs = {}
        if 'mimic_as_jobs' in raw_configs:
            assert raw_configs['mimic_as_jobs'] is True, 'mimic_as_jobs must be True or do not appear!'
            raw_configs.pop('mimic_as_jobs')
            mimic_configs = raw_configs
        else:
            assert 'student' in raw_configs.keys(), 'Wrong mimic setting! Single mimic job must contain the student key. \
                If you want to mimic multi jobs, set the mimic_as_jobs=True and name each mimic as a mimic_job'
            mimic_configs['mimic_job0'] = raw_configs
        teacher_configs = {}
        for mjx in mimic_configs.keys():
            assert 'mimic_job' in mjx, 'mimic jobs must start with the mimic_job prefix!'
            mimic_configs[mjx]['t_names'] = []
            for atk in mimic_configs[mjx].keys():
                if 'teacher' in atk:
                    assert atk not in teacher_configs, 'The teacher keys are unique even in different mimic jobs. \
                        If you want to use a same teacher model with multi mimic methods, \
                        you must copy and rename it to multi teacher model configs.'
                    teacher_configs.update({atk: mimic_configs[mjx][atk]})
                    mimic_configs[mjx]['t_names'].append(atk)
            assert len(mimic_configs[mjx]['t_names']) > 0, 'useless mimic job {}'.format(mjx)
        no_config_teachers = list(set(self.teacher_models.keys()) - set(teacher_configs.keys()))
        extra_config_teachers = list(set(teacher_configs.keys()) - set(self.teacher_models.keys()))
        if len(no_config_teachers) > 0:
            logger.warning('teachers {} has no configs. \
                    This is a waste of CUDA memory'.format(','.join(no_config_teachers)))
        assert len(extra_config_teachers) == 0, "The configs of {} is invalid \
                because you do not define these teacher models".format(','.join(extra_config_teachers))
        return mimic_configs, teacher_configs

    def load_teacher_weight(self, teacher_name, teacher_model, teacher_weight_path, teacher_bn_mode):
        # teacher_weight_path = mimic_configs.get('teacher_weight', None)
        if teacher_weight_path:
            logger.info(f'loading {teacher_name} weight: {teacher_weight_path}')
            state = Saver.load_checkpoint(teacher_weight_path)
            if 'ema' in state:
                if "ema_state_dict" in state['ema']:
                    logger.info("Load ema pretrain model")
                    st = state['ema']['ema_state_dict']
                else:
                    st = state['model']
            else:
                st = state['model']
            teacher_model.load(st)
        if teacher_bn_mode == 'train':
            teacher_model.train()
        else:
            teacher_model.eval()

    def build_mimic(self):
        self.mimic_jobs = {}
        for key in self.mimic_configs.keys():
            mimic_ins_type = self.mimic_configs[key].get('mimic_ins_type', 'base')
            mimic_ins = MIMIC_REGISTRY[mimic_ins_type]
            teacher_models = [self.teacher_models[tmk] for tmk in self.mimic_configs[key]['t_names']]
            teacher_mimic_names = [self.teacher_configs[tmk]['mimic_name'] for tmk in self.mimic_configs[key]['t_names']]  # noqa

            loss_weight = self.mimic_configs[key].get('loss_weight', 1.0)
            warm_up_iters = self.mimic_configs[key].get('warm_up_iters', -1)
            cfgs = self.mimic_configs[key].get('cfgs', {})
            self.mimic_jobs[key] = mimic_ins(teacher_model=teacher_models, student_model=self.model,
                                             teacher_names=self.mimic_configs[key]['t_names'],
                                             teacher_mimic_names=teacher_mimic_names,
                                             student_mimic_names=self.mimic_configs[key]['student']['mimic_name'],
                                             loss_weight=loss_weight, warm_up_iters=warm_up_iters,
                                             configs=cfgs)

    def forward_model(self, batch):
        if self.ema is not None and not self.model.training:
            return self.ema.model(batch)
        return self.model(batch)

    def forward(self, batch, return_output=False):
        if self.backend == 'dist' and self.fp16:
            with torch.cuda.amp.autocast(enabled=True):
                output = self.forward_model(batch)
        else:
            output = self.forward_model(batch)
        self._temporaries['last_output'] = output
        losses = [val for name, val in output.items() if name.find('loss') >= 0]
        loss = sum(losses)
        if return_output:
            return loss, output
        else:
            return loss

    def forward_train(self, batch):
        assert self.model.training
        self._hooks('before_forward', self.cur_iter, batch)
        loss, output = self.forward(batch, return_output=True)
        return loss, output

    def train(self):
        self.model.cuda().train()
        for iter_idx in range(self.start_iter, self.max_iter):
            for key in self.mimic_configs.keys():
                self.mimic_jobs[key].prepare()
            batch = self.get_batch('train')
            task_loss, output = self.forward_train(batch)
            output.update({'cur_iter': iter_idx})
            mimic_loss = 0
            for key in self.mimic_configs.keys():
                teacher_output = []
                with torch.no_grad():
                    for tmk in self.mimic_configs[key]['t_names']:
                        teacher_output.append(self.teacher_models[tmk](batch))
                mimic_output = self.mimic_jobs[key].mimic(s_output=output, t_output=teacher_output)
                mimic_loss += sum([val for name, val in mimic_output.items() if name.find('loss') >= 0])
                mok = list(mimic_output.keys())
                for mk in mok:
                    mimic_output['Mimic.' + key + '.' + mk] = mimic_output.pop(mk)
                output.update(mimic_output)
            self._hooks('after_forward', self.cur_iter, output)
            if self.task_loss:
                loss = mimic_loss + task_loss
            else:
                loss = mimic_loss
            self.backward(loss)
            self.update()
            if self.ema is not None:
                self.ema.step(self.model, curr_step=iter_idx)
            if self.is_test(iter_idx):
                if iter_idx == self.start_iter:
                    continue
                if self.config['runtime']['async_norm']:
                    from up.utils.env.dist_helper import all_reduce_norm
                    logger.info("all reduce norm")
                    all_reduce_norm(self.model)
                    if self.ema is not None:
                        all_reduce_norm(self.ema.model)
                self.evaluate()
                self.model.train()
            if self.only_save_latest:
                self.save_epoch_ckpt(iter_idx)
            else:
                if self.is_save(iter_idx):
                    self.save()
            self.lr_scheduler.step()
