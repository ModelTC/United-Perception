import torch
from .base_runner import BaseRunner
from up.utils.general.registry_factory import MODEL_HELPER_REGISTRY, RUNNER_REGISTRY
from up.utils.general.log_helper import default_logger as logger
from up.utils.env.gene_env import print_network
from up.utils.general.saver_helper import Saver
from up.tasks.distill import Mimicker, MimicJob

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
        self.teacher_model = model_helper_ins(self.config['teacher'], **model_helper_kwargs)
        if self.device == 'cuda':
            self.model = self.model.cuda()
            self.teacher_model = self.teacher_model.cuda()
        if self.config['runtime']['special_bn_init']:
            self.special_bn_init()
        if self.fp16 and self.backend == 'linklink':
            self.model = self.model.half()
            self.teacher_model = self.teacher_model.half()
        if self.config['mimic']['teacher'].get('teacher_weight', None):
            teacher_weight_path = self.config['mimic']['teacher']['teacher_weight']
            logger.info(f'loading teacher weight: {teacher_weight_path}')
            state = Saver.load_checkpoint(teacher_weight_path)
            if 'ema' in state:
                if "ema_state_dict" in state['ema']:
                    logger.info("Load ema pretrain model")
                    st = state['ema']['ema_state_dict']
                else:
                    st = state['model']
            else:
                st = state['model']
            self.teacher_model.load(st)

        if self.config['mimic']['teacher'].get('teacher_bn_mode', None) == 'train':
            self.teacher_model.train()
        else:
            self.teacher_model.eval()
        logger.info(print_network(self.model))
        logger.info(print_network(self.teacher_model))
        self.build_mimic()

    def build_mimic(self):
        self.mimic_name = self.config['mimic'].get('mimic_name', 'job1')
        self.mimic_loss_type = self.config['mimic'].get('mimic_type', 'kl')
        self.mimic_loss_weight = self.config['mimic'].get('loss_weight', 1.0)
        self.mimic_norm = self.config['mimic'].get('mimic_norm', False)
        self.task_loss = self.config['mimic'].get('task_loss', False)
        self.teacher_mimic_name = self.config['mimic']['teacher'].get('mimic_name', None)
        self.student_mimic_name = self.config['mimic']['student'].get('mimic_name', None)
        self.mimicker = Mimicker(teacher_model=self.teacher_model, student_model=self.model)
        # create a mimic job
        mimicjob = MimicJob(self.mimic_name,
                            self.mimic_loss_type, self.student_mimic_name, self.teacher_mimic_name,
                            mimic_loss_weight=self.mimic_loss_weight)
        self.mimicker.register_job(mimicjob)

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
            self.mimicker.prepare()
            batch = self.get_batch('train')
            task_loss, output = self.forward_train(batch)
            with torch.no_grad():
                self.teacher_model(batch)
            loss = self.mimicker.mimic()
            loss = sum(loss)
            output['KD.loss'] = loss
            self._hooks('after_forward', self.cur_iter, output)
            if self.task_loss:
                loss += task_loss
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
