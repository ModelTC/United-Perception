import os
import json
import random
import torch
import copy
from up.runner.base_runner import BaseRunner
from up.utils.general.registry_factory import MODEL_HELPER_REGISTRY, RUNNER_REGISTRY
from up.utils.general.log_helper import default_logger as logger
from up.utils.env.dist_helper import env, reduce_gradients, barrier, broadcast_object
from up.utils.env.gene_env import print_network
from up.data.metrics.base_evaluator import Metric

from up.tasks.nas.bignas.utils.registry_factory import CONTROLLER_REGISTRY
from up.tasks.nas.bignas.controller.utils.misc import clever_format, parse_flops, \
    rewrite_batch_norm_bias


__all__ = ['BignasRunner']


@RUNNER_REGISTRY.register("bignas")
class BignasRunner(BaseRunner):
    def __init__(self, config, work_dir='./', training=True):
        super(BignasRunner, self).__init__(config, work_dir, training)

    def build(self):
        super(BignasRunner, self).build()
        self.build_bignas_env()

    def build_bignas_env(self):
        self.controller_type = self.config.get('controller_type', 'BaseController')
        self.build_teacher()
        self.controller = CONTROLLER_REGISTRY[self.controller_type](
            self.config['bignas'], self.config, self.model, self.teacher_models)
        self.bignas_path = os.path.join(self.work_dir, 'bignas')
        if env.is_master() and not os.path.exists(self.bignas_path):
            os.makedirs(self.bignas_path)
        self.controller.set_path(self.bignas_path)
        self.sample_mode = 'max'

    def prepare_fp16(self):
        super().prepare_fp16()
        if self.config['runtime'].get('fp16', False) \
                and self.config['runtime']['fp16'].get('keep_batchnorm_fp32', True) \
                and self.backend == 'linklink':
            import spring.linklink as linklink
            from up.tasks.nas.bignas.models.ops.dynamic_ops import DynamicBatchNorm2d, \
                DynamicSyncBatchNorm2d, DynamicGroupSyncBatchNorm2d
            linklink.fp16.register_float_module(DynamicBatchNorm2d, cast_args=True)
            linklink.fp16.register_float_module(DynamicSyncBatchNorm2d, cast_args=True)
            linklink.fp16.register_float_module(DynamicGroupSyncBatchNorm2d, cast_args=True)

    def build_teacher(self):
        self.teacher_models = {}
        if not self.config['bignas'].get('distiller', False):
            return
        model_helper_type = self.config['runtime']['model_helper']['type']
        model_helper_kwargs = self.config['runtime']['model_helper']['kwargs']
        model_helper_ins = MODEL_HELPER_REGISTRY[model_helper_type]
        self.teacher_nums = 0
        for key in self.config['bignas']['distiller'].keys():
            if 'teacher' in key:
                self.teacher_nums += 1
                if self.config['bignas']['distiller'][key]:
                    self.teacher_models.update({key: model_helper_ins(self.config['bignas']['distiller'][key],
                                                                      **model_helper_kwargs)})
                else:
                    self.teacher_models.update({key: self.model})
        logger.info(f"The student model mimics from {str(self.teacher_nums)} teacher networks \
            which are {','.join(self.teacher_models.keys())}.")
        if self.device == 'cuda':
            for tmk in self.teacher_models.keys():
                if self.config['bignas']['distiller'][tmk]:
                    self.teacher_models[tmk] = self.teacher_models[tmk].cuda()
        if self.fp16 and self.backend == 'linklink':
            for tmk in self.teacher_models.keys():
                if self.config['bignas']['distiller'][tmk]:
                    self.teacher_models[tmk] = self.teacher_models[tmk].half()
        for tmk in self.teacher_models:
            logger.info(print_network(self.teacher_models[tmk]))

    def adjust_model(self, batch):
        # adjust model
        curr_step = self.curr_step if hasattr(self, 'curr_step') else 0
        curr_subnet_num = self.curr_subnet_num if hasattr(self, 'curr_subnet_num') else 0
        subnet_seed = int('%d%.3d' % (curr_step, curr_subnet_num))
        random.seed(subnet_seed)
        subnet_settings, sample_mode = self.controller.adjust_model(curr_step, curr_subnet_num)
        self.sample_mode = sample_mode

        # adjust input with sample mode
        input = self.controller.adjust_input(batch['image'], curr_subnet_num, sample_mode)
        batch['image'] = input
        self.controller.subnet_log(curr_subnet_num, batch['image'], subnet_settings)

    def backward(self, loss):
        if self.backend == 'dist':
            if self.fp16:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
        elif self.backend == 'linklink':
            loss = loss / env.world_size
            if self.fp16:
                self.optimizer.backward(loss)
            else:
                loss.backward()
            self._hooks('before_allreduce', self.cur_iter)
            reduce_gradients(self.model, not self.args['asynchronize'],
                             self.args.get('allow_dead_parameter', False))
            self._hooks('after_allreduce', self.cur_iter)
        else:
            raise NotImplementedError
        self._hooks('after_backward', self.cur_iter, loss)
        return loss

    def forward_train(self, batch):
        assert self.model.training
        self._hooks('before_forward', self.cur_iter, batch)
        loss, output = self.forward(batch, return_output=True)
        return loss, output

    def train(self):
        self.valid_before_train = self.config['bignas']['train'].get('valid_before_train', False)
        if self.valid_before_train:
            self.evaluate_specific_subnets()
        self.display = self.config['bignas'].get('display', 10)

        self.model.cuda().train()
        self.start_iter = self.start_iter - self.start_iter % 4
        for iter_idx in range(self.start_iter, self.max_iter):
            batch = self.get_batch('train')
            self.model.zero_grad()
            for curr_subnet_num in range(self.controller.sample_subnet_num):
                self.curr_step = iter_idx
                self.curr_subnet_num = curr_subnet_num
                self.controller.adjust_teacher(batch, curr_subnet_num)
                self.adjust_model(batch)
                task_loss, output = self.forward_train(batch)
                output.update({'cur_iter': iter_idx})
                self._hooks('after_forward', self.cur_iter, output)
                mimic_loss = self.controller.get_distiller_loss(self.sample_mode, output, curr_subnet_num)
                loss = mimic_loss + task_loss
                if self.backend == 'dist':
                    fake_loss = 0.0
                    for param in self.model.parameters():
                        fake_loss += param.sum() * 0
                    loss += fake_loss
                self.backward(loss)
            if iter_idx % self.display == 0:
                self.controller.show_subnet_log()
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
                self.evaluate_specific_subnets()
                self.model.train()
            if self.only_save_latest:
                self.save_epoch_ckpt(iter_idx)
            else:
                if self.is_save(iter_idx):
                    self.save()
            self.lr_scheduler.step()

    def ofa_calib_bn(self, model, sample_mode='max'):
        calib_data = self.data_loaders['train_calib']
        model_dtype = torch.float32
        if self.fp16 and self.backend == 'linklink':
            model_dtype = self.model.dtype

        model = self.controller.reset_subnet_running_statistics(model, calib_data, model_dtype)
        return model

    def get_subnet_accuracy(self, image_size=None, subnet_settings=None, calib_bn=False):
        if self.ema:
            self.model_load_ema_state_dict()
        if image_size is None:
            image_size = self.controller.sample_image_size(sample_mode='random')
        if subnet_settings is None:
            subnet_settings = self.controller.sample_subnet_settings(sample_mode='random')
        else:
            subnet_settings = self.controller.sample_subnet_settings('subnet', subnet_settings)
        if calib_bn:
            self.model = self.ofa_calib_bn(self.model, image_size)
        metrics = self.evaluate()
        if self.ema:
            self.model_recover_old_state_dict()
        acc1 = round(metrics[self.controller.metric1], 3)
        acc2 = round(metrics[self.controller.metric2], 3)
        subnet = {'subnet_settings': subnet_settings, 'image_size': image_size,
                  self.controller.metric1: acc1, self.controller.metric2: acc2}
        logger.info('Subnet with settings:\t{}'.format(json.dumps(subnet)))
        return acc1, acc2

    def model_load_ema_state_dict(self):
        self.old_state_dict = self.model.state_dict()
        ema_state_dict = self.ema.model.state_dict()
        self.model.load_state_dict(ema_state_dict, strict=False)
        return self.model

    def model_recover_old_state_dict(self):
        self.model.load_state_dict(self.old_state_dict, strict=False)

    def forward_model(self, batch):
        return self.model(batch)

    def get_subnet_latency(self, image_size, subnet_settings, flops, onnx_name=None):
        raise NotImplementedError

    # build finetune subnet
    def build_subnet_finetune_dataset(self, image_size, max_iter=None):
        if max_iter is None:
            max_iter = self.max_iter
        self.start_iter = 0
        self.max_iter = max_iter
        logger.info('build subnet finetune training dataset with max_iter {}'.format(self.max_iter))

    def sample_subnet_weight(self):
        self.subnet = self.controller.subnet
        assert self.subnet is not None

        subnet = self.controller.get_subnet_weight(self.subnet.subnet_settings)
        logger.info(subnet)

        state_dict = {}
        state_dict['model'] = subnet.state_dict()
        ckpt_name = f'{self.bignas_path}/ckpt_subnet.pth.tar'
        torch.save(state_dict, ckpt_name)

    # During the training process, sample maximum and minimum model and get accuracy
    def evaluate_specific_subnets(self):
        for sample_mode in ['max', 'min']:
            subnet_settings, sample_mode = self.controller.adjust_model(0, 0, sample_mode=sample_mode)
            logger.info('evaluate {} models with subnet settings {}'.format(sample_mode, subnet_settings))
            self.get_subnet_accuracy(subnet_settings=subnet_settings)
        return

    # evaluate subnet
    def evaluate_subnet(self):
        self.subnet = self.controller.subnet
        assert self.subnet is not None

        calib_bn = self.subnet.get('calib_bn', False)
        flops, params, _, _ = self.controller.get_subnet_flops(self.subnet.image_size, self.subnet.subnet_settings)
        acc1, acc2 = self.get_subnet_accuracy(self.subnet.image_size, self.subnet.subnet_settings, calib_bn=calib_bn)
        subnet = {'subnet_settings': self.subnet.subnet_settings, 'image_size': self.subnet.image_size,
                  self.controller.metric1: acc1, self.controller.metric2: acc2,
                  'flops': flops, 'params': params}
        logger.info('Evaluate_subnet:\t{}'.format(json.dumps(subnet)))
        rewrite_batch_norm_bias(self.model)

        self.save_subnet_weight = self.subnet.get('save_subnet_weight', True)
        self.test_subnet_latency = self.subnet.get('test_subnet_latency', True)

        if self.test_subnet_latency:
            self.get_subnet_latency(self.subnet.image_size, self.subnet.subnet_settings, flops)
        if self.save_subnet_weight:
            self.sample_subnet_weight()

        return flops, params, acc1, acc2

    # finetune
    def finetune_subnet(self):
        self.subnet = self.controller.subnet
        assert self.subnet is not None
        acc1, acc2 = self.get_subnet_accuracy(self.subnet.image_size, self.subnet.subnet_settings)
        flops, params, _, _ = self.controller.get_subnet_flops(self.subnet.image_size, self.subnet.subnet_settings)
        subnet = {'subnet_settings': self.subnet.subnet_settings, 'image_size': self.subnet.image_size,
                  self.controller.metric1: acc1, self.controller.metric2: acc2,
                  'flops': flops, 'params': params}
        logger.info('Before finetune:\t{}'.format(json.dumps(subnet)))
        self.build_subnet_finetune_dataset(self.subnet.image_size)
        self.train()
        acc1, acc2 = self.get_subnet_accuracy(self.subnet.image_size, self.subnet.subnet_settings, calib_bn=True)
        subnet[self.controller.metric1] = acc1
        subnet[self.controller.metric2] = acc2
        logger.info('After finetune:\t{}'.format(json.dumps(subnet)))
        return flops, params, acc1, acc2

    def sample_multiple_subnet_flops(self):
        self.lut_table, _ = self.controller.sample_subnet_lut(test_latency=False)

    def sample_multiple_subnet_accuracy(self):
        self.subnet = self.controller.subnet
        assert self.subnet is not None
        _, self.lut_table = self.controller.sample_subnet_lut(test_latency=False)
        self.sample_with_finetune = self.subnet.get('sample_with_finetune', False)
        self.performance_dict = []
        self.baseline_flops = parse_flops(self.subnet.get('baseline_flops', None))
        self.test_subnet_latency = self.subnet.get('test_subnet_latency', False)

        for k, v in self.lut_table.items():
            self.subnet.image_size = v['image_size']
            self.subnet.subnet_settings = v['subnet_settings']
            if self.sample_with_finetune:
                loadpath = self.config['model'].get('loadpath', None)
                assert loadpath is not None
                state = torch.load(loadpath, map_location='cpu')
                self.model.load_state_dict(state)
                self.build_subnet_finetune_dataset(self.subnet.image_size)
                _, _, acc1, acc2 = self.finetune_subnet()
            else:
                acc1, acc2 = self.get_subnet_accuracy(v['image_size'], v['subnet_settings'], calib_bn=True)
            v[self.controller.metric1] = acc1
            v[self.controller.metric2] = acc2

            logger.info('Sample_subnet_({}):\t{}'.format(k, json.dumps(v)))
            self.performance_dict.append(v)

        logger.info('All Subnets Information')
        for v in self.performance_dict:
            logger.info(json.dumps(v))
        logger.info('--------------------------------------------')

        self.get_top10_subnets(copy.deepcopy(self.performance_dict))
        self.get_pareto_subnets(copy.deepcopy(self.performance_dict))

    def get_top10_subnets(self, performance_dict):
        self.baseline_flops = parse_flops(self.subnet.get('baseline_flops', None))
        if self.baseline_flops is None:
            return
        performance_dict = sorted(performance_dict, key=lambda x: x[self.controller.metric1], reverse=True)
        candidate_dict = []
        for info in performance_dict:
            flop = info['flops']['total']
            if (flop - self.baseline_flops) / self.baseline_flops < 0.01:
                candidate_dict.append(info)
        length = len(candidate_dict) if len(candidate_dict) > 10 else 10
        if length == 0:
            return
        subnet_count = 10 if length > 10 else length
        logger.info('---------------top10------------------------')
        for c in (candidate_dict[length - subnet_count:length]):
            c['flops'] = clever_format(c['flops'])
            logger.info(json.dumps(c))
        logger.info('--------------------------------------------')

    def get_pareto_subnets(self, performance_dict):
        pareto = []
        performance_dict = sorted(performance_dict, key=lambda x: x[self.controller.metric1], reverse=True)
        for info in performance_dict:
            flag = True
            for _ in performance_dict:
                if info[self.controller.metric1] < _[self.controller.metric1] \
                   and info['flops']['total'] >= _['flops']['total']:
                    flag = False
                    break
                if info[self.controller.metric1] <= _[self.controller.metric1] \
                   and info['flops']['total'] > _['flops']['total']:
                    flag = False
                    break
            if flag:
                pareto.append(info)

        logger.info('------------------pareto--------------------')
        for p in pareto:
            p['flops'] = clever_format(p['flops'])
            logger.info(json.dumps(p))
        logger.info('--------------------------------------------')

    @torch.no_grad()
    def evaluate(self):
        res_file, all_device_results_list = self.inference()
        if env.is_master():
            logger.info("begin evaluate")
            metrics = self.data_loaders['test'].dataset.evaluate(res_file, all_device_results_list)
            try:
                logger.info(json.dumps(metrics, indent=2))
            except BaseException:
                logger.warning('metrics type is incompatible for display')
        else:
            metrics = Metric({})
        barrier()
        metrics = broadcast_object(metrics)
        try:
            self._hooks('after_eval', metrics)
        except BaseException:
            logger.warning('metrics type is incompatible for hook')
        self.set_cur_eval_iter()
        return metrics
