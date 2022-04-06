from up.runner.base_runner import BaseRunner
from up.utils.general.registry_factory import MODEL_HELPER_REGISTRY, RUNNER_REGISTRY
from up.utils.env.gene_env import get_env_info
from up.utils.env.dist_helper import env
from up.utils.general.log_helper import default_logger as logger
from up.utils.general.cfg_helper import format_cfg
from mqbench.utils.state import enable_calibration, enable_quantization

__all__ = ['QuantRunner']


@RUNNER_REGISTRY.register("quant")
class QuantRunner(BaseRunner):
    def __init__(self, config, work_dir='./', training=True):
        self.do_ptq = True
        super(QuantRunner, self).__init__(config, work_dir, training)

    def build(self):
        self.build_env()
        self.get_git_version()
        self.build_dataloaders()
        self.build_model()
        self.build_ema()
        self.build_saver()
        self.build_hooks()
        self.load_ckpt()
        if self.training:
            self.resume_model_from_fp()
            self.quantize_model()
            self.build_trainer()
            self.resume_model_from_quant()
            if self.do_ptq:
                self.calibrate()
        else:
            self.quantize_model()
            self.resume_model_from_quant()
            enable_quantization(self.model)
        self.prepare_dist_model()
        get_env_info()
        self.save_running_cfg()
        if not self.args.get('no_running_config', False):
            logger.info('Running with config:\n{}'.format(format_cfg(self.config)))

    def build_model(self):
        QuantModelHelper = MODEL_HELPER_REGISTRY['quant']
        self.model = QuantModelHelper(self.config)
        self.model_list = self.model.model_list
        self.model_map = self.model.model_map
        if env.is_master():
            print(self.model)
            print(self.model_list)
            print(self.model_map)
        if self.device == 'cuda':
            self.model = self.model.cuda()
        if self.config['runtime']['special_bn_init']:
            self.special_bn_init()

    def load_ckpt(self):
        self.ckpt = self.saver.load_pretrain_or_resume()
        if 'ema' in self.ckpt:
            if "ema_state_dict" in self.ckpt['ema']:
                self.ckpt['model'] = self.ckpt['ema']['ema_state_dict']

    def resume_model_from_fp(self):
        def add_prefix(state_dict):
            f = lambda x: self.model_map[x.split('.')[0]] + '.' + x
            return {f(key): value for key, value in state_dict.items()}
        if 'qat' not in self.ckpt:
            self.model.load_state_dict(add_prefix(self.ckpt['model']))

    def resume_model_from_quant(self):
        if 'qat' in self.ckpt:
            self.do_ptq = False
            self.model.load_state_dict(self.ckpt['model'])
            if self.ckpt['qat'] and self.training and 'optimizer' in self.ckpt:
                start_epoch = self.ckpt['epoch']
                self.optimizer.load_state_dict(self.ckpt['optimizer'])
                self.lr_scheduler.load_state_dict(self.ckpt['lr_scheduler'])
                logger.info(f'resume from epoch:{start_epoch} iteration:{self.cur_iter}')

    @property
    def backend_type(self):
        from mqbench.prepare_by_platform import BackendType
        return {'tensorrt': BackendType.Tensorrt,
                'snpe': BackendType.SNPE,
                'vitis': BackendType.Vitis,
                'academic': BackendType.Academic}

    def get_leaf_module(self, leaf_module):
        from mqbench.nn.modules import FrozenBatchNorm2d
        from up.tasks.det.plugins.yolov5.models.components import Space2Depth
        leaf_module_map = {'FrozenBatchNorm2d': FrozenBatchNorm2d,
                           'Space2Depth': Space2Depth}
        leaf_module_ret = [leaf_module_map[k] for k in leaf_module]
        return tuple(leaf_module_ret)

    def get_extra_quantizer_dict(self, extra_quantizer_dict):
        from mqbench.nn.intrinsic.qat.modules import ConvFreezebn2d, ConvFreezebnReLU2d
        additional_module_type = extra_quantizer_dict.get('additional_module_type')
        additional_module_type_map = {'ConvFreezebn2d': ConvFreezebn2d,
                                      'ConvFreezebnReLU2d': ConvFreezebnReLU2d}
        extra_quantizer_dict_ret = {}
        if additional_module_type is not None:
            additional_module_type = [additional_module_type_map[k] for k in additional_module_type]
            extra_quantizer_dict_ret['additional_module_type'] = tuple(additional_module_type)
        return extra_quantizer_dict_ret

    def quantize_model(self):
        from mqbench.prepare_by_platform import prepare_by_platform
        logger.info("prepare quantize model")
        deploy_backend = self.config['quant']['deploy_backend']
        prepare_args = self.config['quant'].get('prepare_args', {})
        prepare_custom_config_dict = {}
        extra_qconfig_dict = prepare_args.get('extra_qconfig_dict', {})
        prepare_custom_config_dict['extra_qconfig_dict'] = extra_qconfig_dict
        leaf_module = prepare_args.get('leaf_module')
        if leaf_module is not None:
            prepare_custom_config_dict['leaf_module'] = self.get_leaf_module(leaf_module)
        extra_quantizer_dict = prepare_args.get('extra_quantizer_dict')
        if extra_quantizer_dict is not None:
            prepare_custom_config_dict['extra_quantizer_dict'] = self.get_extra_quantizer_dict(extra_quantizer_dict)

        self.model.train().cuda()
        for mname in self.model_list:
            mod = getattr(self.model, mname)
            mod = prepare_by_platform(mod, self.backend_type[deploy_backend], prepare_custom_config_dict)
            setattr(self.model, mname, mod)
            if env.is_master():
                print(mod)

        self.model.train().cuda()

        if env.is_master():
            print(self.model)

        # exit(0)

    def calibrate(self):
        logger.info("calibrate model")
        self.model.eval().cuda()
        enable_calibration(self.model)
        for _ in range(self.config['quant']['cali_batch_size']):
            batch = self.get_batch('train')
            self.model(batch)
        enable_quantization(self.model)
        self.eval_ptq()
        self.save_ptq()

    def eval_ptq(self):
        self.model.eval()
        self.evaluate()

    def save_ptq(self):
        save_dict = self.get_dump_dict()
        save_dict['spacial_name'] = 'ptq_model'
        save_dict['qat'] = False
        if env.is_master():
            self.saver.save(**save_dict)

    def save(self, auto_save=None, lns=None):
        save_dict = self.get_dump_dict()
        if auto_save is not None:
            save_dict['auto_save'] = auto_save
        if lns is not None:
            save_dict['lns'] = lns
        save_dict['qat'] = True
        if env.is_master():
            self.saver.save(**save_dict)

    def train(self):
        if self.config['quant']['ptq_only']:
            return
        enable_quantization(self.model)
        self.model.train()
        for iter_idx in range(self.start_iter, self.max_iter):
            batch = self.get_batch('train')
            loss = self.forward_train(batch)
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
