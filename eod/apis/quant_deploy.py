import copy
import torch
from eod.utils.general.log_helper import default_logger as logger
from eod.utils.general.cfg_helper import merge_opts_into_cfg
from eod.utils.general.registry_factory import MODEL_HELPER_REGISTRY
from eod.utils.general.registry_factory import DEPLOY_REGISTRY


__all__ = ['QuantDeploy']


@DEPLOY_REGISTRY.register('quant')
class QuantDeploy(object):
    def __init__(self, config, work_dir='./'):
        config = copy.deepcopy(config)
        self.config = self.upgrade_config(config)
        self.set_default_cfg()
        self.device = self.config['runtime']['device']
        self.fp16 = False
        self.ckpt_path = self.config['args']['ckpt']
        self.input_shape = [int(i) for i in self.config['args']['input_shape'].split(' ')]
        self.build()

    def build(self):
        self.build_model()
        self.quantize_model()

    def set_default_cfg(self):
        self.config['runtime'] = self.config.setdefault('runtime', {})
        self.config['runtime']['rank_init'] = self.config['runtime'].setdefault('rank_init', True)
        self.config['runtime']['random_seed'] = self.config['runtime'].setdefault('random_seed', 131)
        self.config['runtime']['aligned'] = self.config['runtime'].setdefault('aligned', False)
        self.config['runtime']['iter_base'] = self.config['runtime'].setdefault('iter_base', False)
        self.config['runtime']['device'] = self.config['runtime'].setdefault('device', 'cuda')
        self.config['runtime']['async_norm'] = self.config['runtime'].setdefault('async_norm', False)
        self.config['runtime']['special_bn_init'] = self.config['runtime'].setdefault('special_bn_init', False)
        model_helper_cfg = self.config['runtime'].get('model_helper', {})
        model_helper_cfg['type'] = model_helper_cfg.get('type', 'base')
        model_helper_cfg['kwargs'] = model_helper_cfg.get('kwargs', {})
        self.config['runtime']['model_helper'] = model_helper_cfg

    def upgrade_config(self, config):
        opts = config.get('args', {}).get('opts', [])
        config = merge_opts_into_cfg(opts, config)
        return config

    def split_cfg(self):
        net_cfg = self.config['net'][:-1]
        post_cfg = [self.config['net'][-1]]
        return net_cfg, post_cfg

    def special_bn_init(self):
        for m in self.model.modules():
            if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.SyncBatchNorm):
                m.eps = 1e-3
                m.momentum = 0.03

    def build_model(self):
        model_helper_type = self.config['runtime']['model_helper']['type']
        model_helper_kwargs = self.config['runtime']['model_helper']['kwargs']
        model_helper_ins = MODEL_HELPER_REGISTRY[model_helper_type]
        net_cfg, post_cfg = self.split_cfg()
        self.model = model_helper_ins(net_cfg, **model_helper_kwargs)
        self.post_process = model_helper_ins(post_cfg, **model_helper_kwargs)
        if self.device == 'cuda':
            self.model = self.model.cuda()
            self.post_process = self.post_process.cuda()
        if self.config['runtime']['special_bn_init']:
            self.special_bn_init()
        if self.fp16:
            self.model = self.model.half()

    @property
    def backend_type(self):
        from mqbench.prepare_by_platform import BackendType
        return {'tensorrt': BackendType.Tensorrt,
                'snpe': BackendType.SNPE,
                'vitis': BackendType.Vitis,
                'academic': BackendType.Academic}

    def get_leaf_module(self, leaf_module):
        from mqbench.nn.modules import FrozenBatchNorm2d
        from eod.tasks.det.plugins.yolov5.models.components import Space2Depth
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
        self.model.deploy = True
        self.model = prepare_by_platform(self.model,
                                         self.backend_type[deploy_backend], prepare_custom_config_dict)
        self.model.cuda()
        print(self.model)

    def load_checkpoint(self):
        def remove_prefix(state_dict, prefix):
            """Old style model is stored with all names of parameters share common prefix 'module.'"""
            f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
            return {f(key): value for key, value in state_dict.items()}

        ckpt = torch.load(self.ckpt_path)
        if 'ema' in ckpt and self.ema is not None:
            state_trained = ckpt['ema']['ema_state_dict']
        else:
            state_trained = ckpt['model']
        state_trained = remove_prefix(state_trained, 'module.')
        self.model.load_state_dict(state_trained)

    def deploy(self):
        self.load_checkpoint()
        logger.info("deploy model")
        from mqbench.convert_deploy import convert_deploy
        deploy_backend = self.config['quant']['deploy_backend']
        self.model.eval()
        image = torch.randn(self.input_shape).cuda()
        convert_deploy(self.model, self.backend_type[deploy_backend], dummy_input={'image': image})
