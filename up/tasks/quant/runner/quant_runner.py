from up.runner.base_runner import BaseRunner
from up.utils.general.registry_factory import MODEL_HELPER_REGISTRY, RUNNER_REGISTRY
from up.utils.env.gene_env import get_env_info
from up.utils.env.dist_helper import env
from up.utils.general.log_helper import default_logger as logger
from up.utils.general.cfg_helper import format_cfg
from mqbench.utils.state import enable_quantization, enable_calibration_woquantization

__all__ = ['QuantRunner']


@RUNNER_REGISTRY.register("quant")
class QuantRunner(BaseRunner):
    def __init__(self, config, work_dir='./', training=True):
        self.do_calib = True
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
            if self.do_calib:
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
        if 'quant' not in self.ckpt:
            self.model.load_state_dict(add_prefix(self.ckpt['model']))

    def resume_model_from_quant(self):
        if 'quant' in self.ckpt:
            self.do_calib = False
            self.model.load_state_dict(self.ckpt['model'])
            if self.ckpt['quant'] == 'qat' and self.training and 'optimizer' in self.ckpt:
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

    @property
    def quant_type(self):
        if hasattr(self, '_quant_type') is False:
            self._quant_type = self.config['quant'].get('quant_type', 'calib_only')
        return self._quant_type

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
        from mqbench.fake_quantize.quantize_base import FakeQuantizeBase
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

        quant_tricks = self.config['quant'].get('tricks', {})
        special_layers = quant_tricks.get('special', None)
        if special_layers is not None:
            specila_bit = quant_tricks.get('bit', 8)
            f = lambda x: self.model_map[x.split('.')[0]] + '.' + x
            special_layers = [f(layer) for layer in special_layers]
            for name, mod in self.model.named_modules():
                if name in special_layers:
                    for _name, _mod in mod.named_modules():
                        if isinstance(_mod, FakeQuantizeBase):
                            self.set_bitwidth(_mod, specila_bit)
                            logger.info(f'set {name}.{_name} to {specila_bit} bit.')
        self.model.train().cuda()
        if env.is_master():
            print(self.model)

    def set_bitwidth(self, fake_quant, bit_width):
        if fake_quant.quant_min == 0:
            new_min = 0
            new_max = 2 ** bit_width - 1
        else:
            new_min = - 2 ** (bit_width - 1)
            new_max = 2 ** (bit_width - 1) - 1
        fake_quant.quant_min = new_min
        fake_quant.quant_max = new_max
        fake_quant.activation_post_process.quant_min = new_min
        fake_quant.activation_post_process.quant_max = new_max
        fake_quant.bitwidth = bit_width

    def calibrate(self):
        logger.info("calibrate model")
        self.model.eval().cuda()
        enable_calibration_woquantization(self.model, quantizer_type='act_fake_quant')
        for _ in range(self.config['quant']['cali_batch_size']):
            batch = self.get_batch('train')
            self.model(batch)
        enable_calibration_woquantization(self.model, quantizer_type='weight_fake_quant')
        for _ in range(1):
            batch = self.get_batch('train')
            self.model(batch)
        enable_quantization(self.model)
        self.sync_quant_params()
        logger.info('eval the calib model')
        self.eval_quant()
        self.save_calib()

    def sync_quant_params(self):
        logger.info('start quant params reduce')
        try:
            import spring.linklink as link
        except:   # noqa
            link = None
        reduce_list = ['scale', 'zero_point', 'min_val', 'max_val']
        reduced = []
        for n, tensor in self.model.named_buffers():
            for rd in reduce_list:
                if n.endswith(rd):
                    tensor.data = tensor.data / link.get_world_size()
                    link.allreduce(tensor.data)
                    reduced.append(n)
        for n, tensor in self.model.named_parameters():
            for rd in reduce_list:
                if n.endswith(rd):
                    tensor.data = tensor.data / link.get_world_size()
                    link.allreduce(tensor.data)
                    reduced.append(n)
        logger.info(f'sync quant params: {reduced}.')

    def eval_quant(self):
        self.model.eval()
        self.evaluate()

    def save_calib(self):
        save_dict = self.get_dump_dict()
        save_dict['spacial_name'] = 'calib_model'
        save_dict['quant'] = 'calib'
        if env.is_master():
            self.saver.save(**save_dict)

    def save_ptq(self):
        save_dict = self.get_dump_dict()
        save_dict['spacial_name'] = 'ptq_model'
        save_dict['quant'] = 'ptq'
        if env.is_master():
            self.saver.save(**save_dict)

    def save_qat(self, auto_save=None, lns=None):
        save_dict = self.get_dump_dict()
        if auto_save is not None:
            save_dict['auto_save'] = auto_save
        if lns is not None:
            save_dict['lns'] = lns
        save_dict['quant'] = 'qat'
        if env.is_master():
            self.saver.save(**save_dict)

    def save_epoch_ckpt(self, iter_idx):
        cur_iter = iter_idx + 1
        epoch_size = self.train_epoch_size
        if self.iter_base:
            epoch_size = 1
        assert not self.iter_base
        if cur_iter % epoch_size == 0 and iter_idx > self.start_iter:
            self.save_qat(auto_save='latest', lns=False)

    def train_qat(self):
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
                    self.save_qat()
            self.lr_scheduler.step()

    def train_ptq(self):
        logger.info('building data for adaround-like ptq.')
        cali_data = []
        for _ in range(self.config['quant']['cali_batch_size']):
            batch = self.get_batch('train')
            cali_data.append(batch)
        from mqbench.advanced_ptq import ptq_reconstruction
        from easydict import EasyDict
        self.model.train()
        self.model = ptq_reconstruction(self.model, cali_data, EasyDict(self.config['quant']['ptq']), self.model_list)
        self.save_ptq()
        self.eval_quant()

    def train(self):
        if self.quant_type == 'calib_only':
            return
        elif self.quant_type == 'qat':
            self.train_qat()
        elif self.quant_type == 'ptq':
            self.train_ptq()
        else:
            raise NotImplementedError('Not supported quant training method.')
