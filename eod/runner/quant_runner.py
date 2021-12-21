from .base_runner import BaseRunner
from eod.utils.general.registry_factory import MODEL_HELPER_REGISTRY, RUNNER_REGISTRY
from eod.utils.env.gene_env import get_env_info
from eod.utils.general.log_helper import default_logger as logger
from eod.utils.general.cfg_helper import format_cfg


__all__ = ['QuantRunner']


@RUNNER_REGISTRY.register("quant")
class QuantRunner(BaseRunner):
    def __init__(self, config, work_dir='./', training=True):
        super(QuantRunner, self).__init__(config, work_dir, training)

    def split_cfg(self):
        net_cfg = self.config['net'][:-1]
        post_cfg = [self.config['net'][-1]]
        return net_cfg, post_cfg

    def build(self):
        self.build_env()
        self.get_git_version()
        self.build_dataloaders()
        self.build_model()
        # self.build_trainer()
        self.build_ema()
        self.build_saver()
        self.build_hooks()
        self.resume_model()
        if self.training:
            self.quantize_model()
            self.calibrate()
            self.build_trainer()
        self.prepare_dist_model()
        get_env_info()
        self.save_running_cfg()
        if not self.args.get('no_running_config', False):
            logger.info('Running with config:\n{}'.format(format_cfg(self.config)))

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

    def forward_model(self, batch):
        output = self.model(batch)
        if self.model.training:
            self.post_process.train()
        else:
            self.post_process.eval()
        output = self.post_process(output)
        return output

    @property
    def backend_type(self):
        from mqbench.prepare_by_platform import BackendType
        return {'tensorrt': BackendType.Tensorrt,
                'snpe': BackendType.SNPE,
                'academic': BackendType.Academic}

    def quantize_model(self):
        from mqbench.prepare_by_platform import prepare_by_platform
        from eod.utils.model.bn_helper import FrozenBatchNorm2d
        from eod.tasks.det.plugins.yolov5.models.components import Space2Depth
        logger.info("prepare quantize model")
        deploy_backend = self.config['quant']['deploy_backend']
        prepare_args = self.config['quant'].get('prepare_args', {})
        self.model.train().cuda()
        self.model = prepare_by_platform(self.model,
                                         self.backend_type[deploy_backend], prepare_args,
                                         leaf_modules=[FrozenBatchNorm2d, Space2Depth],
                                         custombn=[FrozenBatchNorm2d])
        print(self.model)

    def calibrate(self):
        logger.info("calibrate model")
        from mqbench.utils.state import enable_calibration, enable_quantization
        self.model.eval().cuda()
        enable_calibration(self.model)
        for _ in range(self.config['quant']['cali_batch_size']):
            batch = self.get_batch('train')
            self.model(batch)
        enable_quantization(self.model)
        self.model.train()

    def train(self):
        self.model.cuda().eval()
        self.evaluate()
        self.model.cuda().train()
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
                    from eod.utils.env.dist_helper import all_reduce_norm
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
