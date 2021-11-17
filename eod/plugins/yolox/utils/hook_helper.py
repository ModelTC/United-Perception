from eod.utils.general.registry_factory import HOOK_REGISTRY
from eod.utils.general.hook_helper import Hook
from eod.utils.env.dist_helper import env
from eod.utils.general.log_helper import default_logger as logger


__all__ = ['YoloxNoaug']


@HOOK_REGISTRY.register('yolox_noaug')
class YoloxNoaug(Hook):
    def __init__(self, runner, no_aug_epoch=15, max_epoch=300, transformer=[], test_freq=1, save_freq=1):
        super(YoloxNoaug, self).__init__(runner)
        self.no_aug_epoch = no_aug_epoch
        self.max_epoch = max_epoch
        self.transformer = transformer
        self.flag = False
        self.test_freq = test_freq
        self.save_freq = save_freq

    def before_forward(self, cur_iter, input):
        runner = self.runner_ref()
        if cur_iter >= runner.data_loaders['train'].get_epoch_size() * (self.max_epoch - self.no_aug_epoch):
            if not self.flag:
                logger.info(f"rebuild dataset transformer cfg {self.transformer}")
                runner.config['dataset']['train']['dataset']['kwargs']['transformer'] = self.transformer
                del runner.data_loaders, runner.data_iterators['train']
                import gc
                gc.collect()
                if not hasattr(self, 'data_iterators'):
                    runner.data_iterators = {}
                logger.info("rebuild dataloader")
                runner.build_dataloaders()
                runner.data_iterators['train'] = iter(runner.data_loaders["train"])
                try:
                    if env.world_size > 1:
                        runner.model.module.yolox_post.use_l1 = True
                    else:
                        runner.model.yolox_post.use_l1 = True
                except:  # noqa
                    pass
                runner.test_freq = self.test_freq
                runner.save_freq = self.save_freq
                self.flag = True
