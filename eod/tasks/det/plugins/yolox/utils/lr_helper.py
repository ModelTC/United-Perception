# Import from third library
import math
from torch.optim.lr_scheduler import CosineAnnealingLR

from eod.utils.general.registry_factory import WARM_LR_REGISTRY, LR_REGISTRY, LR_SCHEDULER_REGISTY
from eod.utils.model.lr_helper import BaseLRScheduler


__all__ = ['YoloXCosineLR', 'YoloXCosWarmUpLR', 'YoloXLRScheduler']


@LR_REGISTRY.register('YoloXCosineLR')
class YoloXCosineLR(CosineAnnealingLR):
    def __init__(self, **kwargs):
        self.min_lr_scale = kwargs["min_lr_scale"]
        self.no_aug_iters = kwargs['no_aug_iters']
        self.total_iters = kwargs['total_iters']
        self.warmup_total_iters = kwargs['warmup_total_iters']
        kwargs.pop('min_lr_scale')
        kwargs.pop('no_aug_epoch')
        kwargs.pop('no_aug_iters')
        kwargs.pop('total_iters')
        kwargs.pop('warmup_total_iters')
        super(YoloXCosineLR, self).__init__(**kwargs)

    def calculate_lr(self, lr):
        min_lr = lr * self.min_lr_scale
        if self.last_epoch >= self.total_iters - self.no_aug_iters:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(
                math.pi * (self.last_epoch - self.warmup_total_iters)
                / (self.total_iters - self.warmup_total_iters - self.no_aug_iters)
            )
            )
        return lr

    def get_lr(self):
        return [self.calculate_lr(lr) for lr in self.base_lrs]


@WARM_LR_REGISTRY.register('yolox_cos')
class YoloXCosWarmUpLR(object):
    def __init__(self, warmup_iter, init_lr, target_lr, warmup_lr_start=0):
        self.warmup_lr_start = warmup_lr_start
        self.warmup_iter = warmup_iter
        self.target_lr = target_lr

    def get_lr(self, last_epoch, base_lrs, optimizer):
        return [(self.target_lr[idx] - self.warmup_lr_start) * pow(
                last_epoch / float(self.warmup_iter), 2
                ) + self.warmup_lr_start for idx in range(len(self.target_lr))]


@LR_SCHEDULER_REGISTY.register('yolox_base')
class YoloXLRScheduler(BaseLRScheduler):
    def update_scheduler_kwargs(self):
        self.cfg_scheduler['kwargs']['optimizer'] = self.optimizer
        self.cfg_scheduler['kwargs']['warmup_total_iters'] = max(
            self.cfg_scheduler.get(
                'warmup_iter', 0), self.cfg_scheduler.get(
                'warmup_epochs', 0) * self.data_size)
        self.cfg_scheduler['kwargs']['no_aug_iters'] = self.cfg_scheduler['kwargs'].get(
            'no_aug_epoch', 0) * self.data_size
        self.cfg_scheduler['kwargs']['total_iters'] = self.cfg_scheduler['kwargs'].get('T_max', 0) * self.data_size
