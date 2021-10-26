import math
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR

from eod.utils.general.registry_factory import WARM_LR_REGISTRY, LR_REGISTRY, WARM_SCHEDULER_REGISTY
from eod.utils.model.lr_helper import BaseWarmScheduler


__all__ = ['YoloV5CosineLR', 'YoloV5LinearWarmUpLR', 'Yolov5WarmLRScheduler']


def cal_cos_lines(x, epochs, min_x):
    return ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - min_x) + min_x


@LR_REGISTRY.register('YoloV5CosineLR')
class YoloV5CosineLR(CosineAnnealingLR):
    def __init__(self, **kwargs):
        self.T_max = kwargs['T_max']
        self.data_size = kwargs["data_size"]
        self.min_lr_scale = kwargs["min_lr_scale"]
        kwargs.pop('data_size')
        kwargs.pop('min_lr_scale')
        super(YoloV5CosineLR, self).__init__(**kwargs)

    def get_lr(self):
        cur_epoch = self.last_epoch // self.data_size
        return [base_lr * cal_cos_lines(cur_epoch, self.T_max, self.min_lr_scale) for base_lr in self.base_lrs]


@WARM_LR_REGISTRY.register('yolov5linear')
class YoloV5LinearWarmUpLR(object):
    def __init__(self, warmup_iter, init_lr, target_lr,
                 warmup_total_epochs, warmup_group_init_lr,
                 warmup_weight_decay, warmup_weight_decay_range,
                 min_lr_scale, data_size):
        self.warmup_iter = warmup_iter
        self.target_lr = target_lr
        self.warmup_group_init_lr = warmup_group_init_lr
        self.warmup_weight_decay = warmup_weight_decay
        self.warmup_weight_decay_range = warmup_weight_decay_range
        self.warmup_epochs = warmup_iter // data_size
        self.total_epochs = warmup_total_epochs
        self.min_lr_scale = min_lr_scale
        self.data_size = data_size

    def get_lr(self, last_epoch, base_lrs, optimizer):
        cur_epoch = last_epoch // self.data_size
        # the momentum warmup only for yolov5
        if self.warmup_weight_decay:
            for x in optimizer.param_groups:
                if 'momentum' in x:
                    x['momentum'] = np.interp(last_epoch, [0, self.warmup_iter], self.warmup_weight_decay_range)
        return [np.interp(last_epoch, [0, self.warmup_iter],
                          [self.warmup_group_init_lr[j],
                           self.target_lr[j] * cal_cos_lines(cur_epoch, self.total_epochs, self.min_lr_scale)])
                for j in range(len(optimizer.param_groups))]


@WARM_SCHEDULER_REGISTY.register('yolov5')
class Yolov5WarmLRScheduler(BaseWarmScheduler):
    def get_warmup_cfg(self, warm_kwargs, warmup_type):
        warmup_cfg = {
            'type': warmup_type,
            'kwargs': warm_kwargs
        }
        warmup_cfg['kwargs'].update({'warmup_total_epochs': self.cfg_scheduler.get('warmup_total_epochs', 300),
                                     'warmup_group_init_lr': self.cfg_scheduler.get('warmup_group_init_lr',
                                                                                    [0., 0., 0.1]),
                                     'warmup_weight_decay': self.cfg_scheduler.get('warmup_weight_decay', False),
                                     'warmup_weight_decay_range': self.cfg_scheduler.get('warmup_weight_decay_range',
                                                                                         [0.8, 0.937]),
                                     'min_lr_scale': self.cfg_scheduler.get('min_lr_scale', 0.2),
                                     'data_size': self.data_size})
        return warmup_cfg
