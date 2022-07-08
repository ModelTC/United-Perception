# Import from third library
import math
import numpy as np

from torch.optim.lr_scheduler import LambdaLR

from up.utils.general.registry_factory import WARM_LR_REGISTRY, WARM_SCHEDULER_REGISTY, LR_SCHEDULER_REGISTY
from up.utils.model.lr_helper import BaseLRScheduler, BaseWarmScheduler, LR_REGISTRY


__all__ = ['YoloV6CosineLR', 'YoloV6WarmScheduler', 'YoloV6CosWarmUpLR', "YoloV6LRScheduler"]


@LR_REGISTRY.register('YoloV6CosineLR')
class YoloV6CosineLR(LambdaLR):
    def __init__(self, **kwargs):
        self.data_size = kwargs.pop('data_size')
        self.warmup_target_lr = kwargs.pop('warmup_target_lr')
        cos_lr = kwargs.pop('cos_lr')
        total_epochs = kwargs.pop('total_epochs')
        def lambda_func(x): return ((1 - math.cos(x * math.pi / total_epochs)) / 2) * (cos_lr - 1) + 1 # noqa
        kwargs.update({'lr_lambda': lambda_func})
        super(YoloV6CosineLR, self).__init__(**kwargs)

    def get_lr(self):
        return [self.warmup_target_lr * lmbda((self.last_epoch // self.data_size) - 1)
                for lmbda, base_lr in zip(self.lr_lambdas, self.base_lrs)]


@WARM_SCHEDULER_REGISTY.register("yolov6_warm_scheduler")
class YoloV6WarmScheduler(BaseWarmScheduler):
    def update_warmup_kwargs(self):
        warm_kwargs = {
            "target_lr": self.target_lr[0],
            "data_size": self.data_size,
            "warmup_epochs": self.warmup_epochs,
            "total_epochs": self.cfg_scheduler.get('total_epochs', 400),
            "warmup_bias_lr": self.cfg_scheduler.get('warmup_bias_lr', 0),
            "warmup_momentum": self.cfg_scheduler.get('warmup_momentum', 0.8),
            "target_momentum": self.cfg_scheduler.get('target_momentum', 0.937),
        }
        return warm_kwargs


@WARM_LR_REGISTRY.register('yolov6_cos')
class YoloV6CosWarmUpLR(object):
    def __init__(self,
                 target_lr,
                 data_size,
                 warmup_epochs,
                 total_epochs,
                 warmup_bias_lr,
                 warmup_momentum,
                 target_momentum,):
        self.target_lr = target_lr
        self.data_size = data_size
        self.warmup_iter = max(round(warmup_epochs * data_size), 1000)
        self.warmup_epochs = warmup_epochs
        self.lambda_func = lambda x: ((1 - math.cos(x * math.pi / total_epochs)) / 2) * (target_lr - 1) + 1
        self.warmup_bias_lr = warmup_bias_lr
        self.warmup_momentum = warmup_momentum
        self.target_momentum = target_momentum

    def get_lr(self, last_epoch, base_lrs, optimizer):
        curr_epoch = last_epoch // self.data_size

        # totle batchsize should greater than 32
        lr_list = []

        for k, param in enumerate(optimizer.param_groups):
            warmup_bias_lr = self.warmup_bias_lr if k == 2 else 0.0
            param['lr'] = np.interp(last_epoch,
                                    [0, self.warmup_iter],
                                    [warmup_bias_lr, self.target_lr * self.lambda_func(curr_epoch)])
            lr_list.append(param['lr'])
            if 'momentum' in param:
                param['momentum'] = np.interp(last_epoch,
                                              [0, self.warmup_iter],
                                              [self.warmup_momentum, self.target_momentum])
        return lr_list


@LR_SCHEDULER_REGISTY.register('yolov6_base')
class YoloV6LRScheduler(BaseLRScheduler):
    def build_scheduler(self):
        standard_scheduler_class = LR_REGISTRY.get(self.cfg_scheduler['type'])
        self.update_scheduler_kwargs()
        warmup_register_type = self.cfg_scheduler.get('warmup_register_type', 'base')
        warmup_register_ins = WARM_SCHEDULER_REGISTY[warmup_register_type](self.cfg_scheduler,
                                                                           self.optimizer,
                                                                           self.data_size,
                                                                           self.lr_scale)
        warmup_scheduler, warmup_initial_lr = warmup_register_ins.build_warmup_scheduler()

        class ChainIterLR(standard_scheduler_class):
            """Unified scheduler that chains warmup scheduler and standard scheduler
            """

            def __init__(self, *args, **kwargs):
                super(ChainIterLR, self).__init__(*args, **kwargs)

            def get_lr(self):
                if self.last_iter <= warmup_scheduler.warmup_iter:
                    self.base_lrs = warmup_initial_lr
                    return warmup_scheduler.get_lr(self.last_iter, self.base_lrs, self.optimizer)
                else:
                    return super(ChainIterLR, self).get_lr()

            @property
            def last_iter(self):
                return self.last_epoch

        if hasattr(standard_scheduler_class, 'warmup_iter') and not self.align:
            self.cfg_scheduler['kwargs']['warmup_iter'] = warmup_scheduler.warmup_iter
        self.cfg_scheduler['kwargs']['warmup_target_lr'] = warmup_scheduler.target_lr

        return ChainIterLR(**self.cfg_scheduler['kwargs'])

    def update_scheduler_kwargs(self):
        self.cfg_scheduler['total_epochs'] = self.cfg_scheduler['kwargs']['T_max']
        self.cfg_scheduler['kwargs']['optimizer'] = self.optimizer
        self.cfg_scheduler['kwargs']['data_size'] = self.data_size
        self.cfg_scheduler['kwargs']['total_epochs'] = self.cfg_scheduler['kwargs'].pop('T_max')
