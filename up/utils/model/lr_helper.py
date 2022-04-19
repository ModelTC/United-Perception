import copy

# Import from third library
import torch
import math
from torch.optim.lr_scheduler import MultiStepLR, StepLR, ReduceLROnPlateau, CosineAnnealingLR, _LRScheduler, LambdaLR
from ..general.registry_factory import WARM_LR_REGISTRY, LR_REGISTRY
from ..general.registry_factory import LR_SCHEDULER_REGISTY, WARM_SCHEDULER_REGISTY


@LR_REGISTRY.register('MultiStepLR')
class _MultiStepLR(MultiStepLR):
    def __init__(self, **kwargs):
        kwargs['milestones'] = [int(e * kwargs["data_size"]) for e in kwargs['milestones']]
        kwargs.pop('data_size')
        super(_MultiStepLR, self).__init__(**kwargs)


@LR_REGISTRY.register('StepLR')
class _StepLR(StepLR):
    def __init__(self, **kwargs):
        kwargs['step_size'] = kwargs['step_size'] * kwargs["data_size"]
        kwargs.pop('data_size')
        super(_StepLR, self).__init__(**kwargs)


@LR_REGISTRY.register('ReduceLROnPlateau')
class _ReduceLROnPlateau(ReduceLROnPlateau):
    def __init__(self, **kwargs):
        kwargs['patience'] = kwargs['patience'] * kwargs["data_size"]
        kwargs.pop('data_size')
        super(_ReduceLROnPlateau, self).__init__(**kwargs)


@LR_REGISTRY.register('CosineAnnealingLR')
class _CosineAnnealingLR(CosineAnnealingLR):
    def __init__(self, **kwargs):
        self.warmup_iter = 0
        if 'warmup_iter' in kwargs:
            self.warmup_iter = kwargs.pop('warmup_iter')
        kwargs['T_max'] = kwargs['T_max'] * kwargs["data_size"]
        kwargs.pop('data_size')
        super(_CosineAnnealingLR, self).__init__(**kwargs)

    def get_lr(self):
        return [self.eta_min + (base_lr - self.eta_min)
                * (1 + math.cos(math.pi * (self.last_epoch - self.warmup_iter) / (self.T_max - self.warmup_iter))) / 2
                for base_lr in self.base_lrs]


@LR_REGISTRY.register('CosineLREpochScheduler')
class _CosineLREpochScheduler(CosineAnnealingLR):
    def __init__(self, **kwargs):
        self.warmup_iter = 0
        if 'warm_epoch' in kwargs:
            self.warm_epoch = kwargs.pop('warm_epoch')
        self.total_iter = kwargs['T_max'] * kwargs["data_size"]
        self.data_size = kwargs.pop('data_size')
        super(_CosineLREpochScheduler, self).__init__(**kwargs)

    def get_lr(self):
        last_epoch = self.last_epoch // self.data_size
        return [self.eta_min + (base_lr - self.eta_min)
                * (1 + math.cos(math.pi * (last_epoch - self.warm_epoch) / self.T_max)) / 2
                for base_lr in self.base_lrs]


@LR_REGISTRY.register('SimSiamScheduler')
class _SimSiamScheduler(LambdaLR):
    def __init__(self, **kwargs):
        self.T_max = kwargs.pop('T_max')
        if 'warm_epoch' in kwargs:
            self.warm_epoch = kwargs.pop('warm_epoch')
            self.T_max = self.T_max - self.warm_epoch
        self.data_size = kwargs.pop('data_size')
        lambda_backbone = lambda s: 0.5 * (1 + math.cos(math.pi * s / self.T_max))
        lambda_pred = lambda s: 1.0
        kwargs.update({'lr_lambda': [lambda_backbone, lambda_pred]})
        super(_SimSiamScheduler, self).__init__(**kwargs)

    def get_lr(self):
        last_epoch = self.last_epoch // self.data_size
        return [base_lr * lmbda(last_epoch - self.warm_epoch)
                for lmbda, base_lr in zip(self.lr_lambdas, self.base_lrs)]


@LR_REGISTRY.register('polylr')
class PolyLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, max_epoch=-1, max_iter=-1, power=1., min_lr=0., **kwargs):
        self.power = power
        if max_iter == -1:
            self.max_epoch = max_epoch * kwargs['data_size']
        else:
            self.max_epoch = max_iter
        self.min_lr = min_lr
        kwargs.pop('data_size')
        super(PolyLR, self).__init__(**kwargs)

    def get_lr(self):
        coeff = (1 - self.last_epoch / self.max_epoch)**self.power
        return [(base_lr - self.min_lr) * coeff + self.min_lr for base_lr in self.base_lrs]


@WARM_LR_REGISTRY.register('exp')
class ExponentialWarmUpLR(object):
    """Scheduler that update learning rate exponentially
    """

    def __init__(self, warmup_iter, init_lr, target_lr):
        lr_scale = [target_lr[i] / init_lr[i] for i in range(len(target_lr))]
        self.gamma = [lr_s**(1.0 / max(1, warmup_iter - 1)) for lr_s in lr_scale]
        self.warmup_iter = warmup_iter

    def get_lr(self, last_epoch, base_lrs, optimizer):
        return [base_lr * self.gamma[idx]**last_epoch for idx, base_lr in enumerate(base_lrs)]


@WARM_LR_REGISTRY.register('linear')
class LinearWarmUpLR(object):
    """Scheduler that update learning rate linearly
    """

    def __init__(self, warmup_iter, init_lr, target_lr):
        self.gamma = [(target_lr[i] - init_lr[i]) / max(1, warmup_iter - 1) for i in range(len(target_lr))]
        self.warmup_iter = warmup_iter

    def get_lr(self, last_epoch, base_lrs, optimizer):
        return [base_lr + self.gamma[idx] * last_epoch for idx, base_lr in enumerate(base_lrs)]


@WARM_SCHEDULER_REGISTY.register("base")
class BaseWarmScheduler(object):
    def __init__(self, cfg_scheduler, optimizer, data_size, lr_scale):
        self.cfg_scheduler = copy.deepcopy(cfg_scheduler)
        self.optimizer = optimizer
        self.data_size = data_size
        self.lr_scale = lr_scale
        self.warmup_epochs = self.cfg_scheduler.get('warmup_epochs', 0)
        self.warmup_iter = max(self.cfg_scheduler.get('warmup_iter', 0), int(self.warmup_epochs * self.data_size))
        self.warmup_ratio = self.cfg_scheduler.get('warmup_ratio', -1)

    def get_init_lr(self):
        init_lr = [group.get('initial_lr', group['lr']) for group in self.optimizer.param_groups]
        self.init_lr = init_lr

    def build_warmup_scheduler(self):
        self.get_init_lr()
        self.get_target_lr()
        self.get_warmup_initial_lr()
        warm_kwargs = self.update_warmup_kwargs()
        warmup_type = self.cfg_scheduler.get('warmup_type', 'exp')
        warmup_cfg = self.get_warmup_cfg(warm_kwargs, warmup_type)
        return WARM_LR_REGISTRY.build(warmup_cfg), self.warmup_initial_lr

    def get_warmup_initial_lr(self):
        if self.warmup_ratio < 0:
            warmup_initial_lr = self.init_lr
        else:
            warmup_initial_lr = [self.warmup_ratio * target_lr for target_lr in self.target_lr]
        self.warmup_initial_lr = warmup_initial_lr

    def update_warmup_kwargs(self):
        warm_kwargs = {
            "warmup_iter": self.warmup_iter,
            "init_lr": self.warmup_initial_lr,
            "target_lr": self.target_lr
        }
        return warm_kwargs

    def get_warmup_cfg(self, warm_kwargs, warmup_type):
        warmup_cfg = {
            'type': warmup_type,
            'kwargs': warm_kwargs
        }
        return warmup_cfg

    def get_target_lr(self):
        if self.warmup_iter > 0:
            target_lr = [init_lr * self.lr_scale for init_lr in self.init_lr]
        else:
            target_lr = self.init_lr
            for i in range(len(self.optimizer.param_groups)):
                self.optimizer.param_groups[i]['lr'] *= self.lr_scale
        self.target_lr = target_lr


@WARM_SCHEDULER_REGISTY.register('no_scale_lr')
class NoScaleWarmScheduler(BaseWarmScheduler):
    def get_target_lr(self):
        self.target_lr = self.init_lr


@LR_SCHEDULER_REGISTY.register('base')
class BaseLRScheduler(object):
    def __init__(self, cfg_scheduler, optimizer, data_size, lr_scale, align=False):
        self.cfg_scheduler = copy.deepcopy(cfg_scheduler)
        self.optimizer = optimizer
        self.data_size = data_size
        self.lr_scale = lr_scale
        self.align = align

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
                if self.last_iter < warmup_scheduler.warmup_iter:
                    self.base_lrs = warmup_initial_lr
                    return warmup_scheduler.get_lr(self.last_iter, self.base_lrs, self.optimizer)
                elif self.last_iter == warmup_scheduler.warmup_iter:
                    self.base_lrs = list(map(lambda group: group['lr'], self.optimizer.param_groups))
                    return super(ChainIterLR, self).get_lr()
                else:
                    return super(ChainIterLR, self).get_lr()

            @property
            def last_iter(self):
                return self.last_epoch
        if hasattr(standard_scheduler_class, 'warmup_iter') and not self.align:
            self.cfg_scheduler['kwargs']['warmup_iter'] = warmup_scheduler.warmup_iter

        return ChainIterLR(**self.cfg_scheduler['kwargs'])

    def update_scheduler_kwargs(self):
        self.cfg_scheduler['kwargs']['optimizer'] = self.optimizer
        self.cfg_scheduler['kwargs']['data_size'] = self.data_size
        if self.cfg_scheduler.get('type') == 'SWALR':
            total_lr = self.optimizer.param_groups[0]['lr'] * self.lr_scale
            self.cfg_scheduler['kwargs'].setdefault("total_lr", total_lr)


@LR_REGISTRY.register('OneCycleLR')
class OneCycleLR(_LRScheduler):
    def __init__(self,
                 max_epoch,
                 data_size,
                 optimizer,
                 max_lr,
                 total_steps=None,
                 epochs=None,
                 steps_per_epoch=None,
                 pct_start=0.3,
                 anneal_strategy='cos',
                 cycle_momentum=True,
                 base_momentum=0.85,
                 max_momentum=0.95,
                 div_factor=25.,
                 final_div_factor=1e4,
                 three_phase=False,
                 last_epoch=-1,
                 verbose=False):
        self.optimizer = optimizer
        self.total_steps = max_epoch * data_size
        self.last_epoch = last_epoch
        self._schedule_phases = [
            {
                'end_step': float(pct_start * self.total_steps) - 1,
                'start_lr': 'initial_lr',
                'end_lr': 'max_lr',
                'start_momentum': 'max_momentum',
                'end_momentum': 'base_momentum',
            },
            {
                'end_step': self.total_steps - 1,
                'start_lr': 'max_lr',
                'end_lr': 'min_lr',
                'start_momentum': 'base_momentum',
                'end_momentum': 'max_momentum',
            },
        ]

        # Initialize learning rate variables
        max_lrs = self._format_param('max_lr', self.optimizer, max_lr)
        if last_epoch == -1:
            for idx, group in enumerate(self.optimizer.param_groups):
                group['initial_lr'] = max_lrs[idx] / div_factor
                group['max_lr'] = max_lrs[idx]
                group['min_lr'] = group['initial_lr'] / final_div_factor

        # Initialize momentum variables
        self.cycle_momentum = cycle_momentum
        if self.cycle_momentum:
            if 'momentum' not in self.optimizer.defaults and 'betas' not in self.optimizer.defaults:
                raise ValueError('optimizer must support momentum with `cycle_momentum` option enabled')
            self.use_beta1 = 'betas' in self.optimizer.defaults
            max_momentums = self._format_param('max_momentum', optimizer, max_momentum)
            base_momentums = self._format_param('base_momentum', optimizer, base_momentum)
            if last_epoch == -1:
                for m_momentum, b_momentum, group in zip(max_momentums, base_momentums, optimizer.param_groups):
                    if self.use_beta1:
                        _, beta2 = group['betas']
                        group['betas'] = (m_momentum, beta2)
                    else:
                        group['momentum'] = m_momentum
                    group['max_momentum'] = m_momentum
                    group['base_momentum'] = b_momentum

        super(OneCycleLR, self).__init__(optimizer, last_epoch)

    def _format_param(self, name, optimizer, param):
        """Return correctly formatted lr/momentum for each param group."""
        if isinstance(param, (list, tuple)):
            if len(param) != len(optimizer.param_groups):
                raise ValueError("expected {} values for {}, got {}".format(
                    len(optimizer.param_groups), name, len(param)))
            return param
        else:
            return [param] * len(optimizer.param_groups)

    def get_lr(self):
        lrs = []
        step_num = self.last_epoch
        for group in self.optimizer.param_groups:
            start_step = 0
            for i, phase in enumerate(self._schedule_phases):
                end_step = phase['end_step']
                if step_num <= end_step or i == len(self._schedule_phases) - 1:
                    pct = (step_num - start_step) / (end_step - start_step)
                    cos_out = math.cos(math.pi * pct) + 1
                    computed_lr = group[phase['end_lr']] + \
                        (group[phase['start_lr']] - group[phase['end_lr']]) / 2.0 * cos_out
                    if self.cycle_momentum:
                        computed_momentum = group[phase['end_momentum']] + \
                            (group[phase['start_momentum']] - group[phase['end_momentum']]) / 2.0 * cos_out
                    break
                start_step = phase['end_step']

            lrs.append(computed_lr)
            if self.cycle_momentum:
                if self.use_beta1:
                    _, beta2 = group['betas']
                    group['betas'] = (computed_momentum, beta2)
                else:
                    group['momentum'] = computed_momentum

        return lrs


if __name__ == '__main__':
    import torchvision
    import os
    import sys
    model = torchvision.models.resnet18()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.1)
    cfg_scheduler = {'warmup_epochs': 1, 'type': 'MultiStepLR', 'kwargs': {'milestones': [10, 20], 'gamma': 0.1}}
    scheduler_ins = LR_SCHEDULER_REGISTY['base'](cfg_scheduler, optimizer, data_size=5, lr_scale=20)
    scheduler = scheduler_ins.build_scheduler()
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        state = torch.load(sys.argv[1])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
    start_iter = scheduler.last_iter
    for i in range(start_iter, 120):
        if i % 30 == 0:
            state = {'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}
            torch.save(state, f'iter{i}.pkl')
            print('saveing to:iter{}.pkl'.format(i))
        optimizer.step()
        scheduler.step()
        lr = [group['lr'] for group in optimizer.param_groups][0]
        print(f'iter:{i}, lr:{scheduler.get_lr()}, optimizer.lr:{lr}')
