import torch
import math
from copy import deepcopy
from ..general.log_helper import default_logger as logger
from ..general.registry_factory import EMA_REGISTRY


@EMA_REGISTRY.register("linear")
class EMA(object):
    def __init__(self, model, decay, inner_T=1, warmup=1):
        self.model = deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.inner_T = inner_T
        self.decay = decay
        self.warmup = warmup
        if self.inner_T > 1:
            self.decay = self.decay ** self.inner_T
            logger.info('EMA: effective decay={}'.format(self.decay))
        self.ema_state_dict = self.model.state_dict()

    def step(self, model, curr_step=None):
        with torch.no_grad():
            if curr_step is None:
                decay = self.decay
            else:
                decay = min(self.decay, (1 + curr_step) / (self.warmup + curr_step))
            if curr_step % self.inner_T != 0:
                return
            msd = model.state_dict()
            for k, v in self.model.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= decay
                    v += (1.0 - decay) * msd[k].detach()

    def state_dict(self):
        state = {'ema_state_dict': self.model.state_dict()}
        state['decay'] = self.decay
        state['warmup'] = self.warmup
        state['inner_T'] = self.inner_T
        return state

    def load_state_dict(self, state):
        self.model.load(state['ema_state_dict'])
        self.decay = state['decay']
        self.inner_T = state['inner_T']
        self.warmup = state['warmup']


@EMA_REGISTRY.register('exp')
class ExpEMA(EMA):
    def __init__(self, model, decay, inner_T=1, warmup=1):
        super(ExpEMA, self).__init__(model, decay, inner_T, warmup)

    def step(self, model, curr_step=None):
        if curr_step is None:
            decay = self.decay
        else:
            df = lambda x: self.decay * (1 - math.exp(-x / 2000))
            decay = df(curr_step)
        with torch.no_grad():
            if hasattr(model, 'module'):
                msd = model.module.state_dict()
            else:
                msd = model.state_dict()  # model state_dict
            for k, v in self.model.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= decay
                    v += (1. - decay) * msd[k].detach()
