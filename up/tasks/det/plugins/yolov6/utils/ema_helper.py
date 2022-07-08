import torch
import math

from up.utils.general.registry_factory import EMA_REGISTRY
from up.utils.model.ema_helper import EMA

__all__ = ["ExpEMAV6"]


@EMA_REGISTRY.register('exp_v6')
class ExpEMAV6(EMA):
    def __init__(self, model, decay, inner_T=1, warmup=1):
        super(ExpEMAV6, self).__init__(model, decay, inner_T, warmup)

    def step(self, model, curr_step=None):
        if curr_step is None:
            decay = self.decay
        else:
            curr_step += 1
            def df(x): return self.decay * (1 - math.exp(-x / 2000)) # noqa
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
