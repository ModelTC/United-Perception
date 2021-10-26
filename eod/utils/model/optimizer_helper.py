"""
  These functions are mainly used in train_val.py
"""

# Standard Library
import copy

# Import from third library
import torch

# Import from local
from ..general.log_helper import default_logger as logger
from ..general.registry_factory import OPTIMIZER_REGISTRY


def build_cls_instance(module, cfg):
    """Build instance for given cls"""
    cls = getattr(module, cfg['type'])
    return cls(**cfg['kwargs'])


@OPTIMIZER_REGISTRY.register('base')
class BaseOptimizer(object):
    def __init__(self, cfg_optim, model):
        self.cfg_optim = cfg_optim
        self.model = model

    def get_trainable_params(self, cfg_optim=None):
        special_param_group = cfg_optim.get('special_param_group', [])
        trainable_params = [{"params": []}] + [{"params": [], "lr": spg['lr']} for spg in special_param_group]
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                gid = 0
                for i, spg in enumerate(special_param_group):
                    if spg['key'] in n:
                        gid = i + 1
                        break
                trainable_params[gid]["params"].append(p)
        return trainable_params

    def get_optimizer(self, cfg_optim):
        optimizer = build_cls_instance(torch.optim, cfg_optim)
        logger.info('build optimizer done')
        return optimizer

    def build_optimizer(self):
        cfg_optim = copy.deepcopy(self.cfg_optim)
        cfg_optim['kwargs']['params'] = self.get_trainable_params(cfg_optim)
        return self.get_optimizer(cfg_optim)
