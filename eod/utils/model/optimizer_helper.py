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


def fused_sgd_reload(self):
    """When the model weights updates, e.g. load pre-train weights, the params stored in FusedFP16SGD
    will be out-of-date, so we need to reload the new weights.
    Remember call optimizer.reload whenever model's weights updated in fp16 training!
    """
    # first, rollback master param to model params in  param_groups
    for i, param_group in enumerate(self.param_groups):
        fp16_params_this_group = copy.copy(self.fp16_groups[i])
        fp32_params_this_group = copy.copy(self.fp32_groups[i])
        fp32_from_fp16_params_this_group = copy.copy(self.fp32_from_fp16_groups[i])
        for j, param in enumerate(param_group['params']):
            if param.requires_grad:
                if len(fp32_params_this_group) > 0 and fp32_params_this_group[0] is param:
                    fp32_params_this_group.pop(0)
                else:
                    master_param = fp32_from_fp16_params_this_group.pop(0)
                    model_param = fp16_params_this_group.pop(0)
                    param_group['params'][j] = model_param
                    if master_param in self.state:
                        self.state[model_param] = self.state.pop(master_param)
    self._promote_fp16_params()


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
        optim_type = cfg_optim['type']
        if optim_type == 'FusedFP16SGD' or optim_type == 'FusedFP16AdamW':
            from spring import linklink
            if optim_type == 'FusedFP16SGD':
                linklink.optim.FusedFP16SGD.reload = fused_sgd_reload
            if optim_type == 'FusedFP16AdamW':
                linklink.optim.FusedFP16AdamW.reload = fused_sgd_reload
            optimizer = build_cls_instance(linklink.optim, cfg_optim)
        else:
            optimizer = build_cls_instance(torch.optim, cfg_optim)
        logger.info('build optimizer done')
        return optimizer

    def build_optimizer(self):
        cfg_optim = copy.deepcopy(self.cfg_optim)
        cfg_optim['kwargs']['params'] = self.get_trainable_params(cfg_optim)
        return self.get_optimizer(cfg_optim)
