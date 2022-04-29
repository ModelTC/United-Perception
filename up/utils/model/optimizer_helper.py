"""
  These functions are mainly used in train_val.py
"""

# Standard Library
import copy

# Import from third library
import torch

# Import from local
from up.utils.model import optim
from ..general.log_helper import default_logger as logger
from ..general.registry_factory import OPTIMIZER_REGISTRY
from up.utils.model.utils import get_layer_id_for_vit


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
        pconfig = cfg_optim.get('pconfig', None)
        if pconfig is not None:
            return self.param_group_all(self.model, pconfig)
        trainable_params = [{"params": []}]
        for spg in special_param_group:
            spg.update({'params': []})
            trainable_params.append(spg)
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                gid = 0
                for i, spg in enumerate(special_param_group):
                    if spg['key'] in n:
                        gid = i + 1
                        break
                trainable_params[gid]["params"].append(p)
        return trainable_params

    def param_group_all(self, model, config):
        pgroup = {}
        param_groups_dict = {}
        nodecay = config.pop('nodecay', [])
        layer_decay = config.pop('layer_decay', None)
        key2type = {
            'bn_b': torch.nn.BatchNorm2d,
            'bn_w': torch.nn.BatchNorm2d,
            'ln_b': torch.nn.LayerNorm,
            'ln_w': torch.nn.LayerNorm,
            'linear_w': torch.nn.Linear,
            'linear_b': torch.nn.Linear
        }
        # Handel Layer Decay
        if layer_decay is not None:
            if layer_decay['type'] == 'vit_base':
                num_layers = model.backbone.num_layers
                layer_scales = list(layer_decay['value'] ** (num_layers - i) for i in range(num_layers + 1))
            else:
                raise NotImplementedError(f"Not support for layer decay type: {layer_decay['type']}")

        for group_keys in config.keys():
            if group_keys not in pgroup:
                pgroup[group_keys] = []

        for name, m in model.named_modules():
            for group_keys in config.keys():
                if isinstance(m, key2type[group_keys]):
                    if config[group_keys]['type'] == 'bias':
                        pgroup[group_keys].append(m.bias)
                    elif config[group_keys]['type'] == 'weight':
                        pgroup[group_keys].append(m.weight)

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            param_groups_dict[p] = {}
            if layer_decay is not None:
                layer_id = get_layer_id_for_vit(name, num_layers)
                param_groups_dict[p].update({'lr': layer_scales[layer_id] * layer_decay['base_lr']})

            for key_nodecay in nodecay:
                if key_nodecay == 'ndim_is1':
                    if p.ndim == 1:
                        param_groups_dict[p].update({'weight_decay': 0.0})
                        break
                elif key_nodecay in name:
                    param_groups_dict[p].update({'weight_decay': 0.0})
                    break

        for ptype in pgroup.keys():
            if ptype in config.keys():
                for p in pgroup[ptype]:
                    param_groups_dict[p].update(dict(config[ptype]['kwargs']))

        # Translate param_groups_dict back to param_groups which can be used in torch.optim
        param_groups = []
        for p, config in param_groups_dict.items():
            param_groups += [{'params': p, **config}]
        return param_groups

    def get_optimizer(self, cfg_optim):
        optim_type = cfg_optim['type']
        if optim_type == 'FusedFP16SGD' or optim_type == 'FusedFP16AdamW':
            from spring import linklink
            if optim_type == 'FusedFP16SGD':
                linklink.optim.FusedFP16SGD.reload = fused_sgd_reload
            if optim_type == 'FusedFP16AdamW':
                linklink.optim.FusedFP16AdamW.reload = fused_sgd_reload
            optimizer = build_cls_instance(linklink.optim, cfg_optim)
        elif optim_type == 'LARS' or optim_type == 'LAMB':
            optimizer = build_cls_instance(optim, cfg_optim)
        else:
            optimizer = build_cls_instance(torch.optim, cfg_optim)
        logger.info('build optimizer done')
        return optimizer

    def build_optimizer(self):
        cfg_optim = copy.deepcopy(self.cfg_optim)
        cfg_optim['kwargs']['params'] = self.get_trainable_params(cfg_optim)
        return self.get_optimizer(cfg_optim)
