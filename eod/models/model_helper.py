# Standard Library
import copy
import importlib

# Import from third library
import torch
import torch.nn as nn
from collections import OrderedDict

from eod.utils.general.log_helper import default_logger as logger
from eod.utils.env.gene_env import to_device, patterns_match
from eod.utils.general.registry_factory import MODULE_ZOO_REGISTRY, MODEL_HELPER_REGISTRY


__all__ = ['ModelHelper']


@MODEL_HELPER_REGISTRY.register('base')
class ModelHelper(nn.Module):
    """Build model from cfg"""

    def __init__(self, cfg, deploy=False):
        super(ModelHelper, self).__init__()

        if 'net' in cfg:
            cfg = cfg['net']

        functional_kwargs, cfg = self.split_cfg(cfg)

        self.model_cfg = copy.deepcopy(cfg)
        for cfg_subnet in cfg:
            mname = cfg_subnet['name']
            kwargs = cfg_subnet['kwargs']
            mtype = cfg_subnet['type']
            if cfg_subnet.get('prev', None) is not None:
                if hasattr(self, cfg_subnet['prev']):
                    prev_module = getattr(self, cfg_subnet['prev'])
                    if hasattr(prev_module, "get_outplanes"):
                        kwargs['inplanes'] = prev_module.get_outplanes()
            module = self.build(mtype, kwargs)
            self.add_module(mname, module)
        self.dtype = torch.float32
        self.device = torch.device('cpu')
        self.deploy = deploy

        self._init_functional(**functional_kwargs)
        self.train()

    def split_cfg(self, cfg_list):
        functional_cfg = [cfg for cfg in cfg_list if cfg['type'] == 'self']
        if len(functional_cfg) > 0:
            assert len(functional_cfg) == 1
            functional_kwargs = functional_cfg[0]['kwargs']
        else:
            functional_kwargs = {}
        model_cfg = [cfg for cfg in cfg_list if cfg['type'] != 'self']
        return functional_kwargs, model_cfg

    def _init_functional(self, freeze_patterns=[]):
        """
        """
        self.freeze_patterns = freeze_patterns
        self.freeze_parameters()

    def build(self, mtype, kwargs):
        # For traditional uses
        if mtype.find('.') >= 0:
            module_name, cls_name = mtype.rsplit('.', 1)
            module = importlib.import_module(module_name)
            cls = getattr(module, cls_name)
            return cls(**kwargs)
        else:  # For usage that models are registered by MODULE_ZOO_REGISTRY
            cfg = {'type': mtype, 'kwargs': kwargs}
            return MODULE_ZOO_REGISTRY.build(cfg)

    def float(self):
        self.dtype = torch.float32
        return super(ModelHelper, self).float()

    def half(self):
        self.dtype = torch.float16
        return super(ModelHelper, self).half()

    def double(self):
        self.dtype = torch.float64
        return super(ModelHelper, self).double()

    def cuda(self):
        self.device = torch.device('cuda')
        return super(ModelHelper, self).cuda()

    def cpu(self):
        self.device = torch.device('cpu')
        return super(ModelHelper, self).cpu()

    def forward(self, input):
        for submodule in self.children():
            output = submodule(input)
            input.update(output)
        if not self.deploy:
            return input
        else:
            return input['preds']

    def load(self, other_state_dict, strict=False):
        """
        1. load resume model or pretained detection model
        2. load pretrained clssification model
        """
        logger.info("Try to load the whole resume model or pretrained detection model...")
        other_state_dict.pop('model_cfg', None)
        model_keys = self.state_dict().keys()
        other_keys = other_state_dict.keys()
        shared_keys, unexpected_keys, missing_keys \
            = self.check_keys(model_keys, other_keys, 'model')

        # check shared_keys size
        new_state_dict = self.check_share_keys_size(self.state_dict(), other_state_dict, shared_keys, model_keys)
        # get share keys info from new_state_dict
        shared_keys, _, missing_keys \
            = self.check_keys(model_keys, new_state_dict.keys(), 'model')
        self.load_state_dict(new_state_dict, strict=strict)

        num_share_keys = len(shared_keys)
        if num_share_keys == 0:
            logger.info(
                'Failed to load the whole detection model directly, '
                'trying to load each part seperately...'
            )
            for mname, module in self.named_children():
                module_keys = module.state_dict().keys()
                other_keys = other_state_dict.keys()
                # check and display info module by module
                shared_keys, unexpected_keys, missing_keys, \
                    = self.check_keys(module_keys, other_keys, mname)
                module_ns_dict = self.check_share_keys_size(
                    module.state_dict(), other_state_dict, shared_keys, module_keys)
                shared_keys, _, missing_keys \
                    = self.check_keys(module_keys, module_ns_dict.keys(), mname)

                module.load_state_dict(module_ns_dict, strict=strict)

                self.display_info(mname, shared_keys, unexpected_keys, missing_keys)
                num_share_keys += len(shared_keys)
        else:
            self.display_info("model", shared_keys, unexpected_keys, missing_keys)
        return num_share_keys

    def check_keys(self, own_keys, other_keys, own_name):
        own_keys = set(own_keys)
        other_keys = set(other_keys)
        shared_keys = own_keys & other_keys
        unexpected_keys = other_keys - own_keys
        missing_keys = own_keys - other_keys
        return shared_keys, unexpected_keys, missing_keys

    def check_share_keys_size(self, model_state_dict, other_state_dict, shared_keys, model_keys):
        new_state_dict = OrderedDict()
        for k in shared_keys:
            if k in model_keys:
                if model_state_dict[k].shape == other_state_dict[k].shape:
                    new_state_dict[k] = other_state_dict[k]
        return new_state_dict

    def display_info(self, mname, shared_keys, unexpected_keys, missing_keys):
        info = "Loading {}:{} shared keys, {} unexpected keys, {} missing keys.".format(
            mname, len(shared_keys), len(unexpected_keys), len(missing_keys))

        if len(missing_keys) > 0:
            info += "\nmissing keys are as follows:\n    {}".format("\n    ".join(missing_keys))
        logger.info(info)

    def state_dict(self, destination=None, prefix='', keep_vars=False, model_cfg=False):
        destination = super(ModelHelper, self).state_dict(destination, prefix, keep_vars)
        if model_cfg:
            destination['model_cfg'] = self.model_cfg
        return destination

    def load_state_dict(self, state_dict, strict=True):
        state_dict.pop('model_cfg', None)
        super(ModelHelper, self).load_state_dict(state_dict, strict)

    @classmethod
    def from_checkpoint(cls, state_dict):
        model_cfg = state_dict.pop('model_cfg')
        model = cls(model_cfg)
        model.load_state_dict(state_dict, True)
        return model

    def freeze_parameters(self):
        for name, p in self.named_parameters():
            if patterns_match(self.freeze_patterns, name):
                p.requires_grad = False

    def train(self, mode=True):
        self.training = mode
        for name, m in self.named_modules():
            if name == "":  # self
                continue
            if patterns_match(self.freeze_patterns, name):
                m.eval()
            else:
                m.train(mode)
        return self
