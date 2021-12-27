import torch.nn as nn
from eod.utils.model.optimizer_helper import BaseOptimizer
from eod.utils.general.registry_factory import OPTIMIZER_REGISTRY


__all__ = ['Yolov5ExceptOptimizer']


@OPTIMIZER_REGISTRY.register('yolov5_except')
class Yolov5ExceptOptimizer(BaseOptimizer):
    def get_trainable_params(self, cfg_optim):
        weight_decay = cfg_optim['kwargs']['weight_decay']
        except_keys = cfg_optim['kwargs'].pop('except_keys', [])
        trainable_params = [{"params": []}, {"params": []}, {"params": []}]
        network_keys = []
        for k, v in self.model.named_modules():
            network_keys.append(k)
        for k in except_keys:
            assert k in network_keys, f'Current network has no key called {k}'
        for k, v in self.model.named_modules():
            if k in except_keys:
                assert hasattr(v, 'weight'), f'Except key {k} must have weight value'
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter) and k not in except_keys:
                trainable_params[2]["params"].append(v.bias)  # biases
                trainable_params[2]["weight_decay"] = 0.0
            if isinstance(v, nn.BatchNorm2d):
                trainable_params[0]["params"].append(v.weight)  # no decay
                trainable_params[0]["weight_decay"] = 0.0
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
                trainable_params[1]["params"].append(v.weight)
                if k in except_keys and hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
                    trainable_params[1]["params"].append(v.bias)
                trainable_params[1]["weight_decay"] = weight_decay
        return trainable_params
