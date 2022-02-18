import torch.nn as nn
from eod.utils.model.optimizer_helper import BaseOptimizer
from eod.utils.general.registry_factory import OPTIMIZER_REGISTRY


__all__ = ['QatWeightsOptimizer']


@OPTIMIZER_REGISTRY.register('qat_weights')
class QatWeightsOptimizer(BaseOptimizer):
    def get_trainable_params(self, cfg_optim):
        weight_decay = cfg_optim['kwargs']['weight_decay']
        scale_lr = cfg_optim['kwargs']['lr'] * 10
        trainable_params = [{"params": []}, {"params": []}, {"params": []}, {"params": []}]
        for k, v in self.model.named_modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
                trainable_params[2]["params"].append(v.bias)  # biases
                trainable_params[2]["weight_decay"] = 0.0
            if isinstance(v, nn.BatchNorm2d):
                trainable_params[0]["params"].append(v.weight)  # no decay
                trainable_params[0]["weight_decay"] = 0.0
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
                trainable_params[1]["params"].append(v.weight)
                trainable_params[1]["weight_decay"] = weight_decay
        return trainable_params

