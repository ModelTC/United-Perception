import torch.nn as nn
from up.utils.env.dist_helper import env
from up.utils.model.optimizer_helper import BaseOptimizer
from up.utils.general.registry_factory import OPTIMIZER_REGISTRY

__all__ = ['Yolov6Optimizer']


@OPTIMIZER_REGISTRY.register('yolov6')
class Yolov6Optimizer(BaseOptimizer):
    def __init__(self, cfg_optim, model):
        super(Yolov6Optimizer, self).__init__(cfg_optim, model)
        self.cfg_optim = cfg_optim
        self.model = model
        self.batch_size = cfg_optim["kwargs"].pop("batch_size")

    def get_trainable_params(self, cfg_optim):
        weight_decay = cfg_optim['kwargs']['weight_decay']
        batch_size_all = self.batch_size * env.world_size
        accumulate = max(1, round(64 / batch_size_all))
        weight_decay *= batch_size_all * accumulate / 64
        trainable_params = [{"params": []}, {"params": []}, {"params": []}]

        trainable_params_list = []
        for k, v in self.model.named_modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
                trainable_params[2]["params"].append(v.bias)  # biases
                trainable_params_list.append(k + ".bias")
                trainable_params[2]["weight_decay"] = 0.0
            if isinstance(v, nn.BatchNorm2d):
                trainable_params[0]["params"].append(v.weight)  # no decay
                trainable_params_list.append(k + ".weight")
                trainable_params[0]["weight_decay"] = 0.0
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
                trainable_params[1]["params"].append(v.weight)
                trainable_params_list.append(k + ".weight")
                trainable_params[1]["weight_decay"] = weight_decay

        return trainable_params
