import numpy as np
import torch

from ..env.dist_helper import env
from eod.utils.general.log_helper import default_logger as logger


class FrozenBatchNorm2d(torch.nn.BatchNorm2d):
    def __init__(self, *args, **kwargs):
        super(FrozenBatchNorm2d, self).__init__(*args, **kwargs)
        self.training = False

    def train(self, mode=False):
        self.training = False
        for module in self.children():
            module.train(False)
        return self


class CaffeFrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters
    are fixed.
        - 1. weight & bias are also frozen in this version
        - 2. there is no eps when computing running_var.rsqrt()
    """

    def __init__(self, n):
        super(CaffeFrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def forward(self, x):
        scale = self.weight * self.running_var.rsqrt()
        bias = self.bias - self.running_mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return x * scale + bias


class GroupNorm(torch.nn.GroupNorm):
    def __init__(self, num_channels, **kwargs):
        kwargs['num_channels'] = num_channels
        super(GroupNorm, self).__init__(**kwargs)


class PyTorchSyncBN(torch.nn.SyncBatchNorm):
    def __init__(self, num_features, group_size, **kwargs):
        super(PyTorchSyncBN, self).__init__(
            num_features,
            process_group=self._get_group(group_size),
            **kwargs
        )

    @staticmethod
    def _get_group(group_size):
        rank = env.rank
        world_size = env.world_size
        if group_size is None or group_size > world_size:
            logger.warning("roup_size of '{}' invalid, reset to {}".format(group_size, world_size))
            group_size = world_size
        num_groups = world_size // group_size
        groups = []
        rank_list = np.split(np.arange(world_size), num_groups)
        rank_list = [list(map(int, x)) for x in rank_list]
        for i in range(num_groups):
            groups.append(torch.distributed.new_group(rank_list[i]))
        group_size = world_size // num_groups
        return groups[rank // group_size]
