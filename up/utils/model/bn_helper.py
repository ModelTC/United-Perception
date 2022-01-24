import numpy as np
import torch

from ..env.dist_helper import env
from up.utils.general.log_helper import default_logger as logger

try:
    import spring.linklink as link
except: # noqa
    link = None


def simple_group_split(world_size, rank, num_groups):
    groups = []
    rank_list = np.split(np.arange(world_size), num_groups)
    rank_list = [list(map(int, x)) for x in rank_list]
    for i in range(num_groups):
        groups.append(link.new_group(ranks=rank_list[i]))
    group_size = world_size // num_groups
    return groups[rank // group_size]


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


class GroupSyncBatchNorm(link.nn.SyncBatchNorm2d):
    group_by_size = {}

    def __init__(self,
                 num_features,
                 group_size=None,
                 momentum=0.1,
                 sync_stats=True,
                 var_mode=link.syncbnVarMode_t.L2):

        self.group_size = group_size

        super(GroupSyncBatchNorm, self).__init__(
            num_features,
            momentum=momentum,
            group=self._get_group(group_size),
            sync_stats=sync_stats,
            var_mode=var_mode
        )

    @staticmethod
    def _get_group(group_size):

        rank = env.rank
        world_size = env.world_size
        if group_size is None or group_size > world_size:
            logger.warning("group_size of '{}' invalid, reset to {}".format(group_size, world_size))
            group_size = world_size

        if group_size in GroupSyncBatchNorm.group_by_size:
            return GroupSyncBatchNorm.group_by_size[group_size]

        assert world_size % group_size == 0
        bn_group_comm = simple_group_split(world_size, rank, world_size // group_size)
        GroupSyncBatchNorm.group_by_size[group_size] = bn_group_comm
        return bn_group_comm

    def __repr__(self):
        return ('{name}({num_features},'
                ' eps={eps},'
                ' momentum={momentum},'
                ' affine={affine},'
                ' group={group},'
                ' group_size={group_size},'
                ' sync_stats={sync_stats},'
                ' var_mode={var_mode})'.format(
                    name=self.__class__.__name__, **self.__dict__))
