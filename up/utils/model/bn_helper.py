import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules._functions import SyncBatchNorm as sync_batch_norm

from ..env.dist_helper import env
from up.utils.general.log_helper import default_logger as logger
from up.utils.env.dist_helper import simple_group_split
try:
    import spring.linklink as link
except:  # noqa
    from up.utils.general.fake_linklink import link


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
    def __init__(self, num_features, group_size=None, **kwargs):
        super(PyTorchSyncBN, self).__init__(
            num_features,
            process_group=self._get_group(group_size),
            **kwargs
        )

    @staticmethod
    def _get_group(group_size):
        rank = env.rank
        world_size = env.world_size
        if group_size is None or group_size == world_size:
            return None
        if group_size > world_size:
            logger.warning("group_size of '{}' invalid, reset to {}".format(group_size, world_size))
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
                 affine=True,
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


class TaskBatchNorm2d(nn.BatchNorm2d):

    def __init__(self,
                 num_features,
                 eps=1e-5,
                 momentum=0.1,
                 affine=True,
                 track_running_stats=True,
                 task_num=1,
                 task_idx=0):
        super(TaskBatchNorm2d, self).__init__(num_features, eps, momentum, affine, track_running_stats)
        self.bn = nn.ModuleList([
            nn.BatchNorm2d(num_features, affine=False) for _ in range(task_num)])
        self.task_idx = task_idx
        self.task_num = task_num

    def set_task_idx(self, idx):
        if idx == 0:
            self.task_bn = [self]
        else:
            self.task_bn = [self.bn[idx]]
        assert self.task_idx <= self.task_num - 1

    def init_param(self):
        for idx in range(1, self.task_num):
            self.bn[idx].running_mean.copy_(self.running_mean)
            self.bn[idx].running_var.copy_(self.running_var)

    def forward(self, x):
        y = nn.functional.batch_norm(
            x,
            self.bn[0].running_mean,
            self.bn[0].running_var,
            self.weight,
            self.bias,
            self.training,
            self.momentum,
            self.eps)
        return y

    @property
    def module_str(self):
        return 'TaskBN'


class TorchSyncTaskBatchNorm2d(PyTorchSyncBN):

    def __init__(self,
                 num_features,
                 group_size=None,
                 momentum=0.1,
                 task_num=1,
                 task_idx=0,
                 skip_multi_task_init: bool = False):
        super(TorchSyncTaskBatchNorm2d, self).__init__(num_features, group_size)
        self.bn = nn.ModuleList([
            nn.BatchNorm2d(num_features, affine=False) for _ in range(task_num)])
        self.bn[0] = None
        self.momentum = momentum
        self.task_idx = task_idx
        self.task_num = task_num
        self.set_task_idx(task_idx)
        self.skip_multi_task_init = skip_multi_task_init
        self.register_buffer('multi_task_init', torch.torch.ByteTensor([False]))

    def set_task_idx(self, idx):
        if idx == 0:
            self.task_bn = [self]
        else:
            self.task_bn = [self.bn[idx]]
        assert self.task_idx <= self.task_num - 1

    def init_param(self):
        for idx in range(1, self.task_num):
            self.bn[idx].running_mean.copy_(self.running_mean)
            self.bn[idx].running_var.copy_(self.running_var)

    def forward(self, x):
        if not self.multi_task_init.item() and not self.skip_multi_task_init:
            self.init_param()
            self.multi_task_init[0] = True
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum
        if self.training:
            bn_training = True
        else:
            bn_training = (self.task_bn[0].running_mean is None) and (self.task_bn[0].running_var is None)

        # Don't sync batchnorm stats in inference mode (model.eval()).
        need_sync = (bn_training and self.training)
        if need_sync:
            process_group = torch.distributed.group.WORLD
            if self.process_group:
                process_group = self.process_group
            world_size = torch.distributed.get_world_size(process_group)
            need_sync = world_size > 1

        # fallback to framework BN when synchronization is not necessary
        if not need_sync:
            y = F.batch_norm(
                x,
                self.task_bn[0].running_mean,
                self.task_bn[0].running_var,
                self.weight,
                self.bias,
                bn_training,
                exponential_average_factor,
                self.eps,
            )
        else:
            assert bn_training
            y = sync_batch_norm.apply(x,
                                      self.weight,
                                      self.bias,
                                      self.task_bn[0].running_mean,
                                      self.task_bn[0].running_var,
                                      exponential_average_factor,
                                      self.eps,
                                      self.process_group,
                                      env.world_size)
        return y

    @property
    def module_str(self):
        return 'TorchSyncTaskBN'


class SyncTaskBatchNorm2d(GroupSyncBatchNorm):

    def __init__(self,
                 num_features,
                 group_size=None,
                 momentum=0.1,
                 sync_stats=True,
                 var_mode=link.syncbnVarMode_t.L2,
                 task_num=1,
                 task_idx=0,
                 skip_multi_task_init: bool = False):
        super(SyncTaskBatchNorm2d, self).__init__(num_features, group_size, momentum, sync_stats, var_mode)
        self.bn = nn.ModuleList([
            nn.BatchNorm2d(num_features, affine=False) for _ in range(task_num)])
        self.bn[0] = None
        self.task_idx = task_idx
        self.task_num = task_num
        self.set_task_idx(task_idx)
        self.skip_multi_task_init = skip_multi_task_init
        self.register_buffer('multi_task_init', torch.torch.ByteTensor([False]))

    def set_task_idx(self, idx):
        if idx == 0:
            self.task_bn = [self]
        else:
            self.task_bn = [self.bn[idx]]
        assert self.task_idx <= self.task_num - 1

    def init_param(self):
        for idx in range(1, self.task_num):
            self.bn[idx].running_mean.copy_(self.running_mean)
            self.bn[idx].running_var.copy_(self.running_var)

    def forward(self, x):
        if not self.multi_task_init.item() and not self.skip_multi_task_init:
            self.init_param()
            self.multi_task_init[0] = True
        y = link.nn.SyncBNFunc.apply(x,
                                     self.weight,
                                     self.bias,
                                     self.task_bn[0].running_mean,
                                     self.task_bn[0].running_var,
                                     self.eps,
                                     self.momentum,
                                     self.group,
                                     self.sync_stats,
                                     self.var_mode,
                                     self.training,
                                     self.async_ar_stats)
        return y

    @property
    def module_str(self):
        return 'SyncTaskBN'
