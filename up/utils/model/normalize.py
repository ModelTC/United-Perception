# Import from third library
import torch
import torch.nn as nn
import torch.nn.functional as F
from up.utils.general.global_flag import DIST_BACKEND
from up.utils.general.log_helper import default_logger as logger

from up.utils.model.bn_helper import (
    CaffeFrozenBatchNorm2d,
    FrozenBatchNorm2d,
    GroupNorm,
    PyTorchSyncBN,
    TaskBatchNorm2d,
    GroupSyncBatchNorm,
    SyncTaskBatchNorm2d,
    TorchSyncTaskBatchNorm2d,
)

_norm_cfg = {
    'solo_bn': ('bn', torch.nn.BatchNorm2d),
    'freeze_bn': ('bn', FrozenBatchNorm2d),
    'caffe_freeze_bn': ('bn', CaffeFrozenBatchNorm2d),
    'gn': ('gn', GroupNorm),
    'pt_sync_bn': ('bn', PyTorchSyncBN),
    'task_solo_bn': ('bn', TaskBatchNorm2d),
    'link_sync_bn': ('bn', GroupSyncBatchNorm),
    'sync_bn': ('bn', GroupSyncBatchNorm),
    'task_sync_bn': ('bn', SyncTaskBatchNorm2d),
    'torch_task_sync_bn': ('bn', TorchSyncTaskBatchNorm2d),
    'ln': ('ln', torch.nn.LayerNorm)
}

try:
    from mqbench.nn.modules import FrozenBatchNorm2d as MqbenchFrozenBatchNorm2d
    _norm_cfg.update({'mqbench_freeze_bn': ('bn', MqbenchFrozenBatchNorm2d)})
except Exception as err:
    logger.warning(f"import error {err}; If you need Mqbench to quantize model, \
     you should add Mqbench to this project. Or just ignore this error.")

try:
    from msbench.nn.modules import FrozenBatchNorm2d as MsbenchFrozenBatchNorm2d
    _norm_cfg.update({'msbench_freeze_bn': ('bn', MsbenchFrozenBatchNorm2d)})
except Exception as err:
    logger.warning(f"import error {err}; If you need Msbench to prune model, \
    you should add Msbench to this project. Or just ignore this error.")


def is_bn(m):
    if isinstance(m, torch.nn.BatchNorm2d):
        return True
    if isinstance(m, FrozenBatchNorm2d):
        return True
    if isinstance(m, CaffeFrozenBatchNorm2d):
        return True
    if isinstance(m, PyTorchSyncBN):
        return True
    if isinstance(m, GroupSyncBatchNorm):
        return True
    if isinstance(m, TaskBatchNorm2d):
        return True
    if isinstance(m, SyncTaskBatchNorm2d):
        return True
    if isinstance(m, torch.nn.LayerNorm):
        return True
    return False


def build_norm_layer(num_features, cfg, postfix=''):
    assert isinstance(cfg, dict) and 'type' in cfg
    cfg = cfg.copy()
    layer_type = cfg.pop('type')
    kwargs = cfg.get('kwargs', {})

    if layer_type not in _norm_cfg:
        raise KeyError('Unrecognized norm type {}'.format(layer_type))
    else:
        if 'task_sync_bn' in layer_type:
            if DIST_BACKEND.backend == 'linklink':
                layer_type = 'task_sync_bn'
            elif DIST_BACKEND.backend == 'dist':
                layer_type = 'torch_task_sync_bn'
            else:
                raise NotImplementedError
        elif 'sync_bn' in layer_type:
            if DIST_BACKEND.backend == 'dist':
                layer_type = 'pt_sync_bn'
            elif DIST_BACKEND.backend == 'linklink':
                layer_type = 'link_sync_bn'
            else:
                raise NotImplementedError
        abbr, norm_layer = _norm_cfg[layer_type]
        if norm_layer is None:
            raise NotImplementedError

    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    layer = norm_layer(num_features, **kwargs)
    return name, layer


def build_conv_norm(in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    padding=0,
                    dilation=1,
                    groups=1,
                    normalize=None,
                    activation=False,
                    relu_first=False):

    conv = nn.Conv2d(
        in_channels, out_channels, kernel_size=kernel_size,
        stride=stride, padding=padding, dilation=dilation,
        groups=groups, bias=(normalize is None))

    # for compability
    if (normalize is None) and (not activation):
        return conv

    seq = nn.Sequential()
    if relu_first and activation:
        seq.add_module('relu', nn.ReLU(inplace=True))
    seq.add_module('conv', conv)
    if normalize is not None:
        norm_name, norm = build_norm_layer(out_channels, normalize)
        seq.add_module(norm_name, norm)
    if activation:
        if not relu_first:
            seq.add_module('relu', nn.ReLU(inplace=True))
    return seq


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
