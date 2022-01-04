# Import from third library
import torch
import torch.nn as nn
from eod.utils.general.global_flag import DIST_BACKEND

from eod.utils.model.bn_helper import (
    CaffeFrozenBatchNorm2d,
    FrozenBatchNorm2d,
    GroupNorm,
    PyTorchSyncBN,
    GroupSyncBatchNorm
)

_norm_cfg = {
    'solo_bn': ('bn', torch.nn.BatchNorm2d),
    'freeze_bn': ('bn', FrozenBatchNorm2d),
    'caffe_freeze_bn': ('bn', CaffeFrozenBatchNorm2d),
    'gn': ('gn', GroupNorm),
    'pt_sync_bn': ('bn', PyTorchSyncBN),
    'link_sync_bn': ('bn', GroupSyncBatchNorm),
    'sync_bn': ('bn', GroupSyncBatchNorm)
}


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
    return False


def build_norm_layer(num_features, cfg, postfix=''):
    assert isinstance(cfg, dict) and 'type' in cfg
    cfg = cfg.copy()
    layer_type = cfg.pop('type')
    kwargs = cfg.get('kwargs', {})

    if layer_type not in _norm_cfg:
        raise KeyError('Unrecognized norm type {}'.format(layer_type))
    else:
        if 'sync_bn' in layer_type:
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
