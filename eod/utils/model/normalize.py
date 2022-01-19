# Import from third library
import torch
import torch.nn as nn

from eod.utils.model.bn_helper import (
    CaffeFrozenBatchNorm2d,
    FrozenBatchNorm2d,
    GroupNorm,
    PyTorchSyncBN,
)

_norm_cfg = {
    'solo_bn': ('bn', torch.nn.BatchNorm2d),
    'freeze_bn': ('bn', FrozenBatchNorm2d),
    'caffe_freeze_bn': ('bn', CaffeFrozenBatchNorm2d),
    'gn': ('gn', GroupNorm),
    'pt_sync_bn': ('bn', PyTorchSyncBN),
}

try:
    from mqbench.nn.modules import FrozenBatchNorm2d as MqbenchFrozenBatchNorm2d
    _norm_cfg.update({'mqbench_freeze_bn': ('bn', MqbenchFrozenBatchNorm2d)})
except Exception as err:
    print(err)
    print("If you need Mqbench to quantize model, you should add Mqbench to this project. Or just ignore this error.")


def build_norm_layer(num_features, cfg, postfix=''):
    assert isinstance(cfg, dict) and 'type' in cfg
    cfg = cfg.copy()
    layer_type = cfg.pop('type')
    kwargs = cfg.get('kwargs', {})

    if layer_type not in _norm_cfg:
        raise KeyError('Unrecognized norm type {}'.format(layer_type))
    else:
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
