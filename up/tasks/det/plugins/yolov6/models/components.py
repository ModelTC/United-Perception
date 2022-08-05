#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# Import from third library
import math
import warnings
import numpy as np

import torch
import torch.nn as nn

from up.utils.model.normalize import build_norm_layer
from up.utils.model.act_fn import build_act_fn


def net_info(depth_multiple=0.33,
             width_multiple=0.25,
             backbone_base_repeats=[1, 6, 12, 18, 6],
             backbone_base_channels=[64, 128, 256, 512, 1024],
             neck_base_repeats=[12, 12, 12, 12],
             neck_base_channels=[256, 128, 128, 256, 256, 512]):
    num_repeats = [
        (max(round(i * depth_multiple), 1) if i > 1 else i) for i in (backbone_base_repeats + neck_base_repeats)
    ]
    channels_list = [
        math.ceil(i * width_multiple / 8) * 8 for i in (backbone_base_channels + neck_base_channels)
    ]
    return channels_list, num_repeats


class Conv(nn.Module):
    '''Normal Conv with SiLU activation'''

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 groups=1,
                 bias=False,
                 normalize={'type': 'solo_bn'},
                 act_fn={'type': 'Silu'}):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        _, self.bn = build_norm_layer(out_channels, normalize)
        _, self.act = build_act_fn(act_fn)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SimConv(nn.Module):
    '''Normal Conv with ReLU activation'''

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 groups=1,
                 bias=False,
                 normalize={'type': 'solo_bn'},
                 act_fn={'type': 'ReLU'}):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        _, self.bn = build_norm_layer(out_channels, normalize)
        _, self.act = build_act_fn(act_fn)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class SimSPPF(nn.Module):
    '''Simplified SPPF with ReLU activation'''

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=5,
                 normalize={'type': 'solo_bn'},
                 act_fn={'type': 'ReLU'}):
        super().__init__()
        c_ = in_channels // 2  # hidden channels
        self.cv1 = SimConv(in_channels, c_, 1, 1, normalize=normalize, act_fn=act_fn)
        self.cv2 = SimConv(c_ * 4, out_channels, 1, 1, normalize=normalize, act_fn=act_fn)
        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


class Transpose(nn.Module):
    '''Normal Transpose, default for upsampling'''

    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super().__init__()
        self.upsample_transpose = torch.nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=True
        )

    def forward(self, x):
        return self.upsample_transpose(x)


class Concat(nn.Module):
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


def conv_bn(in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=1,
            normalize={'type': 'solo_bn'}):
    '''Basic cell for rep-style block, including conv and bn'''
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=padding,
                                        groups=groups,
                                        bias=False))
    name, bn = build_norm_layer(out_channels, normalize)
    result.add_module(name, bn)
    return result


class RepBlock(nn.Module):
    '''
        RepBlock is a stage block with rep-style basic block
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 n=1,
                 normalize={'type': 'solo_bn'},
                 act_fn={'type': 'ReLU'}):
        super().__init__()
        self.conv1 = RepVGGBlock(in_channels, out_channels, normalize=normalize, act_fn=act_fn)
        self.block = nn.Sequential(
            *(RepVGGBlock(out_channels, out_channels, normalize=normalize, act_fn=act_fn)
                for _ in range(n - 1))
        ) if n > 1 else None

    def forward(self, x):
        x = self.conv1(x)
        if self.block is not None:
            x = self.block(x)
        return x


class RepVGGBlock(nn.Module):
    '''RepVGGBlock is a basic rep-style block, including training and deploy status
    This code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 padding_mode='zeros',
                 normalize={'type': 'solo_bn'},
                 act_fn={'type': 'ReLU'}):
        super(RepVGGBlock, self).__init__()
        """ Initialization of the class.
        Args:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int or tuple): Size of the convolving kernel
            stride (int or tuple, optional): Stride of the convolution. Default: 1
            padding (int or tuple, optional): Zero-padding added to both sides of
                the input. Default: 1
            dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
            groups (int, optional): Number of blocked connections from input
                channels to output channels. Default: 1
            padding_mode (string, optional): Default: 'zeros
        """
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2
        _, self.nonlinearity = build_act_fn(act_fn)

        self.se = nn.Identity()

        if out_channels == in_channels and stride == 1:
            _, self.rbr_identity = build_norm_layer(in_channels, normalize, 0)
        else:
            self.rbr_identity = None

        self.rbr_dense = conv_bn(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=kernel_size,
                                 stride=stride,
                                 padding=padding,
                                 groups=groups,
                                 normalize=normalize)
        self.rbr_1x1 = conv_bn(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=1,
                               stride=stride,
                               padding=padding_11,
                               groups=groups,
                               normalize=normalize)

    def forward(self, inputs):
        '''Forward process'''
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))

    def fuse_model(self):
        if hasattr(self, 'rbr_reparam'):
            self.__delattr__('rbr_reparam')
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels,
                                     out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.rbr_dense.conv.kernel_size,
                                     stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding,
                                     dilation=self.rbr_dense.conv.dilation,
                                     groups=self.rbr_dense.conv.groups,
                                     bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias

        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std
