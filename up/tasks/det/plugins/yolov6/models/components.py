#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# Import from third library
import warnings

import torch
import torch.nn as nn

from up.utils.model.normalize import build_norm_layer
from up.utils.model.act_fn import build_act_fn


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
        _, self.bn = build_norm_layer(out_channels, normalize, 0)
        _, self.act = build_act_fn(act_fn)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


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
        _, self.bn = build_norm_layer(out_channels, normalize, 0)
        _, self.act = build_act_fn(act_fn)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class SimSPPF(nn.Module):
    '''Simplified SPPF with ReLU activation'''

    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        c_ = in_channels // 2  # hidden channels
        self.cv1 = SimConv(in_channels, c_, 1, 1)
        self.cv2 = SimConv(c_ * 4, out_channels, 1, 1)
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
    if normalize['type'] == 'solo_bn':
        result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    else:
        name, bn = build_norm_layer(out_channels, normalize, 0)
        result.add_module(name, bn)
    return result


class RepBlock(nn.Module):
    '''
        RepBlock is a stage block with rep-style basic block
    '''

    def __init__(self, in_channels, out_channels, n=1, normalize={'type': 'solo_bn'}, act_fn={'type': 'ReLU'}):
        super().__init__()
        self.conv1 = RepVGGBlock(in_channels, out_channels, normalize=normalize, act_fn=act_fn)
        self.block = nn.Sequential(
            *(RepVGGBlock(out_channels, out_channels, normalize=normalize, act_fn=act_fn) for _ in range(n - 1))
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
                 deploy=False,
                 use_se=False,
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
            padding_mode (string, optional): Default: 'zeros'
            deploy: Whether to be deploy status or training status. Default: False
            use_se: Whether to use se. Default: False
        """
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2
        _, self.nonlinearity = build_act_fn(act_fn)

        if use_se:
            raise NotImplementedError("se block not supported yet")
        else:
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
        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))
