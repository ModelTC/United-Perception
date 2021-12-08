# !/usr/bin/env python
# -*- coding:utf-8 -*-
# cp from light seg
# Author: Peiwen Lin, Xiangtai Li
# Email: linpeiwen@sensetime.com, lixiangtai@sensetime.com
import torch
import torch.nn as nn
from eod.utils.model.normalize import build_norm_layer
from eod.utils.model.bn_helper import PyTorchSyncBN
from eod.utils.general.registry_factory import MODULE_ZOO_REGISTRY


__all__ = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': '/mnt/lustre/lixiangtai/pretrained/resnet18-deep.pth',
    'resnet34': '/mnt/lustre/share/HiLight/model_zoo/resnet34-333f7ec4.pth',
    'resnet50': '/mnt/lustre/share/HiLight/model_zoo/resnet50-ebb6acbb.pth',
    'resnet101': '/mnt/lustre/share/HiLight/model_zoo/resnet101-2a57e44d.pth',
    'resnet152': '/mnt/lustre/share/HiLight/model_zoo/resnet152-0d43d698.pth',
}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, normalize={'type': 'solo_bn'}):
        super(BasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = build_norm_layer(planes, normalize)[1]
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = build_norm_layer(planes, normalize)[1]
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, normalize={'type': 'solo_bn'}):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = build_norm_layer(width, normalize)[1]
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = build_norm_layer(width, normalize)[1]
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = build_norm_layer(planes * self.expansion, normalize)[1]
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 zero_init_residual=False,
                 groups=1,
                 width_per_group=64,
                 replace_stride_with_dilation=[False, False, False],
                 normalize={'type': 'solo_bn'},
                 multi_grid=False,
                 out_strides=[],
                 out_layers=[]):
        super(ResNet, self).__init__()
        self.out_strides = out_strides
        self.inplanes = 128
        self.dilation = 1
        self._normalize = normalize
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]

        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Sequential(
            conv3x3(3, 64, stride=2),
            build_norm_layer(64, normalize)[1],
            nn.ReLU(inplace=True),
            conv3x3(64, 64),
            build_norm_layer(64, normalize)[1],
            nn.ReLU(inplace=True),
            conv3x3(64, self.inplanes))
        self.bn1 = build_norm_layer(self.inplanes, normalize)[1]
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)  # change

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2],
                                       multi_grid=multi_grid)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, PyTorchSyncBN)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def get_outplanes(self):
        return self.inplanes

    def get_auxplanes(self):
        return self.inplanes // 2

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, multi_grid=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                build_norm_layer(planes * block.expansion, self._normalize)[1],
            )

        grids = [1] * blocks
        if multi_grid:
            grids = [2, 2, 4]

        layers = []
        layers.append(block(self.inplanes,
                            planes,
                            stride,
                            downsample,
                            self.groups,
                            self.base_width,
                            previous_dilation * grids[0],
                            self._normalize))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes,
                                planes,
                                groups=self.groups,
                                base_width=self.base_width,
                                dilation=self.dilation * grids[i],
                                normalize=self._normalize))

        return nn.Sequential(*layers)

    def forward(self, input):
        img = input["image"]
        size = img.size()
        x = self.relu(self.bn1(self.conv1(img)))
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        res = {}
        res["features"] = [x1, x2, x3, x4]
        res['size'] = size
        return res

    def get_outstrides(self):
        """
        Returns:

            - out (:obj:`int`): number of channels of output
        """

        return torch.tensor(self.out_strides, dtype=torch.int)


@MODULE_ZOO_REGISTRY.register('seg_resnet18')
def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


@MODULE_ZOO_REGISTRY.register('seg_resnet34')
def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


@MODULE_ZOO_REGISTRY.register('seg_resnet50')
def resnet50(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


@MODULE_ZOO_REGISTRY.register('seg_resnet50c')
def resnet50c(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs, deep_stem=True)
    return model


@MODULE_ZOO_REGISTRY.register('seg_resnet50d')
def resnet50d(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs, deep_stem=True, avg_down=True)
    return model


@MODULE_ZOO_REGISTRY.register('seg_resnet101')
def resnet101(pretrained=True, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


@MODULE_ZOO_REGISTRY.register('seg_resnet152')
def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model
