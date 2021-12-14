# Author: Xiangtai Li
# Email: lixiangtai@sensetime.com
# Partial Order Pruning: for Best Speed/Accuracy Trade-off in Neural Architecture Search (CVPR2019)
from __future__ import print_function, division, absolute_import

import torch
import torch.nn as nn
from eod.utils.model.normalize import build_norm_layer
from eod.utils.general.registry_factory import MODULE_ZOO_REGISTRY

__all__ = ['dfnetv1', 'dfnetv2']

model_urls = {
    'dfv1': '/mnt/lustre/share/HiLight/model_zoo/df1_imagenet.pth',
    'dfv2': '/mnt/lustre/share/HiLight/model_zoo/df2_imagenet.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, normalize={'type': 'solo_bn'}):
        super(BasicBlock, self).__init__()
        self._normalize = normalize
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = build_norm_layer(planes, normalize)[1]
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = build_norm_layer(planes, normalize)[1]
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


@MODULE_ZOO_REGISTRY.register('dfnetv1')
class dfnetv1(nn.Module):
    def __init__(self, num_classes=1000, fpn=True, normalize={'type': 'solo_bn'}):
        super(dfnetv1, self).__init__()
        self.inplanes = 64
        self.fpn = fpn
        self._normalize = normalize

        self.out_planes = 512
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=2, bias=False),
            build_norm_layer(32, normalize)[1],
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2, bias=False),
            build_norm_layer(64, normalize)[1],
            nn.ReLU(inplace=True)
        )

        self.stage2 = self._make_layer(64, 3, stride=2)
        self.stage3 = self._make_layer(128, 3, stride=2)
        self.stage4 = self._make_layer(256, 3, stride=2)
        self.stage5 = self._make_layer(512, 1, stride=1)
        self.avgpool = nn.AvgPool2d(7, stride=1)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * BasicBlock.expansion,
                          kernel_size=1, stride=stride, bias=False),
                build_norm_layer(planes * BasicBlock.expansion, self._normalize)[1],
            )

        layers = []
        layers.append(BasicBlock(self.inplanes, planes, stride, downsample, normalize=self._normalize))
        self.inplanes = planes * BasicBlock.expansion
        for i in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes, normalize=self._normalize))

        return nn.Sequential(*layers)

    def forward(self, input):
        img = input['image']
        size = img.size()[2:]
        x = self.stage1(img)  # 4x32
        x = self.stage2(x)  # 8x64
        x3 = self.stage3(x)  # 16x128
        x4 = self.stage4(x3)  # 32x256
        x5 = self.stage5(x4)  # 32x512

        res = {}
        res['features'] = [x3, x4, x5]
        res['size'] = size
        res['gt_seg'] = input['gt_seg']
        return res

    def get_outplanes(self):
        return self.out_planes


@MODULE_ZOO_REGISTRY.register('dfnetv2')
class dfnetv2(nn.Module):
    def __init__(self, num_classes=1000, fpn=True, normalize={'type': 'solo_bn'}):
        super(dfnetv2, self).__init__()
        self.inplanes = 64
        self.fpn = fpn
        self._normalize = normalize
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=2, bias=False),
            build_norm_layer(32, normalize)[1],
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2, bias=False),
            build_norm_layer(64, normalize)[1],
            nn.ReLU(inplace=True)
        )
        self.out_planes = 512
        self.stage2_1 = self._make_layer(64, 2, stride=2)
        self.stage2_2 = self._make_layer(128, 1, stride=1)
        self.stage3_1 = self._make_layer(128, 10, stride=2)
        self.stage3_2 = self._make_layer(256, 1, stride=1)
        self.stage4_1 = self._make_layer(256, 4, stride=2)
        self.stage4_2 = self._make_layer(512, 2, stride=1)
        self.avgpool = nn.AvgPool2d(7, stride=1)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * BasicBlock.expansion,
                          kernel_size=1, stride=stride, bias=False),
                build_norm_layer(planes * BasicBlock.expansion, self._normalize)[1],
            )

        layers = []
        layers.append(BasicBlock(self.inplanes, planes, stride, downsample, normalize=self._normalize))
        self.inplanes = planes * BasicBlock.expansion
        for i in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes, normalize=self._normalize))

        return nn.Sequential(*layers)

    def forward(self, input):
        img = input['image']
        size = img.size()[2:]
        x = self.stage1(img)  # 4x32
        x = self.stage2_1(x)  # 8x64
        x3 = self.stage2_2(x)  # 8x64
        x4 = self.stage3_1(x3)  # 16x128
        x4 = self.stage3_2(x4)  # 16x128
        x5 = self.stage4_1(x4)  # 32x256
        x5 = self.stage4_2(x5)  # 32x256

        res = {}
        res['features'] = [x3, x4, x5]
        res['size'] = size
        res['gt_seg'] = input['gt_seg']
        return res

    def get_outplanes(self):
        return self.out_planes


def DFNetv1(pretrained=True, **kwargs):
    model = dfnetv1(**kwargs)
    if pretrained:
        model.load_state_dict(torch.load(model_urls['dfv1'], map_location="cpu"), strict=False)
    return model


def DFNetv2(pretrained=True, **kwargs):
    model = dfnetv2(**kwargs)
    if pretrained:
        model.load_state_dict(torch.load(model_urls['dfv2'], map_location="cpu"), strict=False)
    return model


if __name__ == '__main__':
    i = torch.Tensor(1, 3, 512, 512).cuda()
    m = dfnetv2().cuda()
    m(i)
