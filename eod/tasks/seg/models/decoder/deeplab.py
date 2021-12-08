# !/usr/bin/env python
# -*- coding:utf-8 -*-
# cp from light seg
# Author: Peiwen Lin, Xiangtai Li
# Email: linpeiwen@sensetime.com, lixiangtai@sensetime.com

import torch
import torch.nn as nn
from torch.nn import functional as F
from eod.utils.model.normalize import build_norm_layer
from eod.utils.general.registry_factory import MODULE_ZOO_REGISTRY


__all__ = ['dec_deeplabv3', 'dec_deeplabv3p']


class ASPP(nn.Module):
    """
    Reference:
        Chen, Liang-Chieh, et al. *"Rethinking Atrous Convolution for Semantic Image Segmentation."*
    """
    def __init__(self, in_planes, inner_planes=256, normalize={'type': 'solo_bn'}, dilations=(12, 24, 36)):
        super(ASPP, self).__init__()
        self.conv1 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                   nn.Conv2d(in_planes, inner_planes, kernel_size=1, padding=0, dilation=1, bias=False),
                                   build_norm_layer(inner_planes, normalize)[1],
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_planes, inner_planes, kernel_size=1, padding=0, dilation=1, bias=False),
                                   build_norm_layer(inner_planes, normalize)[1],
                                   nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(in_planes, inner_planes, kernel_size=3,
                                   padding=dilations[0], dilation=dilations[0], bias=False),
                                   build_norm_layer(inner_planes, normalize)[1],
                                   nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(in_planes, inner_planes, kernel_size=3,
                                   padding=dilations[1], dilation=dilations[1], bias=False),
                                   build_norm_layer(inner_planes, normalize)[1],
                                   nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(nn.Conv2d(in_planes, inner_planes, kernel_size=3,
                                   padding=dilations[2], dilation=dilations[2], bias=False),
                                   build_norm_layer(inner_planes, normalize)[1],
                                   nn.ReLU(inplace=True))

        self.out_planes = (len(dilations) + 2) * inner_planes

    def get_outplanes(self):
        return self.out_planes

    def forward(self, x):
        _, _, h, w = x.size()
        feat1 = F.upsample(self.conv1(x), size=(h, w), mode='bilinear', align_corners=True)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)
        aspp_out = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)
        return aspp_out


class Aux_Module(nn.Module):
    def __init__(self, in_planes, num_classes=19, normalize={'type': 'solo_bn'}):
        super(Aux_Module, self).__init__()

        self.aux = nn.Sequential(nn.Conv2d(in_planes, 256, kernel_size=3, stride=1, padding=1),
                                 build_norm_layer(256, normalize)[1],
                                 nn.ReLU(inplace=True),
                                 nn.Dropout2d(0.1),
                                 nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True))

    def forward(self, x):
        res = self.aux(x)
        return res


@MODULE_ZOO_REGISTRY.register('deeplabv3')
class dec_deeplabv3(nn.Module):
    """
    Reference:
        Chen, Liang-Chieh, et al. *"Rethinking Atrous Convolution for Semantic Image Segmentation."*
    """
    def __init__(self,
                 in_planes,
                 num_classes=19,
                 inner_planes=256,
                 dilations=(12, 24, 36),
                 with_aux=True,
                 criterion=None,
                 normalize={'type': 'solo_bn'}):
        super(dec_deeplabv3, self).__init__()

        self.with_aux = with_aux
        self.aspp = ASPP(in_planes, inner_planes=inner_planes, normalize=normalize, dilations=dilations)
        self.head = nn.Sequential(
            nn.Conv2d(self.aspp.get_outplanes(), 256, kernel_size=3, padding=1, dilation=1, bias=False),
            build_norm_layer(256, normalize)[1],
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True))
        if self.with_aux:
            self.aux_layer = Aux_Module(in_planes // 2, num_classes, normalize=normalize)
        self.criterion = criterion

    def forward(self, x):
        x1, x2, x3, x4 = x['features']
        size = x['size']
        aspp_out = self.aspp(x4)
        pred = self.head(aspp_out)

        pred = F.upsample(pred, size=size, mode='bilinear', align_corners=True)
        if self.training and self.with_aux:
            gt_seg = x['gt_seg']
            aux_pred = self.aux_layer(x3)
            aux_pred = F.upsample(aux_pred, size=size, mode='bilinear', align_corners=True)
            pred = pred, aux_pred
            return {"loss": self.criterion(pred, gt_seg), "blob_pred": pred[0]}
        else:
            return {"blob_pred": pred}


@MODULE_ZOO_REGISTRY.register('deeplabv3+')
class dec_deeplabv3p(nn.Module):
    """
    Reference:
        Chen, Liang-Chieh, et al.. Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation"*
    """
    def __init__(self,
                 in_planes,
                 num_classes=19,
                 inner_planes=256,
                 normalize={'type': 'solo_bn'},
                 dilations=(12, 24, 36),
                 criterion=None):
        super(dec_deeplabv3p, self).__init__()
        self.aspp = ASPP(in_planes, inner_planes=inner_planes, normalize=normalize, dilations=dilations)

        self.head = nn.Sequential(
            nn.Conv2d(self.aspp.get_outplanes() + 48, 256, kernel_size=3, padding=1, dilation=1, bias=False),
            build_norm_layer(256, normalize)[1],
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True))

        self.fine = nn.Conv2d(256, 48, 1)

        if self.with_aux:
            self.aux_layer = Aux_Module(in_planes // 2, num_classes, normalize=normalize)
        self.criterion = criterion

    def forward(self, x):
        x1, x2, x3, x4 = x['features']
        size = x['size']
        aspp_out = self.aspp(x4)
        bot = self.fine(x1)
        out = torch.cat([aspp_out, bot], dim=1)
        pred = self.head(out)
        pred = F.upsample(pred, size=size, mode='bilinear', align_corners=True)
        if self.training and self.with_aux:
            gt_seg = x['gt_seg']
            aux_pred = self.aux_layer(x3)
            aux_pred = F.upsample(aux_pred, size=size, mode='bilinear', align_corners=True)
            pred = pred, aux_pred
            return {"loss": self.criterion(pred, gt_seg), "blob_pred": pred[0]}
        else:
            return {"blob_pred": pred}
