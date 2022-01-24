# !/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Peiwen Lin, Xiangtai Li
# Email: linpeiwen@sensetime.com, lixiangtai@sensetime.com

import torch.nn as nn
from torch.nn import functional as F

from up.utils.model.normalize import build_norm_layer
from up.utils.general.registry_factory import MODULE_ZOO_REGISTRY
from up.models.losses import build_loss
from ..components import PSPModule, Aux_Module

__all__ = ['dec_pspnet']


@MODULE_ZOO_REGISTRY.register('pspnet')
class dec_pspnet(nn.Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """

    def __init__(self, inplanes, with_aux=True, num_classes=19, inner_planes=512,
                 normalize={'type': 'solo_bn'}, sizes=(1, 2, 3, 6), loss=None):
        super(dec_pspnet, self).__init__()
        self.prefix = self.__class__.__name__
        self.ppm = PSPModule(inplanes, out_planes=inner_planes, normalize=normalize, sizes=sizes)
        self.head = nn.Sequential(
            nn.Conv2d(self.ppm.get_outplanes(), 512, kernel_size=3, padding=1, dilation=1, bias=False),
            build_norm_layer(512, normalize)[1],
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True))
        self.with_aux = with_aux
        self.loss = build_loss(loss)
        if self.with_aux:
            self.aux_layer = Aux_Module(inplanes // 2, num_classes, normalize)

    def forward(self, x):
        x1, x2, x3, x4 = x['features']
        size = x['size']
        ppm_out = self.ppm(x4)
        pred = self.head(ppm_out)
        pred = F.upsample(pred, size=size, mode='bilinear', align_corners=True)
        if self.training and self.with_aux:
            gt_seg = x['gt_seg']
            aux_pred = self.aux_layer(x3)
            aux_pred = F.upsample(aux_pred, size=size, mode='bilinear', align_corners=True)
            pred = pred, aux_pred
            loss = self.loss(pred, gt_seg)
            return {f"{self.prefix}.loss": loss, "blob_pred": pred[0]}
        else:
            return {"blob_pred": pred}
