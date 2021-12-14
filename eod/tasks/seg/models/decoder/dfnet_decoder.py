# Author: Xiangtai Li
# Email: lixiangtai@sensetime.com

import torch
import torch.nn as nn
from torch.nn import functional as F
from eod.utils.model.normalize import build_norm_layer
from eod.utils.general.registry_factory import MODULE_ZOO_REGISTRY
from eod.models.losses import build_loss

__all__ = ['DFSegDecoder']


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


class Aux_Module(nn.Module):
    def __init__(self, in_planes, num_classes=19, normalize={'type': 'solo_bn'}):
        super(Aux_Module, self).__init__()

        self.aux = nn.Sequential(
            nn.Conv2d(in_planes, 256, kernel_size=3, stride=1, padding=1),
            build_norm_layer(256, normalize)[1],
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True))

    def forward(self, x):
        res = self.aux(x)
        return res


class FusionNode(nn.Module):
    def __init__(self, inplane):
        super(FusionNode, self).__init__()
        self.fusion = conv3x3(inplane * 2, inplane)

    def forward(self, x):
        x_h, x_l = x
        size = x_l.size()[2:]
        x_h = F.upsample(x_h, size, mode="bilinear", align_corners=True)
        res = self.fusion(torch.cat([x_h, x_l], dim=1))
        return res


class PSPModule(nn.Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """

    def __init__(self, inplanes, out_planes=512, sizes=(1, 2, 3, 6), normalize={'type': 'solo_bn'}):
        super(PSPModule, self).__init__()
        self.stages = []
        self._normalize = normalize
        self.out_planes = out_planes
        self.stages = nn.ModuleList([self._make_stage(inplanes, out_planes, size, normalize) for size in sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(inplanes + len(sizes) * out_planes, out_planes, kernel_size=1, padding=1, dilation=1, bias=False),
            build_norm_layer(out_planes, normalize)[1],
            nn.ReLU(),
            nn.Dropout2d(0.1)
        )

    def _make_stage(self, inplanes, out_planes, size, normalize={'type': 'solo_bn'}):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(inplanes, out_planes, kernel_size=1, bias=False)
        bn = build_norm_layer(out_planes, normalize)[1]
        return nn.Sequential(prior, conv, bn)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in
                  self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle

    def get_outplanes(self):
        return self.out_planes


@MODULE_ZOO_REGISTRY.register('dfnet_decoder')
class DFSegDecoder(nn.Module):
    def __init__(self, inplanes=512, num_classes=19, inner_planes=128, normalize={'type': 'solo_bn'}, loss=None):
        super(DFSegDecoder, self).__init__()
        self.prefix = self.__class__.__name__
        self.cc5 = nn.Conv2d(128, inner_planes, 1)
        self.cc4 = nn.Conv2d(256, inner_planes, 1)
        self.cc3 = nn.Conv2d(128, inner_planes, 1)

        self.ppm = PSPModule(inplanes, inner_planes, normalize=normalize)

        self.fn4 = FusionNode(inner_planes)
        self.fn3 = FusionNode(inner_planes)

        self.fc = Aux_Module(128, num_classes, normalize=normalize)

        self.loss = build_loss(loss)

    def forward(self, x):
        x3, x4, x5 = x['features']
        size = x['size']
        gt_seg = x['gt_seg']
        x5 = self.ppm(x5)
        x5 = self.cc5(x5)
        x4 = self.cc4(x4)
        f4 = self.fn4([x5, x4])
        x3 = self.cc3(x3)
        out = self.fn3([f4, x3])
        pred = self.fc(out)

        pred = F.upsample(pred, size=size, mode='bilinear', align_corners=True)
        if self.training:
            loss = self.loss(pred, gt_seg)
            return {f"{self.prefix}.loss": loss, "blob_pred": pred}
        else:
            return {"blob_pred": pred}
