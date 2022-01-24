# Author: Xiangtai Li
# Email: lixiangtai@sensetime.com

import torch.nn as nn
from torch.nn import functional as F
from up.utils.general.registry_factory import MODULE_ZOO_REGISTRY
from up.models.losses import build_loss
from ..components import Aux_Module, PSPModule, FusionNode

__all__ = ['DFSegDecoder']


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
