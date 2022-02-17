# Author: Xiangtai Li
# Email: lixiangtai@sensetime.com

import torch.nn as nn
import torch.nn.functional as F

from up.utils.general.registry_factory import MODULE_ZOO_REGISTRY
from up.models.losses import build_loss
from ..components import CascadeFeatureFusion

__all__ = ['ICnetDecoder']


@MODULE_ZOO_REGISTRY.register('icnet_decoder')
class ICnetDecoder(nn.Module):
    def __init__(self, num_classes, inner_planes=128, inplanes=512, normalize={'type': 'solo_bn'}, loss=None):
        super(ICnetDecoder, self).__init__()
        self.prefix = self.__class__.__name__
        self.cff_12 = CascadeFeatureFusion(128, 64, inner_planes, num_classes, normalize)
        self.cff_24 = CascadeFeatureFusion(256, 256, inner_planes, num_classes, normalize)

        self.conv_cls = nn.Conv2d(inner_planes, num_classes, 1, bias=False)
        self.loss = build_loss(loss)

    def forward(self, x):
        x_sub1, x_sub2, x_sub4 = x['features']
        size = x['size']
        x_cff_24, x_24_cls = self.cff_24(x_sub4, x_sub2)
        pred = []
        pred.append(x_24_cls)
        x_cff_12, x_12_cls = self.cff_12(x_cff_24, x_sub1)
        pred.append(x_12_cls)

        up_x2 = F.interpolate(x_cff_12, scale_factor=2, mode='bilinear', align_corners=True)
        up_x2 = self.conv_cls(up_x2)
        pred.append(up_x2)
        up_x8 = F.interpolate(up_x2, scale_factor=4, mode='bilinear', align_corners=True)
        pred.append(up_x8)
        # 1 -> 1/4 -> 1/8 -> 1/16
        pred.reverse()

        if self.training:
            gt_seg = x['gt_semantic_seg']
            loss = self.loss(pred, gt_seg)
            return {f"{self.prefix}.loss": loss, "blob_pred": pred}
        else:
            return {"blob_pred": F.upsample(pred[0], size=size, mode='bilinear', align_corners=True)}
