# Author: Xiangtai Li
# Email: lixiangtai@sensetime.com
import torch.nn as nn
import torch.nn.functional as F
from eod.utils.general.registry_factory import MODULE_ZOO_REGISTRY
from .seg_resnet import resnet50
from ..components import PSPModule, ConvBnRelu

__all__ = ['ICNetEncoder']


@MODULE_ZOO_REGISTRY.register('icnet_encoder')
class ICNetEncoder(nn.Module):
    def __init__(self, out_planes=512, **kwargs):
        super(ICNetEncoder, self).__init__()
        self.conv_sub1 = nn.Sequential(
            ConvBnRelu(3, 32, 3, 2, 1, normalize=kwargs.get('normalize', {'type': 'solo_bn'})),
            ConvBnRelu(32, 32, 3, 2, 1, normalize=kwargs.get('normalize', {'type': 'solo_bn'})),
            ConvBnRelu(32, 64, 3, 2, 1, normalize=kwargs.get('normalize', {'type': 'solo_bn'}))
        )
        self.backbone = resnet50(**kwargs)
        self.out_planes = out_planes
        self.head = PSPModule(2048, self.out_planes, normalize=kwargs.get('normalize', {'type': 'solo_bn'}))
        self.conv_sub4 = ConvBnRelu(512, 256, 1, normalize=kwargs.get('normalize', {'type': 'solo_bn'}))
        self.conv_sub2 = ConvBnRelu(512, 256, 1, normalize=kwargs.get('normalize', {'type': 'solo_bn'}))

    def forward(self, input):
        img = input["image"]
        size = img.size()[2:]
        # sub 1
        x_sub1_out = self.conv_sub1(img)

        # sub 2
        x_sub2 = F.interpolate(img, scale_factor=0.5, mode='bilinear', align_corners=True)

        x = self.backbone.relu(self.backbone.bn1(self.backbone.conv1(x_sub2)))
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x_sub2_out = self.backbone.layer2(x)

        # sub 4
        x_sub4 = F.interpolate(x_sub2_out, scale_factor=0.5, mode='bilinear', align_corners=True)

        x = self.backbone.layer3(x_sub4)
        x = self.backbone.layer4(x)
        x_sub4_out = self.head(x)

        x_sub4_out = self.conv_sub4(x_sub4_out)
        x_sub2_out = self.conv_sub2(x_sub2_out)

        res = {}
        res['size'] = size
        res['features'] = x_sub1_out, x_sub2_out, x_sub4_out
        res['gt_seg'] = input['gt_seg']
        return res

    def get_outplanes(self):
        return self.out_planes
