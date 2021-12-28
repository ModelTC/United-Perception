# Author: Xiangtai Li
# Email: lixiangtai@sensetime.com
import torch
import torch.nn as nn
import torch.nn.functional as F
from eod.utils.general.registry_factory import MODULE_ZOO_REGISTRY
from eod.utils.model.normalize import build_norm_layer

__all__ = ['ICNetEncoder']

from .seg_resnet import resnet50


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


class CascadeFeatureFusion(nn.Module):
    """CFF Unit"""

    def __init__(self, low_channels, high_channels, out_channels, nclass, normalize={'type': 'solo_bn'}):
        super(CascadeFeatureFusion, self).__init__()
        self.conv_low = nn.Sequential(
            nn.Conv2d(low_channels, out_channels, 3, padding=2, dilation=2, bias=False),
            build_norm_layer(out_channels, normalize)[1]
        )
        self.conv_high = nn.Sequential(
            nn.Conv2d(high_channels, out_channels, 1, bias=False),
            build_norm_layer(out_channels, normalize)[1]
        )
        self.conv_low_cls = nn.Conv2d(out_channels, nclass, 1, bias=False)

    def forward(self, x_low, x_high):
        x_low = F.interpolate(x_low, size=x_high.size()[2:], mode='bilinear', align_corners=True)
        x_low = self.conv_low(x_low)
        x_high = self.conv_high(x_high)
        x = x_low + x_high
        x = F.relu(x, inplace=True)
        x_low_cls = self.conv_low_cls(x_low)

        return x, x_low_cls


class ConvBnRelu(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, stride=1, pad=0, dilation=1,
                 groups=1, has_bn=True, normalize={'type': 'solo_bn'}, bn_eps=1e-5,
                 has_relu=True, inplace=True, has_bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize,
                              stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = build_norm_layer(out_planes, normalize)[1]
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)

        return x


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
