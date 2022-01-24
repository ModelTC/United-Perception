import torch
import torch.nn as nn
from torch.nn import functional as F
from up.utils.model.normalize import build_norm_layer


class PSPModule(nn.Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """

    def __init__(self, inplanes, out_planes=512, sizes=(1, 2, 3, 6), normalize={'type': 'solo_bn'}):
        super(PSPModule, self).__init__()
        self.stages = []
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


class ConvBnRelu(nn.Module):
    def __init__(self,
                 in_planes,
                 out_planes,
                 ksize,
                 stride=1,
                 pad=0,
                 dilation=1,
                 groups=1,
                 has_bn=True,
                 normalize={'type': 'solo_bn'},
                 has_relu=True,
                 inplace=True,
                 has_bias=False):
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


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
