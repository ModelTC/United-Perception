# Author: Xiangtai Li
# Implementation of my paper: Semantic Flow for Fast and Accurate Scene Parsing
# Email: lixiangtai@sensetime.com

import torch
import torch.nn as nn
import torch.nn.functional as F
from up.utils.model.normalize import build_norm_layer
from up.utils.general.registry_factory import MODULE_ZOO_REGISTRY
from up.models.losses import build_loss
from ..components import ConvBnRelu, PSPModule

__all__ = ['SFSegDecoder']


class AlignedModule(nn.Module):
    def __init__(self, inplane, outplane, kernel_size=3):
        super(AlignedModule, self).__init__()
        self.down_h = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.down_l = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.flow_make = nn.Conv2d(outplane * 2, 2, kernel_size=kernel_size, padding=1, bias=False)

    def forward(self, x):
        low_feature, h_feature = x
        h_feature_orign = h_feature
        h, w = low_feature.size()[2:]
        size = (h, w)
        low_feature = self.down_l(low_feature)
        h_feature = self.down_h(h_feature)
        h_feature = F.upsample(h_feature, size=size, mode="bilinear", align_corners=True)
        flow = self.flow_make(torch.cat([h_feature, low_feature], 1))
        h_feature = self.flow_warp(h_feature_orign, flow, size=size)

        return h_feature

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        w = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        h = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((h.unsqueeze(2), w.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid)
        return output


class AlignHead(nn.Module):

    def __init__(self, inplane, num_class, normalize={'type': 'solo_bn'}, fpn_inplanes=[256, 512, 1024, 2048],
                 fpn_dim=256):
        super(AlignHead, self).__init__()
        self.ppm = PSPModule(inplane, out_planes=fpn_dim, normalize=normalize)
        self.fpn_in = []
        for fpn_inplane in fpn_inplanes[:-1]:
            self.fpn_in.append(
                nn.Sequential(
                    nn.Conv2d(fpn_inplane, fpn_dim, 1),
                    build_norm_layer(fpn_dim, normalize)[1],
                    nn.ReLU(inplace=False)
                )
            )
        self.fpn_in = nn.ModuleList(self.fpn_in)

        self.fpn_out = []
        self.fpn_out_align = []
        for i in range(len(fpn_inplanes) - 1):
            self.fpn_out.append(nn.Sequential(
                ConvBnRelu(fpn_dim, fpn_dim, 1),
            ))
            self.fpn_out_align.append(
                AlignedModule(inplane=fpn_dim, outplane=fpn_dim // 2)
            )

        self.fpn_out = nn.ModuleList(self.fpn_out)
        self.fpn_out_align = nn.ModuleList(self.fpn_out_align)

        self.conv_last = nn.Sequential(
            ConvBnRelu(len(fpn_inplanes) * fpn_dim, fpn_dim, 1),
            nn.Conv2d(fpn_dim, num_class, kernel_size=1)
        )

    def forward(self, conv_out):
        psp_out = self.ppm(conv_out[-1])

        f = psp_out
        fpn_feature_list = [psp_out]
        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x)  # lateral branch
            f = self.fpn_out_align[i]([conv_x, f])
            f = conv_x + f
            fpn_feature_list.append(self.fpn_out[i](f))

        fpn_feature_list.reverse()  # [P2 - P5]
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]

        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(nn.functional.interpolate(
                fpn_feature_list[i],
                output_size,
                mode='bilinear', align_corners=True))

        fusion_out = torch.cat(fusion_list, 1)
        x = self.conv_last(fusion_out)

        return x


@MODULE_ZOO_REGISTRY.register('sfnet_decoder')
class SFSegDecoder(nn.Module):
    def __init__(self, inplanes=512, inner_planes=64, num_classes=19, normalize={'type': 'solo_bn'}, sync_bn=True,
                 loss=None):
        super(SFSegDecoder, self).__init__()
        self.prefix = self.__class__.__name__
        self.head = AlignHead(inplane=inplanes, num_class=num_classes, normalize={'type': 'solo_bn'},
                              fpn_inplanes=[128, 256, 512], fpn_dim=inner_planes)
        self.loss = build_loss(loss)

    def forward(self, x):
        x3, x4, x5 = x['features']
        size = x['size']
        gt_seg = x['gt_semantic_seg']
        pred = self.head([x3, x4, x5])

        pred = F.upsample(pred, size=size, mode='bilinear', align_corners=True)

        if self.training:
            loss = self.loss(pred, gt_seg)
            return {f"{self.prefix}.loss": loss, "blob_pred": pred}
        else:
            return {"blob_pred": pred}
