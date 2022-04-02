import torch
import torch.nn as nn
from torch.nn import functional as F
from up.utils.general.registry_factory import MODULE_ZOO_REGISTRY
from up.models.losses import build_loss
from up.tasks.seg.models.components import ConvBnRelu

__all__ = ['SegFormerHead']


@MODULE_ZOO_REGISTRY.register('segformer_decoder')
class SegFormerHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """

    def __init__(self,
                 in_channels,
                 in_index=[0, 1, 2, 3],
                 channels=256,
                 dropout_ratio=0.1,
                 num_classes=19,
                 norm_cfg=None,
                 align_corners=False,
                 loss=None,
                 interpolate_mode='bilinear'):
        super(SegFormerHead, self).__init__()
        self.prefix = self.__class__.__name__
        self.in_channels = in_channels
        self.in_index = in_index
        self.channels = channels
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.norm_cfg = norm_cfg
        self.loss = build_loss(loss)
        self.align_corners = align_corners

        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None
        self.fp16_enabled = False
        self.interpolate_mode = interpolate_mode

        num_inputs = len(self.in_channels)
        assert num_inputs == len(self.in_index)

        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvBnRelu(
                    in_planes=self.in_channels[i],
                    out_planes=self.channels,
                    ksize=1,
                    stride=1,
                    normalize=self.norm_cfg
                ))

        self.fusion_conv = ConvBnRelu(
            in_planes=self.channels * num_inputs,
            out_planes=self.channels,
            ksize=1,
            normalize=self.norm_cfg
        )

    def forward(self, inputs):
        features = inputs['features']
        size = inputs['size']
        x = [features[i] for i in self.in_index]  # len=4, 1/4,1/8,1/16,1/32
        outs = []
        for idx in range(len(x)):
            x_single = x[idx]
            conv = self.convs[idx]
            outs.append(
                F.interpolate(
                    input=conv(x_single),
                    size=x[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        out = self.fusion_conv(torch.cat(outs, dim=1))

        if self.dropout is not None:
            out = self.dropout(out)
        out = self.conv_seg(out)

        out = F.interpolate(out, size=size, mode='bilinear', align_corners=self.align_corners)

        if self.training:
            loss = self.loss(out, inputs['gt_semantic_seg'])
            return {f"{self.prefix}.loss": loss, "blob_pred": out}
        else:
            return {"blob_pred": out}
