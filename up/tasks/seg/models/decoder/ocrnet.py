from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np

import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F
from up.utils.model.normalize import build_norm_layer
from up.models.losses import build_loss
from up.utils.general.registry_factory import MODULE_ZOO_REGISTRY


__all__ = ['OCR', 'Seghead']

ALIGN_CORNERS = True


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class SpatialGather_Module(nn.Module):
    """
        Aggregate the context features according to the initial
        predicted probability distribution.
        Employ the soft-weighted method to aggregate the context.
    """
    def __init__(self, cls_num=0, scale=1):
        super(SpatialGather_Module, self).__init__()
        self.cls_num = cls_num
        self.scale = scale

    def forward(self, feats, probs):
        batch_size, c, _, _ = probs.size(0), probs.size(1), probs.size(2), probs.size(3)
        probs = probs.view(batch_size, c, -1)
        feats = feats.view(batch_size, feats.size(1), -1)
        feats = feats.permute(0, 2, 1)  # batch x hw x c
        probs = F.softmax(self.scale * probs, dim=2)  # batch x k x hw
        ocr_context = torch.matmul(probs, feats).permute(0, 2, 1).unsqueeze(3)  # batch x k x c
        # ocr_context = torch.mm(probs.squeeze(0), feats.squeeze(0)).unsqueeze(0).permute(0, 2, 1).unsqueeze(3)
        return ocr_context


class _ObjectAttentionBlock(nn.Module):
    '''
    The basic implementation for object context block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
        bn_type           : specify the bn type
    Return:
        N X C X H X W
    '''
    def __init__(self,
                 in_channels,
                 key_channels,
                 scale=1,
                 normalize={'type': 'solo_bn'}):
        super(_ObjectAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_pixel = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            build_norm_layer(self.key_channels, normalize)[1],
            nn.ReLU(),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            build_norm_layer(self.key_channels, normalize)[1],
            nn.ReLU(),
        )
        self.f_object = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            build_norm_layer(self.key_channels, normalize)[1],
            nn.ReLU(),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            build_norm_layer(self.key_channels, normalize)[1],
            nn.ReLU(),
        )
        self.f_down = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            build_norm_layer(self.key_channels, normalize)[1],
            nn.ReLU(),
        )
        self.f_up = nn.Sequential(
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.in_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            build_norm_layer(self.in_channels, normalize)[1],
            nn.ReLU(),
        )

    def forward(self, x, proxy):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)

        query = self.f_pixel(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_object(proxy).view(batch_size, self.key_channels, -1)
        value = self.f_down(proxy).view(batch_size, self.key_channels, -1)
        value = value.permute(0, 2, 1)

        sim_map = torch.matmul(query, key)
        # sim_map = torch.mm(query.squeeze(0), key.squeeze(0)).unsqueeze(0)

        sim_map = (self.key_channels**-.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        # add bg context ...
        context = torch.matmul(sim_map, value)
        # context = torch.mm(sim_map.squeeze(0), value.squeeze(0)).unsqueeze(0)

        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.key_channels, *x.size()[2:])
        context = self.f_up(context)
        if self.scale > 1:
            upsample = nn.UpsamplingBilinear2d(size=(h, w))
            context = upsample(context)

        return context


class ObjectAttentionBlock2D(_ObjectAttentionBlock):
    def __init__(self,
                 in_channels,
                 key_channels,
                 scale=1,
                 normalize={'type': 'solo_bn'}):
        super(ObjectAttentionBlock2D, self).__init__(in_channels,
                                                     key_channels,
                                                     scale,
                                                     normalize=normalize)


class SpatialOCR_Module(nn.Module):
    """
    Implementation of the OCR module:
    We aggregate the global object representation to update the representation for each pixel.
    """
    def __init__(self,
                 in_channels,
                 key_channels,
                 out_channels,
                 scale=1,
                 dropout=0.1,
                 normalize={'type': 'solo_bn'}):
        super(SpatialOCR_Module, self).__init__()
        self.object_context_block = ObjectAttentionBlock2D(in_channels,
                                                           key_channels,
                                                           scale,
                                                           normalize=normalize
                                                           )
        _in_channels = 2 * in_channels

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(_in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            build_norm_layer(out_channels, normalize)[1],
            nn.ReLU(),
            nn.Dropout2d(dropout)
        )

    def forward(self, feats, proxy_feats):
        context = self.object_context_block(feats, proxy_feats)

        output = self.conv_bn_dropout(torch.cat([context, feats], 1))

        return output


@MODULE_ZOO_REGISTRY.register('ocr_head')
class OCR(nn.Module):
    def __init__(self, inplanes, num_classes=19, loss=None, normalize={'type': 'solo_bn'}):
        super(OCR, self).__init__()
        self.prefix = self.__class__.__name__
        last_inp_channels = np.int(np.sum(inplanes))
        ocr_mid_channels = 512
        ocr_key_channels = 256

        self.conv3x3_ocr = nn.Sequential(
            nn.Conv2d(last_inp_channels, ocr_mid_channels,
                      kernel_size=3, stride=1, padding=1),
            build_norm_layer(ocr_mid_channels, normalize)[1],
            nn.ReLU(),
        )
        self.ocr_gather_head = SpatialGather_Module(num_classes)

        self.ocr_distri_head = SpatialOCR_Module(in_channels=ocr_mid_channels,
                                                 key_channels=ocr_key_channels,
                                                 out_channels=ocr_mid_channels,
                                                 scale=1,
                                                 dropout=0.05,
                                                 normalize=normalize
                                                 )
        self.cls_head = nn.Conv2d(
            ocr_mid_channels, num_classes, kernel_size=1, stride=1, padding=0, bias=True)

        self.aux_head = nn.Sequential(
            nn.Conv2d(last_inp_channels, last_inp_channels,
                      kernel_size=1, stride=1, padding=0),
            build_norm_layer(last_inp_channels, normalize)[1],
            nn.ReLU(),
            nn.Conv2d(last_inp_channels, num_classes,
                      kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.loss = build_loss(loss)

    def forward(self, input):
        feats = input['features']
        input_size = input['size']
        # ocr
        out_aux = self.aux_head(feats)
        # compute contrast feature
        feats = self.conv3x3_ocr(feats)

        context = self.ocr_gather_head(feats, out_aux)
        feats = self.ocr_distri_head(feats, context)

        out = self.cls_head(feats)

        if self.training:
            target = input['gt_semantic_seg']
            upsample = nn.UpsamplingBilinear2d(size=input_size)
            aux_out = upsample(out)
            aux_aux_out = upsample(out_aux)
            out_aux_seg = aux_out, aux_aux_out
            return {f"{self.prefix}.loss": self.loss(out_aux_seg, target), "blob_pred": out_aux_seg[0]}
        else:
            upsample = nn.UpsamplingBilinear2d(size=input_size)
            aux_out = upsample(out)
            return {"blob_pred": aux_out}


@MODULE_ZOO_REGISTRY.register('hr_seg_head')
class Seghead(nn.Module):
    def __init__(self, inplanes , num_classes=19, loss=None, normalize={'type': 'solo_bn'}):
        super(Seghead, self).__init__()
        self.prefix = self.__class__.__name__
        last_inp_channels = np.int(np.sum(inplanes))
        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=last_inp_channels,
                kernel_size=1,
                stride=1,
                padding=0),
            build_norm_layer(last_inp_channels, normalize)[1],
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=num_classes,
                kernel_size=1,
                stride=1,
                padding=0)
        )
        self.loss = build_loss(loss)

    def forward(self, input):
        x = input['features']
        input_size = input['size']
        x = self.last_layer(x)
        if self.training:
            target = input['gt_semantic_seg']
            upsample = nn.UpsamplingBilinear2d(size=input_size)
            out = upsample(x)
            return {f"{self.prefix}.loss": self.loss(out, target), "blob_pred": out}
        else:
            upsample = nn.UpsamplingBilinear2d(size=input_size)
            out = upsample(x)
            return {"blob_pred": out}
