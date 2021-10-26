import torch
import torch.nn as nn
import math

from eod.models.heads.utils import accuracy as A  # noqa F401
from eod.models.utils.initializer import initialize_from_cfg
from eod.utils.general.registry_factory import MODULE_ZOO_REGISTRY

from eod.plugins.yolox.models.backbone.cspdarknet import DWConv

from eod.plugins.yolov5.models.components import ConvBnAct


__all__ = ['YoloXHead']


@MODULE_ZOO_REGISTRY.register('YoloXHead')
class YoloXHead(nn.Module):
    def __init__(self,
                 num_classes,
                 num_point=1,
                 width=0.375,
                 inplanes=[256, 512, 1024],
                 outplanes=256,
                 act_fn={'type': 'Silu'},
                 depthwise=False,
                 initializer=None,
                 class_activation='sigmoid',
                 normalize={'type': 'solo_bn'}):
        super(YoloXHead, self).__init__()
        self.prefix = self.__class__.__name__
        self.num_levels = len(inplanes)
        self.num_point = num_point
        class_channel = {'sigmoid': -1, 'softmax': 0}[class_activation] + num_classes

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()
        Conv = DWConv if depthwise else ConvBnAct

        for i in range(self.num_levels):
            inplane = int(inplanes[i])
            outplane = int(outplanes * width)
            self.stems.append(
                ConvBnAct(
                    inplane,
                    outplane,
                    kernel_size=1,
                    stride=1,
                    act_fn=act_fn,
                    normalize=normalize
                )
            )
            self.cls_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            outplane,
                            outplane,
                            kernel_size=3,
                            stride=1,
                            act_fn=act_fn,
                            normalize=normalize
                        ),
                        Conv(
                            outplane,
                            outplane,
                            kernel_size=3,
                            stride=1,
                            act_fn=act_fn,
                            normalize=normalize
                        ),
                    ]
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            outplane,
                            outplane,
                            kernel_size=3,
                            stride=1,
                            act_fn=act_fn,
                            normalize=normalize
                        ),
                        Conv(
                            outplane,
                            outplane,
                            kernel_size=3,
                            stride=1,
                            act_fn=act_fn,
                            normalize=normalize
                        ),
                    ]
                )
            )
            self.cls_preds.append(
                nn.Conv2d(
                    in_channels=outplane,
                    out_channels=self.num_point * class_channel,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.reg_preds.append(
                nn.Conv2d(
                    in_channels=outplane,
                    out_channels=self.num_point * 4,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.obj_preds.append(
                nn.Conv2d(
                    in_channels=outplane,
                    out_channels=self.num_point,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )

        if initializer is not None:
            initialize_from_cfg(self, initializer)
        self.initialize_biases(1e-2)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03

    def forward_net(self, features, idx=0):
        mlvl_preds = []
        for i in range(self.num_levels):
            feat = self.stems[i](features[i])
            cls_feat = self.cls_convs[i](feat)
            loc_feat = self.reg_convs[i](feat)
            cls_pred = self.cls_preds[i](cls_feat)
            loc_pred = self.reg_preds[i](loc_feat)
            obj_pred = self.obj_preds[i](loc_feat)
            mlvl_preds.append((cls_pred, loc_pred, obj_pred))
        return mlvl_preds

    def initialize_biases(self, prior_prob):
        for conv in self.cls_preds:
            b = conv.bias.view(self.num_point, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.obj_preds:
            b = conv.bias.view(self.num_point, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, input):
        features = input['features']
        mlvl_raw_preds = self.forward_net(features)
        output = {}
        output['preds'] = mlvl_raw_preds
        return output
