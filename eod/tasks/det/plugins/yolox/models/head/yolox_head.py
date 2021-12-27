import torch
import torch.nn as nn
import math

from eod.utils.model import accuracy as A  # noqa F401
from eod.utils.model.initializer import initialize_from_cfg
from eod.utils.general.registry_factory import MODULE_ZOO_REGISTRY
from eod.tasks.det.plugins.yolox.models.backbone.cspdarknet import DWConv
from eod.tasks.det.plugins.yolov5.models.components import ConvBnAct
from eod.utils.model.normalize import build_norm_layer
from eod.utils.model.act_fn import build_act_fn


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
                 normalize={'type': 'solo_bn'},
                 init_prior=0.01):
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
        self.initialize_biases(init_prior)

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


@MODULE_ZOO_REGISTRY.register('YoloXHeadShare')
class YoloXHeadShare(nn.Module):
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
                 normalize={'type': 'solo_bn'},
                 init_prior=0.01,
                 num_conv=2):
        super(YoloXHeadShare, self).__init__()
        self.prefix = self.__class__.__name__
        self.num_levels = len(inplanes)
        self.num_point = num_point
        class_channel = {'sigmoid': -1, 'softmax': 0}[class_activation] + num_classes

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.stems = nn.ModuleList()
        outplane = int(outplanes * width)
        for i in range(self.num_levels):
            inplane = int(inplanes[i])
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
        self.cls_convs = self.build(num_conv, outplane, outplane, normalize, act_fn)
        self.reg_convs = self.build(num_conv, outplane, outplane, normalize, act_fn)
        self.cls_pred = nn.Conv2d(in_channels=outplane,
                                  out_channels=self.num_point * class_channel,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0)
        self.reg_pred = nn.Conv2d(in_channels=outplane,
                                  out_channels=self.num_point * 4,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0)
        self.obj_pred = nn.Conv2d(in_channels=outplane,
                                  out_channels=self.num_point,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0)

        if initializer is not None:
            initialize_from_cfg(self, initializer)
        self.initialize_biases(init_prior)

    def initialize_biases(self, prior_prob):
        b = self.cls_pred.bias.view(self.num_point, -1)
        b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
        self.cls_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        b = self.obj_pred.bias.view(self.num_point, -1)
        b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
        self.obj_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def build(self, num_conv, input_planes, feat_planes, normalize, act_fn):
        mlvl_heads = nn.ModuleList()
        for lvl in range(self.num_levels):
            layers = []
            inplanes = input_planes
            for conv_idx in range(num_conv):
                if lvl == 0:
                    layers.append(nn.Sequential(
                        nn.Conv2d(inplanes, feat_planes, kernel_size=3, stride=1, padding=1, bias=False),
                        build_norm_layer(feat_planes, normalize)[1],
                        build_act_fn(act_fn)[1]))
                    inplanes = feat_planes
                else:
                    layers.append(nn.Sequential(
                        mlvl_heads[-1][conv_idx][0],
                        build_norm_layer(feat_planes, normalize)[1],
                        build_act_fn(act_fn)[1]))
                    inplanes = feat_planes
            mlvl_heads.append(nn.Sequential(*layers))
        return mlvl_heads

    def forward_net(self, features, idx=0):
        mlvl_preds = []
        for i in range(self.num_levels):
            feat = self.stems[i](features[i])
            cls_feat = self.cls_convs[i](feat)
            loc_feat = self.reg_convs[i](feat)
            cls_pred = self.cls_pred(cls_feat)
            loc_pred = self.reg_pred(loc_feat)
            obj_pred = self.obj_pred(loc_feat)
            mlvl_preds.append((cls_pred, loc_pred, obj_pred))
        return mlvl_preds

    def forward(self, input):
        features = input['features']
        mlvl_raw_preds = self.forward_net(features)
        output = {}
        output['preds'] = mlvl_raw_preds
        return output
