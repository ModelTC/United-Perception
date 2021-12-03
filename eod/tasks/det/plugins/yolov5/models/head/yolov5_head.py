# Standard Library
import math

# Import from third library
import torch
import torch.nn as nn

from eod.utils.model import accuracy as A  # noqa F401
from eod.utils.model.initializer import initialize_from_cfg
from eod.utils.general.registry_factory import MODULE_ZOO_REGISTRY


__all__ = ['YoloV5Head']


@MODULE_ZOO_REGISTRY.register('YoloV5Head')
class YoloV5Head(nn.Module):
    def __init__(self,
                 inplanes,
                 num_classes,
                 num_levels=3,
                 num_anchors_per_level=3,
                 out_strides=[8, 16, 32],
                 initializer=None):
        super(YoloV5Head, self).__init__()

        if not isinstance(inplanes, list):
            inplanes = [inplanes] * num_levels

        self.prefix = self.__class__.__name__
        self.num_classes = num_classes - 1
        self.single_cls = (num_classes == 2)

        self.num_levels = num_levels
        self.num_anchors_per_level = num_anchors_per_level
        self.out_strides = out_strides

        self.cls_subnet = nn.ModuleList()       # obj and cls branch
        self.loc_subnet = nn.ModuleList()       # box branch
        self.obj_subnet = nn.ModuleList()
        for inp in inplanes:
            if self.single_cls:
                self.cls_subnet.append(nn.Conv2d(inp, num_anchors_per_level * 1, 1))
            else:
                self.cls_subnet.append(nn.Conv2d(inp, num_anchors_per_level * self.num_classes, 1))
            self.loc_subnet.append(nn.Conv2d(inp, num_anchors_per_level * 4, 1))
            self.obj_subnet.append(nn.Conv2d(inp, num_anchors_per_level * 1, 1))

        if initializer is not None:
            initialize_from_cfg(self, initializer)
        self._initialize_biases()

    def _initialize_biases(self):  # initialize biases into Detect(), cf is class frequency
        with torch.no_grad():
            for o, cls, s in zip(self.obj_subnet, self.cls_subnet, self.out_strides):  # from
                cls_b = cls.bias.view(self.num_anchors_per_level, -1)  # like conv.bias(255) to (3,85)
                obj_b = o.bias.view(self.num_anchors_per_level, -1)
                obj_b += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
                cls_b += math.log(0.6 / (self.num_classes - 0.99))  # cls
                cls.bias = torch.nn.Parameter(cls_b.view(-1), requires_grad=True)
                o.bias = torch.nn.Parameter(obj_b.view(-1), requires_grad=True)

    def forward_net(self, features):
        mlvl_preds = []
        for i in range(self.num_levels):
            loc_feat = self.loc_subnet[i](features[i])
            cls_feat = self.cls_subnet[i](features[i])
            obj_feat = self.obj_subnet[i](features[i])
            # import ipdb
            # ipdb.set_trace()
            mlvl_preds.append((loc_feat, cls_feat, obj_feat))
        return mlvl_preds

    def forward(self, input):
        features = input['features']
        assert isinstance(features, list), 'Only support list'
        raw_mlvl_preds = self.forward_net(features)
        input['preds'] = raw_mlvl_preds
        return input
