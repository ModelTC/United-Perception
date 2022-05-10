import torch.nn as nn
import json
import numpy as np
from up.utils.general.registry_factory import MODULE_ZOO_REGISTRY
from up.utils.model.initializer import init_bias_focal, initialize_from_cfg

__all__ = ['AnchorHeadSingle']


@MODULE_ZOO_REGISTRY.register('base_anchor_head')
class BaseAnchorHead(nn.Module):
    def __init__(self, inplanes, num_classes):
        super(BaseAnchorHead, self).__init__()
        self.prefix = self.__class__.__name__
        self.num_classes = num_classes
        assert self.num_classes > 0

        if isinstance(inplanes, list):
            inplanes_length = len(inplanes)
            for i in range(1, inplanes_length):
                if inplanes[i] != inplanes[0]:
                    raise ValueError('list inplanes elements are inconsistent with {}'.format(inplanes[i]))
            inplanes = inplanes[0]
        assert isinstance(inplanes, int)
        self.inplanes = inplanes

    def forward(self, **kwargs):
        raise NotImplementedError


@MODULE_ZOO_REGISTRY.register('anchor_head_single')
class AnchorHeadSingle(BaseAnchorHead):
    def __init__(self, inplanes, num_classes, base_anchors_file, code_size, num_dir_bins,
                 use_direction_classifier=True, initializer=None, class_activation='sigmoid', init_prior=0.01):
        super(AnchorHeadSingle, self).__init__(inplanes, num_classes)

        self.class_activation = class_activation
        with open(base_anchors_file, 'r') as f:
            anchor_classes = np.array(json.load(f))
        num_anchors_per_location = []
        for anchor_class in anchor_classes:
            num_rotations = len(anchor_class['anchor_rotations'])
            num_size = len(anchor_class['anchor_sizes'])
            num_class = len(anchor_class['anchor_bottom_heights'])
            num_anchors_per_location.append(num_rotations * num_size * num_class)
        self.num_anchors = sum(num_anchors_per_location)

        self.cls_subnet_pred = nn.Conv2d(
            inplanes, self.num_anchors * self.num_classes, kernel_size=1)
        self.box_subnet_pred = nn.Conv2d(
            inplanes, self.num_anchors * code_size, kernel_size=1)

        if use_direction_classifier:
            self.dir_subnet_pred = nn.Conv2d(
                inplanes, self.num_anchors * num_dir_bins, kernel_size=1)
        else:
            self.dir_subnet_pred = None

        init_bias_focal(self.cls_subnet_pred, self.class_activation, init_prior, self.num_classes)
        initialize_from_cfg(self.box_subnet_pred, initializer)

    def forward_net(self, x):
        cls_preds = self.cls_subnet_pred(x)
        box_preds = self.box_subnet_pred(x)
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()

        if self.dir_subnet_pred is not None:
            dir_cls_preds = self.dir_subnet_pred(x)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
        else:
            dir_cls_preds = None
        return cls_preds, box_preds, dir_cls_preds

    def forward(self, input):
        features = input['spatial_features_2d']
        output = {}
        cls_preds, box_preds, dir_cls_preds = self.forward_net(features)
        output.update({'cls_preds': cls_preds, 'box_preds': box_preds,
                      'dir_pred': dir_cls_preds, 'num_anchors': self.num_anchors})
        return output
