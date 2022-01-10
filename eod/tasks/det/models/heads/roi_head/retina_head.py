import torch.nn as nn

from eod.utils.model.normalize import build_conv_norm, build_norm_layer
from eod.utils.model.initializer import init_bias_focal, initialize_from_cfg
from eod.utils.general.registry_factory import MODULE_ZOO_REGISTRY
from eod.utils.general.global_flag import FP16_FLAG


__all__ = ['BaseNet', 'RetinaSubNet', 'RetinaHeadWithBN', 'RetinaHeadWithBNIOU', 'RetinaHeadWithBNSep']


class BaseNet(nn.Module):
    """
    Head for the first stage detection task

    .. note::

        0 is always for the background class.
    """

    def __init__(self, inplanes, num_classes, num_levels=5):
        """
        Arguments:
            - inplanes (:obj:`list` or :obj:`int`): input channel,
              which is a number or list contains a single element
            - num_classes (:obj:`int`): number of classes,
              including the background class, RPN is always 2,
              RetinaNet is (number of all classes + 1)
            - cfg (:obj:`dict`): config for training or test
        """
        super(BaseNet, self).__init__()
        self.prefix = self.__class__.__name__
        self.num_classes = num_classes
        self.inplanes = inplanes
        self.num_levels = num_levels

    def forward(self, input):
        features = input['features']
        mlvl_raw_preds = [self.forward_net(features[lvl], lvl) for lvl in range(self.num_levels)]
        output = {}
        output['preds'] = mlvl_raw_preds
        return output


@MODULE_ZOO_REGISTRY.register('RetinaSubNet')
class RetinaSubNet(BaseNet):
    """
    Classify and regress Anchors direclty (all classes)
    """

    def __init__(self,
                 inplanes,
                 feat_planes,
                 num_classes,
                 normalize=None,
                 initializer=None,
                 num_conv=4,
                 class_activation='sigmoid',
                 init_prior=0.01,
                 num_anchors=9,
                 num_levels=5):
        super(RetinaSubNet, self).__init__(inplanes, num_classes, num_levels)

        inplanes = self.inplanes
        self.class_activation = class_activation
        self.num_anchors = num_anchors

        self.cls_subnet = self.build(num_conv, inplanes, feat_planes, normalize)
        self.box_subnet = self.build(num_conv, inplanes, feat_planes, normalize)
        class_channel = {'sigmoid': -1, 'softmax': 0}[self.class_activation] + self.num_classes
        self.class_channel = class_channel
        self.cls_subnet_pred = nn.Conv2d(
            feat_planes, self.num_anchors * class_channel, kernel_size=3, stride=1, padding=1)
        self.box_subnet_pred = nn.Conv2d(
            feat_planes, self.num_anchors * 4, kernel_size=3, stride=1, padding=1)

        initialize_from_cfg(self, initializer)
        init_bias_focal(self.cls_subnet_pred, self.class_activation, init_prior, self.num_classes)

    def build(self, num_conv, inplanes, feat_planes, normalize):
        layers = []
        if normalize is None:
            for i in range(num_conv):
                layers.append(nn.Conv2d(inplanes, feat_planes, kernel_size=3, stride=1, padding=1))
                layers.append(nn.ReLU(inplace=True))
                inplanes = feat_planes
        else:
            for i in range(num_conv):
                module = build_conv_norm(inplanes,
                                         feat_planes,
                                         kernel_size=3,
                                         stride=1,
                                         padding=1,
                                         normalize=normalize,
                                         activation=True)
                for child in module.children():
                    layers.append(child)
        return nn.Sequential(*layers)

    def forward_net(self, x, lvl=None):
        cls_feature = self.cls_subnet(x)
        box_feature = self.box_subnet(x)
        cls_pred = self.cls_subnet_pred(cls_feature)
        loc_pred = self.box_subnet_pred(box_feature)
        if FP16_FLAG.fp16:
            cls_pred = cls_pred.float()
            loc_pred = loc_pred.float()
        return cls_pred, loc_pred


@MODULE_ZOO_REGISTRY.register('RetinaHeadWithIOU')
class RetinaHeadWithIOU(RetinaSubNet):
    """
    Classify and regress Anchors direclty (all classes)
    """

    def __init__(self,
                 inplanes,
                 feat_planes,
                 num_classes,
                 normalize=None,
                 initializer=None,
                 num_conv=4,
                 class_activation='sigmoid',
                 init_prior=0.01,
                 num_anchors=1,
                 num_levels=5):
        """
        Implementation for ATSS https://arxiv.org/abs/1912.02424
        """
        super(RetinaHeadWithIOU, self).__init__(inplanes,
                                                feat_planes,
                                                num_classes,
                                                normalize,
                                                initializer,
                                                num_conv,
                                                class_activation,
                                                init_prior,
                                                num_anchors,
                                                num_levels)
        assert num_levels is not None, "num_levels must be provided !!!"
        self.iou_pred = nn.Conv2d(feat_planes, self.num_anchors, kernel_size=3, stride=1, padding=1)
        initialize_from_cfg(self, initializer)
        init_bias_focal(self.cls_subnet_pred, self.class_activation, init_prior, self.num_classes)

    def forward_net(self, x, lvl=None):
        cls_feature = self.cls_subnet(x)
        box_feature = self.box_subnet(x)
        cls_pred = self.cls_subnet_pred(cls_feature)
        loc_pred = self.box_subnet_pred(box_feature)
        iou_pred = self.iou_pred(box_feature)
        if FP16_FLAG.fp16:
            cls_pred = cls_pred.float()
            loc_pred = loc_pred.float()
            iou_pred = iou_pred.float()
        return cls_pred, loc_pred, iou_pred


@MODULE_ZOO_REGISTRY.register('RetinaHeadWithBN')
class RetinaHeadWithBN(RetinaSubNet):
    def __init__(self,
                 inplanes,
                 feat_planes,
                 num_classes,
                 normalize={'type': 'solo_bn'},
                 initializer=None,
                 num_conv=4,
                 num_levels=5,
                 class_activation='sigmoid',
                 init_prior=0.01,
                 num_anchors=9):
        self.num_levels = num_levels
        super(RetinaHeadWithBN, self).__init__(inplanes,
                                               feat_planes,
                                               num_classes,
                                               normalize,
                                               initializer,
                                               num_conv,
                                               class_activation,
                                               init_prior,
                                               num_anchors,
                                               num_levels)

        assert num_levels is not None, "num_levels must be provided !!!"

    def build(self, num_conv, input_planes, feat_planes, normalize):
        mlvl_heads = nn.ModuleList()
        for lvl in range(self.num_levels):
            layers = []
            inplanes = input_planes
            for conv_idx in range(num_conv):
                if lvl == 0:
                    layers.append(nn.Sequential(
                        nn.Conv2d(inplanes, feat_planes, kernel_size=3, stride=1, padding=1, bias=False),
                        build_norm_layer(feat_planes, normalize)[1],
                        nn.ReLU(inplace=True)))
                    inplanes = feat_planes
                else:
                    layers.append(nn.Sequential(
                        mlvl_heads[-1][conv_idx][0],
                        build_norm_layer(feat_planes, normalize)[1],
                        nn.ReLU(inplace=True)))
                    inplanes = feat_planes
            mlvl_heads.append(nn.Sequential(*layers))
        return mlvl_heads

    def forward_net(self, x, lvl=None):
        cls_feature = self.cls_subnet[lvl](x)
        box_feature = self.box_subnet[lvl](x)
        cls_pred = self.cls_subnet_pred(cls_feature)
        loc_pred = self.box_subnet_pred(box_feature)
        if FP16_FLAG.fp16:
            cls_pred = cls_pred.float()
            loc_pred = loc_pred.float()
        return cls_pred, loc_pred


@MODULE_ZOO_REGISTRY.register('RetinaHeadWithBNSep')
class RetinaHeadWithBNSep(RetinaSubNet):
    def __init__(self,
                 inplanes,
                 feat_planes,
                 num_classes,
                 normalize={'type': 'solo_bn'},
                 initializer=None,
                 num_conv=4,
                 num_levels=5,
                 class_activation='sigmoid',
                 init_prior=0.01,
                 num_anchors=9):
        self.num_levels = num_levels
        super(RetinaHeadWithBNSep, self).__init__(inplanes,
                                                  feat_planes,
                                                  num_classes,
                                                  normalize,
                                                  initializer,
                                                  num_conv,
                                                  class_activation,
                                                  init_prior,
                                                  num_anchors,
                                                  num_levels)

        assert num_levels is not None, "num_levels must be provided !!!"

    def build(self, num_conv, input_planes, feat_planes, normalize):
        mlvl_heads = nn.ModuleList()
        for lvl in range(self.num_levels):
            layers = []
            inplanes = input_planes
            for conv_idx in range(num_conv):
                layers.append(nn.Sequential(
                    nn.Conv2d(inplanes, feat_planes, kernel_size=3, stride=1, padding=1, bias=False),
                    build_norm_layer(feat_planes, normalize)[1],
                    nn.ReLU(inplace=True)))
                inplanes = feat_planes
            mlvl_heads.append(nn.Sequential(*layers))
        return mlvl_heads

    def forward_net(self, x, lvl=None):
        cls_feature = self.cls_subnet[lvl](x)
        box_feature = self.box_subnet[lvl](x)
        cls_pred = self.cls_subnet_pred(cls_feature)
        loc_pred = self.box_subnet_pred(box_feature)
        return cls_pred, loc_pred


@MODULE_ZOO_REGISTRY.register('RetinaHeadWithBNIOU')
class RetinaHeadWithBNIOU(RetinaHeadWithBN):
    """
    Classify and regress Anchors direclty (all classes)
    """

    def __init__(self,
                 inplanes,
                 feat_planes,
                 num_classes,
                 normalize={'type': 'solo_bn'},
                 initializer=None,
                 num_conv=4,
                 num_levels=5,
                 class_activation='sigmoid',
                 init_prior=0.01,
                 num_anchors=9):
        self.num_levels = num_levels
        super(RetinaHeadWithBNIOU, self).__init__(inplanes,
                                                  feat_planes,
                                                  num_classes,
                                                  normalize,
                                                  initializer,
                                                  num_conv,
                                                  num_levels,
                                                  class_activation,
                                                  init_prior,
                                                  num_anchors)
        assert num_levels is not None, "num_levels must be provided !!!"
        self.iou_pred = nn.Conv2d(feat_planes, self.num_anchors, kernel_size=3, stride=1, padding=1)
        initialize_from_cfg(self, initializer)
        init_bias_focal(self.cls_subnet_pred, self.class_activation, init_prior, self.num_classes)

    def forward_net(self, x, lvl=None):
        cls_feature = self.cls_subnet[lvl](x)
        box_feature = self.box_subnet[lvl](x)
        cls_pred = self.cls_subnet_pred(cls_feature)
        loc_pred = self.box_subnet_pred(box_feature)
        iou_pred = self.iou_pred(box_feature)
        if FP16_FLAG.fp16:
            cls_pred = cls_pred.float()
            loc_pred = loc_pred.float()
            iou_pred = iou_pred.float()
        return cls_pred, loc_pred, iou_pred
