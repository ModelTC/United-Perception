import torch.nn as nn

from up.utils.model.normalize import build_conv_norm, build_norm_layer
from up.utils.model.initializer import init_bias_focal, initialize_from_cfg
from up.utils.general.registry_factory import MODULE_ZOO_REGISTRY
from up.utils.general.global_flag import FP16_FLAG


__all__ = ['BaseNet', 'RetinaSubNet', 'RetinaHeadWithBN', 'RetinaHeadWithBNIOU', 'RetinaHeadWithBNSep', 'NaiveRPN']


class BaseNet(nn.Module):
    """
    Head for the first stage detection task

    .. note::

        0 is always for the background class.
    """

    def __init__(self, inplanes, num_classes, num_level=5):
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
        self.num_level = num_level

        if isinstance(inplanes, list):
            inplanes_length = len(inplanes)
            for i in range(1, inplanes_length):
                if inplanes[i] != inplanes[0]:
                    raise ValueError('list inplanes elements are inconsistent with {}'.format(inplanes[i]))
            inplanes = inplanes[0]
        assert isinstance(inplanes, int)
        self.inplanes = inplanes

    def forward(self, input):
        features = input['features']
        mlvl_raw_preds = [self.forward_net(features[lvl], lvl) for lvl in range(self.num_level)]
        output = {}
        output['preds'] = mlvl_raw_preds
        if 'RPN' in self.prefix:
            output.update({'rpn_preds': output['preds']})
        output.update({'deploy_output_node': output['preds']})
        return output


@MODULE_ZOO_REGISTRY.register('NaiveRPN')
class NaiveRPN(BaseNet):
    """
    Classify Anchors (2 cls): foreground or background. This module is usually
    used with :class:`~pod.models.heads.bbox_head.bbox_head.BboxNet`
    """

    def __init__(self,
                 inplanes,
                 feat_planes,
                 num_classes,
                 num_anchors,
                 class_activation,
                 num_level=5,
                 Normalize=None,
                 initializer=None):
        assert num_classes == 2
        super(NaiveRPN, self).__init__(inplanes, num_classes, num_level)

        inplanes = self.inplanes
        num_level = self.num_level
        self.class_activation = class_activation
        self.conv3x3 = nn.Conv2d(inplanes, feat_planes, kernel_size=3, stride=1, padding=1)
        self.relu3x3 = nn.ReLU(inplace=True)

        class_channel = {'sigmoid': 1, 'softmax': 2}[self.class_activation]
        self.conv_cls = nn.Conv2d(feat_planes, num_anchors * class_channel, kernel_size=1, stride=1)
        self.conv_loc = nn.Conv2d(feat_planes, num_anchors * 4, kernel_size=1, stride=1)

        initialize_from_cfg(self, initializer)

    def forward_net(self, x, lvl=None):
        x = self.conv3x3(x)
        x = self.relu3x3(x)
        cls_pred = self.conv_cls(x)
        loc_pred = self.conv_loc(x)
        if FP16_FLAG.fp16:
            cls_pred = cls_pred.float()
            loc_pred = loc_pred.float()
        return cls_pred, loc_pred


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
                 num_level=5,
                 share_subnet=False,
                 share_location=True):
        super(RetinaSubNet, self).__init__(inplanes, num_classes, num_level)

        inplanes = self.inplanes
        self.share_location = share_location

        self.class_activation = class_activation
        self.num_anchors = num_anchors
        self.share_subnet = share_subnet

        self.cls_subnet = self.build(num_conv, inplanes, feat_planes, normalize)
        if not self.share_subnet:
            self.box_subnet = self.build(num_conv, inplanes, feat_planes, normalize)
        class_channel = {'sigmoid': -1, 'softmax': 0}[self.class_activation] + self.num_classes
        self.class_channel = class_channel
        self.num_loc = 1
        if not self.share_location:
            self.num_loc = self.class_channel
        self.cls_subnet_pred = nn.Conv2d(
            feat_planes, self.num_anchors * class_channel, kernel_size=3, stride=1, padding=1)
        self.box_subnet_pred = nn.Conv2d(
            feat_planes, self.num_anchors * 4 * self.num_loc, kernel_size=3, stride=1, padding=1)

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
                inplanes = feat_planes
        return nn.Sequential(*layers)

    def forward_net(self, x, lvl=None):
        cls_feature = self.cls_subnet(x)
        if self.share_subnet:
            box_feature = cls_feature
        else:
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
                 num_level=5,
                 share_subnet=False,
                 share_location=True):
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
                                                num_level,
                                                share_subnet,
                                                share_location)
        assert num_level is not None, "num_level must be provided !!!"
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
                 num_level=5,
                 class_activation='sigmoid',
                 init_prior=0.01,
                 num_anchors=9,
                 share_subnet=False,
                 share_location=True):
        self.num_level = num_level
        super(RetinaHeadWithBN, self).__init__(inplanes,
                                               feat_planes,
                                               num_classes,
                                               normalize,
                                               initializer,
                                               num_conv,
                                               class_activation,
                                               init_prior,
                                               num_anchors,
                                               num_level,
                                               share_subnet,
                                               share_location)

        assert num_level is not None, "num_level must be provided !!!"

    def build(self, num_conv, input_planes, feat_planes, normalize):
        mlvl_heads = nn.ModuleList()
        for lvl in range(self.num_level):
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
        if self.share_subnet:
            box_feature = cls_feature
        else:
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
                 num_level=5,
                 class_activation='sigmoid',
                 init_prior=0.01,
                 num_anchors=9,
                 share_subnet=False,
                 share_location=True):
        self.num_level = num_level
        super(RetinaHeadWithBNSep, self).__init__(inplanes,
                                                  feat_planes,
                                                  num_classes,
                                                  normalize,
                                                  initializer,
                                                  num_conv,
                                                  class_activation,
                                                  init_prior,
                                                  num_anchors,
                                                  num_level,
                                                  share_subnet,
                                                  share_location)

        assert num_level is not None, "num_level must be provided !!!"

    def build(self, num_conv, input_planes, feat_planes, normalize):
        mlvl_heads = nn.ModuleList()
        for lvl in range(self.num_level):
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
        if self.share_subnet:
            box_feature = cls_feature
        else:
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
                 num_level=5,
                 class_activation='sigmoid',
                 init_prior=0.01,
                 num_anchors=9,
                 share_subnet=False,
                 share_location=True):
        self.num_level = num_level
        super(RetinaHeadWithBNIOU, self).__init__(inplanes,
                                                  feat_planes,
                                                  num_classes,
                                                  normalize,
                                                  initializer,
                                                  num_conv,
                                                  num_level,
                                                  class_activation,
                                                  init_prior,
                                                  num_anchors,
                                                  share_subnet,
                                                  share_location)
        assert num_level is not None, "num_level must be provided !!!"
        self.iou_pred = nn.Conv2d(feat_planes, self.num_anchors, kernel_size=3, stride=1, padding=1)
        initialize_from_cfg(self, initializer)
        init_bias_focal(self.cls_subnet_pred, self.class_activation, init_prior, self.num_classes)

    def forward_net(self, x, lvl=None):
        cls_feature = self.cls_subnet[lvl](x)
        if self.share_subnet:
            cls_feature = cls_feature
        else:
            box_feature = self.box_subnet[lvl](x)
        cls_pred = self.cls_subnet_pred(cls_feature)
        loc_pred = self.box_subnet_pred(box_feature)
        iou_pred = self.iou_pred(box_feature)
        if FP16_FLAG.fp16:
            cls_pred = cls_pred.float()
            loc_pred = loc_pred.float()
            iou_pred = iou_pred.float()
        return cls_pred, loc_pred, iou_pred
