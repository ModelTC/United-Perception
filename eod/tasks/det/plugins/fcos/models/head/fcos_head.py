# Import from third library
import torch
import torch.nn as nn

# Import from eod
from eod.utils.model.normalize import build_conv_norm
from eod.utils.model.initializer import init_bias_focal, initialize_from_cfg
from eod.utils.general.registry_factory import MODULE_ZOO_REGISTRY

__all__ = ['FcosHead']


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


@MODULE_ZOO_REGISTRY.register('FcosHead')
class FcosHead(nn.Module):
    def __init__(self, inplanes, num_conv, feat_planes, num_classes, dense_points, loc_ranges,
                 normalize=None, initializer=None, class_activation=None, init_prior=None, has_mask=False):
        """
        Arguments:
            - inplanes (:obj:`list` or :obj:`int`): input channel,
              which is a number or list contains a single element
            - num_conv (:obj: `int`): num of subnet
            - feat_planes (:obj:`int`): channels of intermediate features
            - num_classes (:obj:`int`): number of classes,
              including the background class, RPN is always 2,
              RetinaNet is (number of all classes + 1)
            - cfg (:obj:`dict`): config for training or test
            - normalize (:obj:`dict`): config of Normalization Layer
            - initializer (:obj:`dict`): config for module parameters initialization
        """
        super(FcosHead, self).__init__()
        self.num_classes = num_classes
        self.dense_points = dense_points
        self.loc_ranges = loc_ranges
        self.has_mask = has_mask
        if isinstance(inplanes, list):
            assert len(inplanes) == 1, 'single input is expected, but found:{} '.format(inplanes)
            inplanes = inplanes[0]
        assert isinstance(inplanes, int)
        if not isinstance(feat_planes, list):
            feat_planes = [feat_planes] * num_conv
        self.cls_subnet = nn.Sequential()
        self.box_subnet = nn.Sequential()
        for i in range(num_conv):
            self.cls_subnet.add_module(
                f'{i}',
                build_conv_norm(
                    inplanes,
                    feat_planes[i],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    normalize=normalize,
                    activation=True))
            self.box_subnet.add_module(
                f'{i}',
                build_conv_norm(
                    inplanes, feat_planes[i], kernel_size=3, stride=1, padding=1, normalize=normalize, activation=True))
            inplanes = feat_planes[i]

        self.cls_loss_type = class_activation
        self.init_prior = init_prior

        C = {'sigmoid': -1, 'softmax': 0}[self.cls_loss_type] + self.num_classes

        self.cls_subnet_pred = nn.Conv2d(feat_planes[-1], self.dense_points * C, kernel_size=3, stride=1, padding=1)
        self.box_subnet_pred = nn.Conv2d(feat_planes[-1], self.dense_points * 4, kernel_size=3, stride=1, padding=1)
        self.centerness = nn.Conv2d(feat_planes[-1], self.dense_points, kernel_size=3, stride=1, padding=1)
        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(len(self.loc_ranges))])
        self.relu = nn.ReLU(inplace=True)

        initialize_from_cfg(self, initializer)
        init_bias_focal(self.cls_subnet_pred, self.cls_loss_type, init_prior, self.num_classes)

    def forward_net(self, x, idx=0):
        cls_feature = self.cls_subnet(x)
        box_feature = self.box_subnet(x)
        cls_pred = self.cls_subnet_pred(cls_feature)
        loc_pred = self.relu(self.scales[idx](self.box_subnet_pred(box_feature)))
        centerness = self.centerness(box_feature)
        if self.has_mask:
            return cls_pred, loc_pred, centerness, box_feature
        return cls_pred, loc_pred, centerness

    def forward(self, input):
        features = input['features']
        mlvl_preds = [self.forward_net(x, idx) for idx, x in enumerate(features)]
        return {'mlvl_preds': mlvl_preds}
