import torch.nn as nn
from up.tasks.nas.bignas.models.heads.big_roi_head import big_roi_head, bignas_roi_head
from up.tasks.nas.bignas.models.ops.dynamic_blocks import DynamicConvBlock
from up.utils.model.initializer import initialize_from_cfg, init_bias_focal
from up.tasks.nas.bignas.models.ops.dynamic_utils import copy_module_weights


class BigRetinaHeadWithBN(nn.Module):
    def __init__(self,
                 inplanes,
                 feat_planes,
                 num_classes,
                 normalize={'type': 'dynamic_solo_bn'},
                 initializer=None,
                 num_conv=4,
                 num_level=5,
                 class_activation='sigmoid',
                 init_prior=0.01,
                 num_anchors=9,
                 pred_conv_kernel_size=3,
                 share_subnet=False,
                 **kwargs):
        self.num_level = num_level
        super(BigRetinaHeadWithBN, self).__init__()

        self.class_activation = class_activation
        self.pred_conv_kernel_size = pred_conv_kernel_size
        self.num_anchors = num_anchors
        self.share_subnet = share_subnet
        self.num_classes = num_classes
        class_channel = {'sigmoid': -1, 'softmax': 0}[self.class_activation] + self.num_classes
        self.class_channel = class_channel
        self.init_prior = init_prior
        self.normalize = normalize
        self.num_conv = num_conv
        self.num_level = num_level

        assert num_level is not None, "num_level must be provided !!!"
        self.cls_subnet = big_roi_head(inplanes=inplanes, num_levels=num_level, num_conv=num_conv,
                                       normalize=normalize, **kwargs)
        if not self.share_subnet:
            self.box_subnet = big_roi_head(inplanes=inplanes, num_levels=num_level, num_conv=num_conv,
                                           normalize=normalize, **kwargs)
        self.cls_subnet_pred = DynamicConvBlock(
            feat_planes, self.num_anchors * self.class_channel, kernel_size_list=pred_conv_kernel_size, stride=1,
            use_bn=False, act_func='', bias=True)
        self.box_subnet_pred = DynamicConvBlock(
            feat_planes, self.num_anchors * 4, kernel_size_list=pred_conv_kernel_size, stride=1,
            use_bn=False, act_func='', bias=True)
        initialize_from_cfg(self, initializer)
        init_bias_focal(self.cls_subnet_pred, self.class_activation, init_prior, num_classes)

    def forward(self, input):
        features = input['features']
        mlvl_raw_preds = [self.forward_net(features[lvl], lvl) for lvl in range(self.num_level)]
        output = {}
        output['preds'] = mlvl_raw_preds
        return output

    def forward_net(self, x, lvl=None):
        cls_feature = self.cls_subnet.mlvl_heads[lvl](x)
        if self.share_subnet:
            box_feature = cls_feature
        else:
            box_feature = self.box_subnet.mlvl_heads[lvl](x)
        cls_pred = self.cls_subnet_pred(cls_feature)
        loc_pred = self.box_subnet_pred(box_feature)
        return cls_pred.float(), loc_pred.float()  # the return type must be fp32 for fp16 support !!!

    def sample_active_subnet_weights(self, subnet_settings, inplanes):
        cls_subnet = self.cls_subnet.sample_active_subnet_weights(subnet_settings['cls_subnet'], inplanes)
        if not self.share_subnet:
            box_subnet = self.box_subnet.sample_active_subnet_weights(subnet_settings['box_subnet'], inplanes)
        static_model = BignasRetinaHeadWithBN(inplanes,
                                              subnet_settings['cls_subnet'].get('out_channel')[-1],
                                              self.num_classes,
                                              normalize=self.normalize,
                                              num_conv=self.num_conv,
                                              num_level=self.num_conv,
                                              class_activation=self.class_activation,
                                              num_anchors=self.num_anchors,
                                              pred_conv_kernel_size=self.pred_conv_kernel_size,
                                              kernel_size=subnet_settings['cls_subnet'].get('kernel_size'),
                                              depth_cls=subnet_settings['cls_subnet'].get('depth'),
                                              depth_box=subnet_settings['box_subnet'].get('depth'),
                                              out_channel_cls=subnet_settings['cls_subnet'].get('out_channel'),
                                              out_channel_box=subnet_settings['box_subnet'].get('out_channel')
                                              )
        static_model.cls_subnet = cls_subnet
        if not self.share_subnet:
            static_model.box_subnet = box_subnet
        copy_module_weights(static_model.cls_subnet_pred, self.cls_subnet_pred)
        copy_module_weights(static_model.box_subnet_pred, self.box_subnet_pred)
        return static_model


class BignasRetinaHeadWithBN(nn.Module):
    """
    Classify and regress Anchors direclty (all classes)
    """

    def __init__(self,
                 inplanes,
                 feat_planes,
                 num_classes,
                 normalize={'type': 'dynamic_solo_bn'},
                 initializer=None,
                 num_conv=4,
                 num_level=5,
                 class_activation='sigmoid',
                 init_prior=0.01,
                 num_anchors=9,
                 share_subnet=False,
                 pred_conv_kernel_size=3,
                 kernel_size=[3],
                 depth_cls=[4],
                 depth_box=[4],
                 out_channel_cls=[256],
                 out_channel_box=[256]):
        super(BignasRetinaHeadWithBN, self).__init__()
        self.class_activation = class_activation
        self.num_anchors = num_anchors
        self.share_subnet = share_subnet
        self.num_classes = num_classes
        self.pred_conv_kernel_size = pred_conv_kernel_size
        class_channel = {'sigmoid': -1, 'softmax': 0}[self.class_activation] + self.num_classes
        self.class_channel = class_channel

        assert num_level is not None, "num_levels must be provided !!!"
        assert num_conv == depth_box[0] and num_conv == depth_cls[0]
        if self.share_subnet:
            assert feat_planes == out_channel_box[0] and feat_planes == out_channel_cls[-1]

        self.cls_subnet = bignas_roi_head(inplanes=inplanes, num_levels=num_level, num_conv=sum(depth_cls),
                                          normalize=normalize,
                                          kernel_size=kernel_size, depth=depth_cls, out_channel=out_channel_cls)
        if not self.share_subnet:
            self.box_subnet = bignas_roi_head(inplanes=inplanes, num_levels=num_level, num_conv=sum(depth_box),
                                              normalize=normalize,
                                              kernel_size=kernel_size, depth=depth_box, out_channel=out_channel_box)

        self.cls_subnet_pred = nn.Conv2d(
            out_channel_cls[-1], self.num_anchors * class_channel,
            kernel_size=pred_conv_kernel_size, stride=1, padding=1, bias=True)
        self.box_subnet_pred = nn.Conv2d(
            out_channel_box[-1], self.num_anchors * 4,
            kernel_size=pred_conv_kernel_size, stride=1, padding=1, bias=True)

        initialize_from_cfg(self, initializer)
        init_bias_focal(self.cls_subnet_pred, self.class_activation, init_prior, self.num_classes)

    def forward(self, input):
        features = input['features']
        mlvl_raw_preds = [self.forward_net(features[lvl], lvl) for lvl in range(self.num_level)]
        output = {}
        output['preds'] = mlvl_raw_preds
        return output

    def forward_net(self, x, lvl=None):
        cls_feature = self.cls_subnet.mlvl_heads[lvl](x)
        if self.share_subnet:
            box_feature = cls_feature
        else:
            box_feature = self.box_subnet.mlvl_heads[lvl](x)
        cls_pred = self.cls_subnet_pred(cls_feature)
        loc_pred = self.box_subnet_pred(box_feature)
        return cls_pred.float(), loc_pred.float()  # the return type must be fp32 for fp16 support !!!
