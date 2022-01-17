import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from eod import extensions as E
from eod.utils.model import accuracy as A
from eod.tasks.det.models.utils.assigner import map_rois_to_level
from eod.tasks.det.models.losses import ohem_loss
from eod.utils.model.initializer import init_weights_normal, initialize_from_cfg, init_bias_focal
from eod.models.losses import build_loss
from eod.utils.general.registry_factory import MODULE_ZOO_REGISTRY

# Import from local
from .bbox import build_bbox_predictor, build_bbox_supervisor

__all__ = ['FC', 'BboxNet', 'RFCN']


class BboxNet(nn.Module):
    """
    classify proposals and compute their bbox regression.
    """

    def __init__(self, inplanes, num_classes, cfg, with_drp=False):
        """
        Arguments:
            - inplanes (:obj:`list` or :obj:`int`): input channel, which is a number or
              list contains a single element
            - num_classes (:obj:`int`):  number of classes, including the background class
            - cfg (:obj:`dict`): configuration
        """
        super(BboxNet, self).__init__()
        self.prefix = 'BboxNet'
        self.tocaffe = False
        self.num_classes = num_classes

        if isinstance(inplanes, list):
            assert len(inplanes) == 1, 'single input is expected, but found:{} '.format(inplanes)
            inplanes = inplanes[0]
        assert isinstance(inplanes, int)
        self.inplanes = inplanes

        self.supervisor = build_bbox_supervisor(cfg['bbox_supervisor'])
        self.predictor = build_bbox_predictor(cfg['bbox_predictor'])

        cfg_fpn = cfg.get('fpn', None)
        self.with_drp = with_drp and cfg_fpn is not None
        if self.with_drp:
            self.roipool = nn.ModuleList([E.build_generic_roipool(cfg['roipooling'])
                                          for _ in range(len(cfg_fpn['fpn_levels']))])
        else:
            self.roipool = E.build_generic_roipool(cfg['roipooling'])
        self.pool_size = cfg['roipooling']['pool_size']

        self.cls_loss = build_loss(cfg['cls_loss'])
        self.loc_loss = build_loss(cfg['loc_loss'])

        self.cfg = copy.deepcopy(cfg)

    def roi_extractor(self, mlvl_rois, mlvl_features, mlvl_strides):
        """Get RoIPooling features
        """
        raise NotImplementedError

    def forward_net(self, rois, x, stride):
        """
        Arguments:
            - rois (FloatTensor): rois in a sinle layer
            - x (FloatTensor): features in a single layer
            - stride: stride for current layer
        """
        raise NotImplementedError

    def forward(self, input):
        """
        Forward tensor to generate predictions

        .. note::

            Training phase and Testing phase have different behavior,
            For training, it samples positive and negative RoIs, then
            calculates their target and loss. For testing,it gets the
            predicted boxes of proposals based on classification score
            and box regression.

        Arguments:
            - input (:obj:`dict`): data from prev module

        Returns:
            - output (:obj:`dict`): output k, v is different for training and testing

        Input example::

            # input
            {
                # (list of FloatTensor): input feature layers,
                # for C4 from backbone, others from FPN
                'features': ..,
                # (list of int): input feature layers,
                # for C4 from backbone, others from FPN
                'strides': [],
                # (list of FloatTensor)
                # [B, 5] (reiszed_h, resized_w, scale_factor, origin_h, origin_w)
                'image_info': [],
                # (FloatTensor): boxes from last module,
                # [N, >=7] (batch_ix, x1, y1, x2, y2, score, cls)
                'dt_bboxes': [],
                # (list of FloatTensor or None): [B] [num_gts, 5] (x1, y1, x2, y2, label)
                'gt_bboxes': [],
                # (list of FloatTensor or None): [B] [num_igs, 4] (x1, y1, x2, y2)
                'gt_ignores': []
            }

        Output example::

            # training output
            {'BboxNet.cls_loss': <tensor>, 'BboxNet.loc_loss': <tensor>, 'BboxNet.accuracy': <tensor>}

            # testing output
            {
                # (FloatTensor), predicted boxes [N, >=7] (batch_ix, x1, y1, x2, y2, score, cls)
                'dt_bboxes': <tensor>,
                # list of list of ndarray, softmax score of predicted boxes, (N, num_classes+1)
                'pred_cls_prob': <tensor>
            }
        """
        output = {}
        if self.training:
            losses = self.get_loss(input)
            output.update(losses)
        else:
            results = self.get_bboxes(input)
            output.update(results)
        return output

    def mlvl_predict(self, x_rois, x_features, x_strides, levels):
        """Predict results level by level"""
        mlvl_rois = []
        mlvl_features = []
        mlvl_strides = []
        for lvl_idx in levels:
            if x_rois[lvl_idx].numel() > 0:
                mlvl_rois.append(x_rois[lvl_idx])
                mlvl_features.append(x_features[lvl_idx])
                mlvl_strides.append(x_strides[lvl_idx])
        assert len(mlvl_rois) > 0, "No rois provided for second stage"
        if self.tocaffe and torch.is_tensor(mlvl_strides[0]):
            mlvl_strides = [int(s) for s in mlvl_strides]
        pooled_feats = self.roi_extractor(mlvl_rois, mlvl_features, mlvl_strides)
        pred_cls, pred_loc = self.forward_net(pooled_feats)
        return pred_cls, pred_loc

    def get_head_output(self, rois, features, strides):
        """
        Assign rois to each level and predict

        Note:
            1.The recovering process is not supposed to be handled in this function,
              because ONNX DON'T support indexing;
            2.numerical type of cls_pred and loc_pred must be float for fp32 support !!!

        Returns:
            rois (FloatTensor): assigned rois
            cls_pred (FloatTensor, fp32): prediction of classification of assigned rois
            loc_pred (FloatTensor, fp32): prediction of localization of assigned rois
            recover_inds (LongTensor): indices of recovering input rois from assigned rois
        """
        if self.cfg.get('fpn', None):
            # assign rois and targets to each level
            fpn = self.cfg['fpn']
            if self.tocaffe and not self.training:
                # to save memory
                # if rois.numel() > 0:
                # rois = rois[0:1]
                # make sure that all pathways included in the computation graph
                mlvl_rois, recover_inds = [rois] * len(fpn['fpn_levels']), None
            else:
                mlvl_rois, recover_inds = map_rois_to_level(fpn['fpn_levels'], fpn['base_scale'], rois)
            cls_pred, loc_pred = self.mlvl_predict(mlvl_rois, features, strides, fpn['fpn_levels'])
            rois = torch.cat(mlvl_rois, dim=0)
        else:
            assert len(features) == 1 and len(strides) == 1, \
                'please setup `fpn` first if you want to use pyramid features'
            if self.tocaffe and torch.is_tensor(strides):
                strides = strides.tolist()
            pooled_feats = self.roi_extractor([rois], features, strides)
            cls_pred, loc_pred = self.forward_net(pooled_feats)
            recover_inds = torch.arange(rois.shape[0], device=rois.device)
        return rois, cls_pred.float(), loc_pred.float(), recover_inds

    def get_loss(self, input):
        """
        Arguments:
            input['features'] (list): input feature layers, for C4 from backbone, others from FPN
            input['strides'] (list): strides of input feature layers
            input['image_info'] (list of FloatTensor): [B, 5] (reiszed_h, resized_w, scale_factor, origin_h, origin_w)
            input['dt_bboxes'] (FloatTensor): proposals, [N, >=7] (batch_ix, x1, y1, x2, y2, score, cls)
            input['gt_bboxes'] (list of FloatTensor): [B] [num_gts, 5] (x1, y1, x2, y2, label)
            input['gt_ignores'] (list of FloatTensor): [B] [num_igs, 4] (x1, y1, x2, y2)

        Returns:
            sample_record (list of tuple): [B, (pos_inds, pos_target_gt_inds)], saved for mask/keypoint head
            cls_loss, loc_loss, acc (FloatTensor)
        """
        features = input['features']
        strides = input['strides']
        image_info = input['image_info']
        B = len(image_info)

        # cls_target (LongTensor): [R]
        # loc_target (FloatTensor): [R, 4]
        # loc_weight (FloatTensor): [R, 4]
        sample_record, sampled_rois, cls_target, loc_target, loc_weight = self.supervisor.get_targets(input)
        rois, cls_pred, loc_pred, recover_inds = self.get_head_output(sampled_rois, features, strides)
        cls_pred = cls_pred[recover_inds]
        loc_pred = loc_pred[recover_inds]

        cls_inds = cls_target
        if self.cfg.get('share_location', 'False'):
            cls_inds = cls_target.clamp(max=0)

        N = loc_pred.shape[0]
        loc_pred = loc_pred.reshape(N, -1, 4)
        inds = torch.arange(N, dtype=torch.int64, device=loc_pred.device)
        if self.cls_loss.activation_type == 'sigmoid' and not self.cfg.get('share_location', 'False'):
            cls_inds = cls_inds - 1
        loc_pred = loc_pred[inds, cls_inds].reshape(-1, 4)

        reduction_override = None if 'ohem' not in self.cfg else 'none'
        normalizer_override = cls_target.shape[0] if 'ohem' not in self.cfg else None

        loc_loss_key_fields = getattr(self.loc_loss, "key_fields", set())
        loc_loss_kwargs = {}
        if "anchor" in loc_loss_key_fields:
            loc_loss_kwargs["anchor"] = rois
        if "bbox_normalize" in loc_loss_key_fields:
            loc_loss_kwargs["bbox_normalize"] = self.cfg.get('bbox_normalize', None)

        cls_loss = self.cls_loss(cls_pred, cls_target, reduction_override=reduction_override)
        loc_loss = self.loc_loss(loc_pred * loc_weight, loc_target,
                                 reduction_override=reduction_override,
                                 normalizer_override=normalizer_override, **loc_loss_kwargs)

        if self.cfg.get('ohem', None):
            cls_loss, loc_loss = ohem_loss(cls_loss, loc_loss, B, self.cfg['ohem'])

        if self.cls_loss.activation_type == 'softmax':
            acc = A.accuracy(cls_pred, cls_target)[0]
        else:
            try:
                acc = A.binary_accuracy(cls_pred, self.cls_loss.expand_target)[0]
            except:  # noqa
                acc = cls_pred.new_zeros(1)
        if self.cfg.get('grid', None):
            loc_loss = loc_loss * 0
        return {
            'sample_record': sample_record,
            self.prefix + '.cls_loss': cls_loss,
            self.prefix + '.loc_loss': loc_loss,
            self.prefix + '.accuracy': acc
        }

    def get_bboxes(self, input):
        features = input['features']
        strides = input['strides']
        image_info = input['image_info']
        rois = input['dt_bboxes']
        rois, cls_pred, loc_pred, recover_inds = self.get_head_output(rois, features, strides)
        if self.cls_loss.activation_type == 'sigmoid':
            cls_pred = torch.sigmoid(cls_pred)
        elif self.cls_loss.activation_type == 'softmax':
            cls_pred = F.softmax(cls_pred, dim=1)
        else:
            cls_pred = self.cls_loss.get_activation(cls_pred)

        start_idx = 0 if self.cls_loss.activation_type == 'sigmoid' else 1
        output = self.predictor.predict(rois, (cls_pred, loc_pred), image_info, start_idx=start_idx)

        if self.tocaffe:
            output[self.prefix + '.blobs.classification'] = cls_pred
            output[self.prefix + '.blobs.localization'] = loc_pred
        return output


@MODULE_ZOO_REGISTRY.register('BboxFC')
class FC(BboxNet):
    """
    FC head to predict RoIs' feature
    """

    def __init__(self, inplanes, feat_planes, num_classes, cfg, initializer=None, with_drp=False):
        """
        Arguments:
            - inplanes (:obj:`list` or :obj:`int`): input channel, which is a number or list
              contains a single element
            - feat_planes (:obj:`int`): channels of intermediate features
            - num_classes (:obj:`int`): number of classes, including the background class
            - cfg (:obj:`dict`): config for training or test
            - initializer (:obj:`dict`): config for module parameters initialization
              e.g. {'method': msra}

        `FC example <http://gitlab.bj.sensetime.com/project-spring/pytorch-object-detection/blob/master
        /configs/baselines/faster-rcnn-R50-FPN-1x.yaml#L132-164>`_
        """
        super(FC, self).__init__(inplanes, num_classes, cfg, with_drp=with_drp)

        inplanes = self.inplanes
        self.relu = nn.ReLU(inplace=True)
        self.fc6 = nn.Linear(self.pool_size * self.pool_size * inplanes, feat_planes)
        self.fc7 = nn.Linear(feat_planes, feat_planes)

        # cls_out_channel = num_classes if self.cls_loss.activation_type == 'softmax' else num_classes - 1
        if self.cls_loss.activation_type == 'sigmoid':
            cls_out_channel = num_classes - 1
        elif self.cls_loss.activation_type == 'softmax':
            cls_out_channel = num_classes
        else:
            cls_out_channel = self.cls_loss.get_channel_num(num_classes)
        self.fc_rcnn_cls = nn.Linear(feat_planes, cls_out_channel)
        if self.cfg.get('share_location', False):
            self.fc_rcnn_loc = nn.Linear(feat_planes, 4)
        else:
            self.fc_rcnn_loc = nn.Linear(feat_planes, cls_out_channel * 4)

        initialize_from_cfg(self, initializer)
        init_weights_normal(self.fc_rcnn_cls, 0.01)
        init_weights_normal(self.fc_rcnn_loc, 0.001)

        if 'sigmoid' in self.cls_loss.activation_type:
            init_prior = self.cls_loss.init_prior
            init_bias_focal(self.fc_rcnn_cls, 'sigmoid', init_prior, num_classes)

    def roi_extractor(self, mlvl_rois, mlvl_features, mlvl_strides):
        if self.tocaffe:
            if not isinstance(mlvl_strides, list):
                mlvl_strides = mlvl_strides.tolist()
            else:
                mlvl_strides = [int(_) for _ in mlvl_strides]
        if self.with_drp:
            pooled_feats = [self.roipool[idx](*args)
                            for idx, args in enumerate(zip(mlvl_rois, mlvl_features, mlvl_strides))]
        else:
            pooled_feats = [self.roipool(*args) for args in zip(mlvl_rois, mlvl_features, mlvl_strides)]

        # ONNX concat requires at least one tensor
        if len(pooled_feats) == 1:
            return pooled_feats[0]
        return torch.cat(pooled_feats, dim=0)

    def forward_net(self, pooled_feats):
        c = pooled_feats.numel() // pooled_feats.shape[0]
        x = pooled_feats.view(-1, c)
        x = self.relu(self.fc6(x))
        x = self.relu(self.fc7(x))
        cls_pred = self.fc_rcnn_cls(x)
        loc_pred = self.fc_rcnn_loc(x)
        return cls_pred, loc_pred


@MODULE_ZOO_REGISTRY.register('BboxRFCN')
class RFCN(BboxNet):
    """
    RFCN-style to predict RoIs' feature
    """

    def __init__(self, inplanes, feat_planes, num_classes, cfg, initializer=None):
        """
        Arguments:
            - inplanes (:obj:`list` or :obj:`int`): input channel, which is a number or
              list contains a single element
            - feat_planes (:obj:`int`): channels of intermediate features
            - num_classes (:obj:`int`): number of classes, including the background class
            - cfg (:obj:`dict`): config for training or test
            - initializer (:obj:`dict`): config for module parameters initialization

        `RFCN example <http://gitlab.bj.sensetime.com/project-spring/pytorch-object-detection/blob/
        master/configs/baselines/rfcn-R101-1x.yaml#L117-145>`_
        """
        super(RFCN, self).__init__(inplanes, num_classes, cfg)

        ps = self.pool_size
        inplanes = self.inplanes

        self.add_relu_after_feature_conv = self.cfg.get("add_relu_after_feature_conv", False)
        self.relu = nn.ReLU(inplace=True)

        self.new_conv = nn.Conv2d(inplanes, feat_planes, kernel_size=1, bias=False)
        cls_out_channel = num_classes if self.cls_loss.activation_type == 'softmax' else num_classes - 1
        self.rfcn_score = nn.Conv2d(feat_planes, ps * ps * cls_out_channel, kernel_size=1)
        if self.cfg.get('share_location', False):
            self.rfcn_bbox = nn.Conv2d(feat_planes, ps * ps * 4, kernel_size=1)
        else:
            self.rfcn_bbox = nn.Conv2d(feat_planes, ps * ps * 4 * cls_out_channel, kernel_size=1)
        self.pool = nn.AvgPool2d((ps, ps), stride=(ps, ps))

        initialize_from_cfg(self, initializer)

    def roi_extractor(self, mlvl_rois, mlvl_features, mlvl_strides):
        pooled_cls = []
        pooled_loc = []
        for rois, feat, stride in zip(mlvl_rois, mlvl_features, mlvl_strides):
            x = self.new_conv(feat)
            if self.add_relu_after_feature_conv:
                x = self.relu(x)
            cls = self.rfcn_score(x)
            loc = self.rfcn_bbox(x)
            pooled_cls.append(self.roipool(rois, cls, stride))
            pooled_loc.append(self.roipool(rois, loc, stride))
        # ONNX concat requires at least one tensor
        if len(pooled_cls) == 1:
            return pooled_cls[0], pooled_loc[0]
        pooled_cls = torch.cat(pooled_cls, dim=0)
        pooled_loc = torch.cat(pooled_loc, dim=0)
        return pooled_cls, pooled_loc

    def forward_net(self, pooled_feats):
        pooled_cls, pooled_loc = pooled_feats

        x_cls = self.pool(pooled_cls)
        x_loc = self.pool(pooled_loc)

        # ONNX is too fool to convert squeeze
        cls_pred = x_cls.view(-1, x_cls.shape[1])
        loc_pred = x_loc.view(-1, x_loc.shape[1])
        # cls_pred = x_cls.squeeze(dim=-1).squeeze(dim=-1)
        # loc_pred = x_loc.squeeze(dim=-1).squeeze(dim=-1)

        return cls_pred, loc_pred
