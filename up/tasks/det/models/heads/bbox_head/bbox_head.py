import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from up import extensions as E
from up.utils.model import accuracy as A
from up.tasks.det.models.utils.assigner import map_rois_to_level
from up.tasks.det.models.losses import ohem_loss
from up.utils.model.initializer import init_weights_normal, initialize_from_cfg, init_bias_focal
from up.models.losses import build_loss
from up.utils.general.registry_factory import MODULE_ZOO_REGISTRY, MODULE_PROCESS_REGISTRY
from up.utils.general.global_flag import QUANT_FLAG

# Import from local
from .bbox import build_bbox_predictor, build_bbox_supervisor

__all__ = ['BboxNet', 'FC', 'RFCN', 'Res5', 'Bbox_PreProcess', 'Bbox_PostProcess',
           'FC_PreProcess', 'Res5_PreProcess', 'RFCN_PreProcess']


class Bbox_PreProcess(nn.Module):
    def __init__(self, cfg, with_drp=False):
        super(Bbox_PreProcess, self).__init__()

        self.tocaffe = False
        self.supervisor = build_bbox_supervisor(cfg['bbox_supervisor'])

        cfg_fpn = cfg.get('fpn', None)
        self.with_drp = with_drp and cfg_fpn is not None
        if self.with_drp:
            self.roipool = nn.ModuleList([E.build_generic_roipool(cfg['roipooling'])
                                          for _ in range(len(cfg_fpn['fpn_levels']))])
        else:
            self.roipool = E.build_generic_roipool(cfg['roipooling'])

        self.cfg = copy.deepcopy(cfg)

    def forward(self, input):
        if self.training:
            features = input['features']
            strides = input['strides']

            # cls_target (LongTensor): [R]
            # loc_target (FloatTensor): [R, 4]
            # loc_weight (FloatTensor): [R, 4]
            sample_record, sampled_rois, cls_target, loc_target, loc_weight = self.supervisor.get_targets(input)
            rois = sampled_rois
        else:
            features = input['features']
            strides = input['strides']
            rois = input['dt_bboxes']

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
            pooled_feats = self.mlvl_predict(mlvl_rois, features, strides, fpn['fpn_levels'])
            rois = torch.cat(mlvl_rois, dim=0)
        else:
            assert len(features) == 1 and len(strides) == 1, \
                'please setup `fpn` first if you want to use pyramid features'
            if torch.is_tensor(strides):
                strides = strides.tolist()
            pooled_feats = self.roi_extractor([rois], features, strides)
            recover_inds = torch.arange(rois.shape[0], device=rois.device)

        if self.training:
            return {'pooled_feats': pooled_feats,
                    'rois': rois,
                    'recover_inds': recover_inds,
                    'sample_record': sample_record,
                    'cls_target': cls_target,
                    'loc_target': loc_target,
                    'loc_weight': loc_weight}
        else:
            return {'pooled_feats': pooled_feats,
                    'rois': rois}

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
        # pred_cls, pred_loc = self.forward_net(pooled_feats)
        return pooled_feats

    def roi_extractor(self, mlvl_rois, mlvl_features, mlvl_strides):
        """Get RoIPooling features
        """
        raise NotImplementedError


class BboxNet(nn.Module):
    """
    classify proposals and compute their bbox regression.
    """

    def __init__(self, inplanes, num_classes, cfg):
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

        self.class_activation = cfg.get('class_activation', 'softmax')
        self.cls_out_channel = {'sigmoid': -1, 'softmax': 0}[self.class_activation] + self.num_classes

        if isinstance(inplanes, list):
            assert len(inplanes) == 1, 'single input is expected, but found:{} '.format(inplanes)
            inplanes = inplanes[0]
        assert isinstance(inplanes, int)
        self.inplanes = inplanes

        self.pool_size = cfg['roipooling']['pool_size']
        self.cfg = copy.deepcopy(cfg)

    def forward(self, input):
        cls_pred, loc_pred = self.forward_net(input['pooled_feats'])
        bbox_preds = [cls_pred, loc_pred]
        return {'bbox_preds': bbox_preds, 'deploy_output_node': bbox_preds}

    def forward_net(self, rois, x, stride):
        """
        Arguments:
            - rois (FloatTensor): rois in a sinle layer
            - x (FloatTensor): features in a single layer
            - stride: stride for current layer
        """
        raise NotImplementedError


@MODULE_PROCESS_REGISTRY.register('Bbox_Post')
class Bbox_PostProcess(nn.Module):
    def __init__(self, cfg):
        super(Bbox_PostProcess, self).__init__()

        self.prefix = 'BboxNet'
        self.tocaffe = False

        self.predictor = build_bbox_predictor(cfg['bbox_predictor'])

        self.cls_loss = build_loss(cfg['cls_loss'])
        self.loc_loss = build_loss(cfg['loc_loss'])

        self.cfg = copy.deepcopy(cfg)

    def forward(self, input):
        if self.training:
            cls_pred = input['bbox_preds'][0]
            loc_pred = input['bbox_preds'][1]
            rois = input['rois']
            recover_inds = input['recover_inds']
            sample_record = input['sample_record']
            cls_target = input['cls_target']
            loc_target = input['loc_target']
            loc_weight = input['loc_weight']

            image_info = input['image_info']
            B = len(image_info)

            cls_pred = cls_pred.float()
            loc_pred = loc_pred.float()

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

        else:
            cls_pred = input['bbox_preds'][0]
            loc_pred = input['bbox_preds'][1]
            rois = input['rois']
            image_info = input['image_info']
            cls_pred = cls_pred.float()
            loc_pred = loc_pred.float()

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


@MODULE_PROCESS_REGISTRY.register('BboxFC_Pre')
class FC_PreProcess(Bbox_PreProcess):
    def __init__(self, cfg, with_drp=False):
        super(FC_PreProcess, self).__init__(cfg, with_drp)

    def roi_extractor(self, mlvl_rois, mlvl_features, mlvl_strides):
        if not QUANT_FLAG.flag and not self.tocaffe:
            if isinstance(mlvl_strides, list):
                mlvl_strides = [int(s) for s in mlvl_strides]
            else:
                mlvl_strides = mlvl_strides.tolist()
        if self.with_drp:
            pooled_feats = [self.roipool[idx](*args)
                            for idx, args in enumerate(zip(mlvl_rois, mlvl_features, mlvl_strides))]
        else:
            pooled_feats = [self.roipool(*args) for args in zip(mlvl_rois, mlvl_features, mlvl_strides)]

        # ONNX concat requires at least one tensor
        if len(pooled_feats) == 1:
            return pooled_feats[0]
        return torch.cat(pooled_feats, dim=0)


@MODULE_ZOO_REGISTRY.register('BboxFC')
class FC(BboxNet):
    """
    FC head to predict RoIs' feature
    """

    def __init__(self, inplanes, feat_planes, num_classes, cfg, initializer=None):
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
        super(FC, self).__init__(inplanes, num_classes, cfg)

        inplanes = self.inplanes
        self.relu = nn.ReLU(inplace=True)
        self.fc6 = nn.Linear(self.pool_size * self.pool_size * inplanes, feat_planes)
        self.fc7 = nn.Linear(feat_planes, feat_planes)

        self.fc_rcnn_cls = nn.Linear(feat_planes, self.cls_out_channel)
        if self.cfg.get('share_location', False):
            self.fc_rcnn_loc = nn.Linear(feat_planes, 4)
        else:
            self.fc_rcnn_loc = nn.Linear(feat_planes, self.cls_out_channel * 4)

        initialize_from_cfg(self, initializer)
        init_weights_normal(self.fc_rcnn_cls, 0.01)
        init_weights_normal(self.fc_rcnn_loc, 0.001)

        if self.class_activation == 'sigmoid':
            init_prior = self.cfg.get('init_prior', 0.01)
            init_bias_focal(self.fc_rcnn_cls, 'sigmoid', init_prior, num_classes)

    def forward_net(self, pooled_feats):
        c = pooled_feats.numel() // pooled_feats.shape[0]
        x = pooled_feats.view(-1, c)
        x = self.relu(self.fc6(x))
        x = self.relu(self.fc7(x))
        cls_pred = self.fc_rcnn_cls(x)
        loc_pred = self.fc_rcnn_loc(x)
        return cls_pred, loc_pred


@MODULE_PROCESS_REGISTRY.register('BboxRes5_Pre')
class Res5_PreProcess(Bbox_PreProcess):
    def __init__(self, cfg, with_drp=False):
        super(Res5_PreProcess, self).__init__(cfg, with_drp)

    def roi_extractor(self, mlvl_rois, mlvl_features, mlvl_strides):
        pooled_feats = [self.roipool(*args) for args in zip(mlvl_rois, mlvl_features, mlvl_strides)]
        # ONNX concat requires at least one tensor
        if len(pooled_feats) == 1:
            return pooled_feats[0]
        return torch.cat(pooled_feats, dim=0)


@MODULE_ZOO_REGISTRY.register('BboxRes5')
class Res5(BboxNet):
    """
    Conv5 head to predict RoIs' feature
    """

    def __init__(self,
                 inplanes,
                 backbone,
                 num_classes,
                 cfg,
                 deformable=False,
                 normalize={'type': 'freeze_bn'},
                 initializer=None):
        """
        Arguments:
            - inplanes (:obj:`list` or :obj:`int`): input channel, which is
              a number or list contains a single element
            - num_classes (:obj:`int`): number of classes, including the background class
            - cfg (:obj:`dict`): config for training or test
            - backbone (:obj:`str`): model type of backbone e.g. ``resnet18``, ``resnet34``
            - deformable (:obj:`bool`): whether to use deformable block
            - initializer (:obj:`dict`): config for module parameters initialization,
              This is only for conv layer, we always use norm initilization for the last fc layer.
            - normalize (:obj:`dict`): config of Normalization Layer

        `Res5 example <http://gitlab.bj.sensetime.com/project-spring/pytorch-object-detection/blob/
        master/configs/baselines/faster-rcnn-R50-C4-1x.yaml#L107-137>`_


        """
        super(Res5, self).__init__(inplanes, num_classes, cfg)

        from up.models.backbones.resnet import BasicBlock
        from up.models.backbones.resnet import Bottleneck
        from up.models.backbones.resnet import DeformBasicBlock
        from up.models.backbones.resnet import DeformBlock
        from up.models.backbones.resnet import make_layer4

        if backbone in ['resnet18', 'resnet34']:
            if deformable:
                block = DeformBasicBlock
            else:
                block = BasicBlock
        elif backbone in ['resnet50', 'resnet101', 'resnet152']:
            if deformable:
                block = DeformBlock
            else:
                block = Bottleneck
        else:
            raise NotImplementedError(f'{backbone} is not supported for Res5 head')

        layer = {
            'resnet18': [2, 2, 2, 2],
            'resnet34': [3, 4, 6, 3],
            'resnet50': [3, 4, 6, 3],
            'resnet101': [3, 4, 23, 3],
            'resnet152': [3, 8, 36, 3]
        }[backbone]

        stride = cfg['roipooling']['pool_size'] // 7
        self.layer4 = make_layer4(self.inplanes, block, 512, layer[3], stride=stride, normalize=normalize)
        self.avgpool = nn.AvgPool2d(7, stride=1)

        # self.fc_cls = nn.Linear(512 * block.expansion, num_classes)
        self.fc_cls = nn.Linear(512 * block.expansion, self.cls_out_channel)
        if self.cfg.get('share_location', False):
            self.fc_loc = nn.Linear(512 * block.expansion, 4)
        else:
            # self.fc_loc = nn.Linear(512 * block.expansion, num_classes * 4)
            self.fc_loc = nn.Linear(512 * block.expansion, self.cls_out_channel * 4)

        initialize_from_cfg(self, initializer)
        init_weights_normal(self.fc_cls, 0.01)
        init_weights_normal(self.fc_loc, 0.001)

    def forward_net(self, pooled_feats):
        x = self.layer4(pooled_feats)
        x = self.avgpool(x)
        c = x.numel() // x.shape[0]
        x = x.view(-1, c)
        cls_pred = self.fc_cls(x)
        loc_pred = self.fc_loc(x)
        return cls_pred, loc_pred


@MODULE_PROCESS_REGISTRY.register('BboxRFCN_Pre')
class RFCN_PreProcess(Bbox_PreProcess):
    def __init__(self, cfg, with_drp=False):
        super(RFCN_PreProcess, self).__init__(cfg, with_drp)

    def roi_extractor(self, mlvl_rois, mlvl_features, mlvl_strides):
        if not QUANT_FLAG.flag and not self.tocaffe:
            if isinstance(mlvl_strides, list):
                mlvl_strides = [int(s) for s in mlvl_strides]
            else:
                mlvl_strides = mlvl_strides.tolist()
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
        self.rfcn_score = nn.Conv2d(feat_planes, ps * ps * self.cls_out_channel, kernel_size=1)
        if self.cfg.get('share_location', False):
            self.rfcn_bbox = nn.Conv2d(feat_planes, ps * ps * 4, kernel_size=1)
        else:
            self.rfcn_bbox = nn.Conv2d(feat_planes, ps * ps * 4 * self.cls_out_channel, kernel_size=1)
        self.pool = nn.AvgPool2d((ps, ps), stride=(ps, ps))

        initialize_from_cfg(self, initializer)

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
