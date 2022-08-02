import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from up import extensions as E
from up.utils.model import accuracy as A
from up.utils.model.initializer import init_weights_normal, initialize_from_cfg
from up.utils.general.registry_factory import MODULE_ZOO_REGISTRY
from up.models.losses import build_loss
from up.tasks.det.models.postprocess.bbox_supervisor import build_bbox_supervisor
from up.tasks.det.models.postprocess.bbox_predictor import build_bbox_predictor
from up.tasks.det.models.utils.assigner import map_rois_to_level
from up.tasks.det.models.losses.ohem import ohem_loss
from up.utils.model.normalize import build_conv_norm

__all__ = ['CascadeBboxNet', 'CascadeFC', 'ConvFC']


class DropBlock(nn.Module):
    """Randomly drop some regions of feature maps.
     Please refer to the method proposed in `DropBlock
     <https://arxiv.org/abs/1810.12890>`_ for details.
    Args:
        drop_prob (float): The probability of dropping each block.
        block_size (int): The size of dropped blocks.
        warmup_iters (int): The drop probability will linearly increase
            from `0` to `drop_prob` during the first `warmup_iters` iterations.
            Default: 2000.
    """

    def __init__(self, drop_prob, block_size, warmup_iters=2000, **kwargs):
        super(DropBlock, self).__init__()
        assert block_size % 2 == 1
        assert 0 < drop_prob <= 1
        assert warmup_iters >= 0
        self.drop_prob = drop_prob
        self.block_size = block_size
        self.warmup_iters = warmup_iters
        self.iter_cnt = 0

    def forward(self, x):
        """
        Args:
            x (Tensor): Input feature map on which some areas will be randomly
                dropped.
        Returns:
            Tensor: The tensor after DropBlock layer.
        """
        if not self.training:
            return x
        self.iter_cnt += 1
        N, C, H, W = list(x.shape)
        gamma = self._compute_gamma((H, W))
        mask_shape = (N, C, H - self.block_size + 1, W - self.block_size + 1)
        mask = torch.bernoulli(torch.full(mask_shape, gamma, device=x.device))

        mask = F.pad(mask, [self.block_size // 2] * 4, value=0)
        mask = F.max_pool2d(
            input=mask,
            stride=(1, 1),
            kernel_size=(self.block_size, self.block_size),
            padding=self.block_size // 2)
        mask = 1 - mask
        mask_sum = mask.sum()
        x = x * mask * mask.numel() / (1e-6 + mask_sum)
        return x.half()

    def _compute_gamma(self, feat_size):
        """Compute the value of gamma according to paper. gamma is the
        parameter of bernoulli distribution, which controls the number of
        features to drop.
        gamma = (drop_prob * fm_area) / (drop_area * keep_area)
        Args:
            feat_size (tuple[int, int]): The height and width of feature map.
        Returns:
            float: The value of gamma.
        """
        gamma = (self.drop_prob * feat_size[0] * feat_size[1])
        gamma /= ((feat_size[0] - self.block_size + 1)
                  * (feat_size[1] - self.block_size + 1))
        gamma /= (self.block_size**2)
        factor = (1.0 if self.iter_cnt > self.warmup_iters else self.iter_cnt
                  / self.warmup_iters)
        return gamma * factor


class conv3x3(nn.Module):
    def __init__(self, inplanes, outplanes, normalize, drop_block=False, coord_conv=False, stride=1):
        super().__init__()
        self.drop_block = drop_block
        self.coord_conv = coord_conv
        if coord_conv:
            inplanes += 2
        self.conv = build_conv_norm(inplanes,
                                    outplanes,
                                    kernel_size=3,
                                    stride=stride,
                                    padding=1,
                                    normalize=normalize,
                                    activation=True)
        if drop_block:
            self.drop = DropBlock(drop_prob=0.9, block_size=1)

    def forward(self, x):
        if self.coord_conv:
            x_range = torch.linspace(-1, 1, x.shape[-1], device=x.device)
            y_range = torch.linspace(-1, 1, x.shape[-2], device=x.device)
            y_range, x_range = torch.meshgrid(y_range, x_range)
            y_range = y_range.expand([x.shape[0], 1, -1, -1])
            x_range = x_range.expand([x.shape[0], 1, -1, -1])
            coord_feat = torch.cat([x_range, y_range], 1)
            if 'Half' in x.type():
                coord_feat = coord_feat.half()
            x = torch.cat([x, coord_feat], 1)
        x = self.conv(x)
        if self.drop_block:
            x = self.drop(x)
        return x


class CascadeBboxNet(nn.Module):
    """
    Cascade boxes prediction and refinement.
    """

    def __init__(self, inplanes, num_classes, cfg):
        """
        Arguments:
            - inplanes (:obj:`list` or :obj:`int`): input channel,
              which is a number or list contains a single element
            - num_classes (:obj:`int`): number of classes, including the background class
            - cfg (:obj:`dict`): config for training or test
        """
        super(CascadeBboxNet, self).__init__()

        self.cfg = copy.deepcopy(cfg)  # runtime configuration
        self.cfg['num_classes'] = num_classes
        self.num_stage = self.cfg.get('num_stage', 1)
        self.stage_weights = self.cfg.get('stage_weights', None)
        self.test_ensemble = self.cfg.get('test_ensemble', True)

        self.supervisor = build_bbox_supervisor(self.cfg['cascade_supervisor'])
        self.predictor = build_bbox_predictor(self.cfg['cascade_predictor'])

        if isinstance(inplanes, list):
            assert len(inplanes) == 1, 'single input is expected, but found:{} '.format(inplanes)
            inplanes = inplanes[0]
        assert isinstance(inplanes, int)
        self.inplanes = inplanes

        roipool = E.build_generic_roipool(cfg['roipooling'])
        self.roipool_list = nn.ModuleList()
        for i in range(self.num_stage):
            self.roipool_list.append(roipool)
        self.pool_size = cfg['roipooling']['pool_size']

        self.cls_loss = build_loss(self.cfg['cls_loss'])
        self.loc_loss = build_loss(self.cfg['loc_loss'])
        self.prefix = 'CascadeBboxNet'
        self.tocaffe = False

    def forward_net(self, rois, x, stride, stage):
        """
        Arguments:
            rois (FloatTensor): rois in a sinle layer
            x (FloatTensor): features in a single layer
            stride: stride for current layer
        """
        raise NotImplementedError

    def forward(self, input):
        """
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
                # (list of FloatTensor): [B] [num_igs, 4] (x1, y1, x2, y2)
                'gt_ignores': []
            }

        Output example::

            # training output
            # 0 <= i < num_stage
            {'BboxNet.cls_loss{i}': <tensor>, 'BboxNet.loc_loss_{i}': <tensor>, 'BboxNet.accuracy_{i}': <tensor>}

            # testing output
            {
                # (FloatTensor), predicted boxes [N, >=7] (batch_ix, x1, y1, x2, y2, score, cls)
                'dt_bboxes': <tensor>,
            }
        """
        prefix = 'BboxNet'

        if isinstance(input['strides'], list):
            input['strides'] = [int(s) for s in input['strides']]
        else:
            input['strides'] = input['strides'].numpy().tolist()

        output = {}
        if self.training:
            stage_sample_record, stage_cls_loss, stage_loc_loss, stage_acc = self.get_loss(input)
            for i in range(self.num_stage):
                # output['sample_record_' + str(i)] = stage_sample_record[i]
                output['sample_record'] = stage_sample_record[i]
                output[prefix + '.cls_loss_' + str(i)] = stage_cls_loss[i]
                output[prefix + '.loc_loss_' + str(i)] = stage_loc_loss[i]
                output[prefix + '.accuracy_' + str(i)] = stage_acc[i]
        else:
            results = self.get_bboxes(input)
            output.update(results)
        return output

    def mlvl_predict(self, x_rois, x_features, x_strides, levels, stage):
        """Predict results level by level"""
        mlvl_cls_pred, mlvl_loc_pred = [], []
        for lvl_idx in levels:
            if x_rois[lvl_idx].numel() > 0:
                rois = x_rois[lvl_idx]
                feature = x_features[lvl_idx]
                stride = x_strides[lvl_idx]
                cls_pred, loc_pred = self.forward_net(rois, feature, stride, stage)
                mlvl_cls_pred.append(cls_pred)
                mlvl_loc_pred.append(loc_pred)
        cls_pred = torch.cat(mlvl_cls_pred, dim=0)
        loc_pred = torch.cat(mlvl_loc_pred, dim=0)
        return cls_pred, loc_pred

    def get_head_output(self, rois, features, strides, stage):
        """
        Assign rois to each level and predict

        Note:
            1.The recovering process is not supposed to be handled in this function,
              because ONNX DON'T support indexing;
            2.numerical type of cls_pred and loc_pred must be float for fp32 support !!!

        Returns:
            - rois (FloatTensor): assigned rois
            - cls_pred (FloatTensor, fp32): prediction of classification of assigned rois
            - loc_pred (FloatTensor, fp32): prediction of localization of assigned rois
            - recover_inds (LongTensor): indices of recovering input rois from assigned rois
        """
        if self.cfg.get('fpn', None):
            # assign rois and targets to each level
            fpn = self.cfg['fpn']
            mlvl_rois, recover_inds = map_rois_to_level(fpn['fpn_levels'], fpn['base_scale'], rois)
            cls_pred, loc_pred = self.mlvl_predict(mlvl_rois, features, strides, fpn['fpn_levels'], stage)
            rois = torch.cat(mlvl_rois, dim=0)
            # cls_pred = cls_pred[recover_inds]
            # loc_pred = loc_pred[recover_inds]
        else:
            assert len(features) == 1 and len(strides) == 1, \
                'please setup `fpn` first if you want to use pyramid features'
            cls_pred, loc_pred = self.forward_net(rois, features[0], strides[0], stage)
            recover_inds = torch.arange(rois.shape[0], device=rois.device)
        return rois, cls_pred.float(), loc_pred.float(), recover_inds

    def get_loss(self, input):
        """
        Arguments:
            input['features'] (list): input feature layers, for C4 from backbone, others from FPN
            input['strides'] (list): strides of input feature layers
            input['image_info'] (list of FloatTensor): [B, 5] (reiszed_h, resized_w, scale_factor, origin_h, origin_w)
            input['dt_bboxes'] (FloatTensor): [N, >=7] (batch_ix, x1, y1, x2, y2, score, cls)
            input['gt_bboxes'] (list of FloatTensor): [B, num_gts, 5] (x1, y1, x2, y2, label)
            input['gt_ignores'] (list of FloatTensor): [B, num_igs, 4] (x1, y1, x2, y2)

        Returns:
            sample_record (list of tuple): [B, (pos_inds, pos_target_gt_inds)], saved for mask/keypoint head
            cls_loss, loc_loss, acc (FloatTensor)
        """
        features = input['features']
        strides = input['strides']
        image_info = input['image_info']
        B = len(image_info)
        rois = input['dt_bboxes']

        stage_sample_record = []
        stage_cls_loss = []
        stage_loc_loss = []
        stage_acc = []
        for cascade_i in range(self.num_stage):
            stage_weight = self.stage_weights[cascade_i]
            # cascade_i_cfg = self.get_cascade_stage_cfg(cascade_i)
            # cls_target (LongTensor): [R]
            # loc_target (FloatTensor): [R, 4]
            # loc_weight (FloatTensor): [R, 4]
            (sample_record, sampled_rois, cls_target, loc_target, loc_weight,
                gt_flags) = self.supervisor.get_targets(cascade_i, rois, input)
            rois, cls_pred, loc_pred, recover_inds = self.get_head_output(sampled_rois, features, strides, cascade_i)
            cls_pred = cls_pred[recover_inds]
            loc_pred = loc_pred[recover_inds]

            cls_inds = cls_target
            # if cascade_i_cfg.get('share_location', 'False'):
            if self.cfg.get('share_location', 'False'):
                cls_inds = cls_target.clamp(max=0)

            N = loc_pred.shape[0]
            loc_pred = loc_pred.reshape(N, -1, 4)
            inds = torch.arange(N, dtype=torch.int64, device=loc_pred.device)
            if self.cls_loss.activation_type == 'sigmoid' and not self.cfg.get('share_location', 'False'):
                cls_inds -= 1
            loc_pred = loc_pred[inds, cls_inds].reshape(-1, 4)

            reduction_override = None if 'ohem' not in self.cfg else 'none'
            normalizer_override = cls_target.shape[0] if 'ohem' not in self.cfg else None

            cls_loss = self.cls_loss(cls_pred, cls_target, reduction_override=reduction_override)
            loc_loss = self.loc_loss(loc_pred * loc_weight, loc_target,
                                     reduction_override=reduction_override,
                                     normalizer_override=normalizer_override)

            if self.cfg.get('ohem', None):
                cls_loss, loc_loss = ohem_loss(cls_loss, loc_loss, B, self.cfg['ohem'])

            acc = A.accuracy(cls_pred, cls_target)[0]

            # collect cascade stage loss and accuracy
            cls_loss = cls_loss * stage_weight
            loc_loss = loc_loss * stage_weight
            stage_sample_record.append(sample_record)
            stage_cls_loss.append(cls_loss)
            stage_loc_loss.append(loc_loss)
            stage_acc.append(acc)
            # refine bboxes before the last stage
            if cascade_i < self.num_stage - 1:
                rois = rois[recover_inds]
                with torch.no_grad():
                    rois = self.predictor.refine(cascade_i, rois, cls_target, loc_pred, image_info, gt_flags)
                    # rois = refine_bboxes(rois, cls_target, loc_pred, image_info, cascade_i_cfg, gt_flags)
        return stage_sample_record, stage_cls_loss, stage_loc_loss, stage_acc

    def get_bboxes(self, input):
        features = input['features']
        strides = input['strides']
        image_info = input['image_info']
        rois = input['dt_bboxes']

        stage_scores = []
        for cascade_i in range(self.num_stage):
            # cascade_i_cfg = self.get_cascade_stage_cfg(cascade_i)
            rois, cls_pred, loc_pred, recover_inds = self.get_head_output(rois, features, strides, cascade_i)
            rois = rois.detach()[recover_inds]
            cls_pred = cls_pred.detach()[recover_inds]
            loc_pred = loc_pred.detach()[recover_inds]
            # cls_pred = F.softmax(cls_pred, dim=1)
            if self.cls_loss.activation_type == 'softmax':
                cls_pred = F.softmax(cls_pred, dim=1)
            else:
                cls_pred = torch.sigmoid(cls_pred)
            stage_scores.append(cls_pred)

            if cascade_i < self.num_stage - 1:
                rois = self.predictor.refine(
                    cascade_i, rois, cls_pred.argmax(dim=1), loc_pred.detach(), image_info)

        if self.test_ensemble:
            cls_pred = sum(stage_scores) / self.num_stage

        start_idx = 0 if self.cls_loss.activation_type == 'sigmoid' else 1
        results = self.predictor.predict(rois, (cls_pred, loc_pred), image_info, start_idx=start_idx)
        return results


@MODULE_ZOO_REGISTRY.register('CascadeFC')
class CascadeFC(CascadeBboxNet):
    """
    Use FC as head
    """

    def __init__(self, inplanes, feat_planes, num_classes, cfg, initializer=None):
        """
        Arguments:
            - inplanes (:obj:`list` or :obj:`int`): input channel,
              which is a number or list contains a single element
            - feat_planes (:obj:`int`): channels of intermediate features
            - num_classes (:obj:`int`): number of classes, including the background class
            - cfg (:obj:`dict`): config for training or test
            - initializer (:obj:`dict`): config for module parameters initialization

        """
        super(CascadeFC, self).__init__(inplanes, num_classes, cfg)

        inplanes = self.inplanes
        self.relu = nn.ReLU(inplace=True)

        self.fc6_list = nn.ModuleList()
        self.fc7_list = nn.ModuleList()
        self.fc_rcnn_cls_list = nn.ModuleList()
        self.fc_rcnn_loc_list = nn.ModuleList()
        cls_out_channel = num_classes if self.cls_loss.activation_type == 'softmax' else num_classes - 1
        for i in range(self.num_stage):
            fc6 = nn.Linear(self.pool_size * self.pool_size * inplanes, feat_planes)
            self.fc6_list.append(fc6)
            fc7 = nn.Linear(feat_planes, feat_planes)
            self.fc7_list.append(fc7)

            # fc_rcnn_cls = nn.Linear(feat_planes, num_classes)
            fc_rcnn_cls = nn.Linear(feat_planes, cls_out_channel)
            self.fc_rcnn_cls_list.append(fc_rcnn_cls)
            if self.cfg.get('share_location', False):
                fc_rcnn_loc = nn.Linear(feat_planes, 4)
            else:
                # fc_rcnn_loc = nn.Linear(feat_planes, num_classes * 4)
                fc_rcnn_loc = nn.Linear(feat_planes, cls_out_channel * 4)
            self.fc_rcnn_loc_list.append(fc_rcnn_loc)

        initialize_from_cfg(self, initializer)
        init_weights_normal(self.fc_rcnn_cls_list, 0.01)
        init_weights_normal(self.fc_rcnn_loc_list, 0.001)

    def forward_net(self, rois, x, stride, stage):
        roipool = self.roipool_list[stage]
        fc6 = self.fc6_list[stage]
        fc7 = self.fc7_list[stage]
        fc_rcnn_cls = self.fc_rcnn_cls_list[stage]
        fc_rcnn_loc = self.fc_rcnn_loc_list[stage]
        x = roipool(rois, x, stride)
        c = x.numel() // x.shape[0]
        x = x.view(-1, c)
        x = self.relu(fc6(x))
        x = self.relu(fc7(x))
        cls_pred = fc_rcnn_cls(x)
        loc_pred = fc_rcnn_loc(x)
        return cls_pred, loc_pred


@MODULE_ZOO_REGISTRY.register('CascadeConvFC')
class ConvFC(CascadeBboxNet):
    """
    Use ConvFC as head
    """

    def __init__(self, inplanes, feat_planes, num_classes, cfg, initializer=None, normalize=None, num_conv=4,
                 drop_block=None, coord_conv=None):
        """
        Arguments:
            - inplanes (:obj:`list` or :obj:`int`): input channel,
              which is a number or list contains a single element
            - feat_planes (:obj:`int`): channels of intermediate features
            - num_classes (:obj:`int`): number of classes, including the background class
            - cfg (:obj:`dict`): config for training or test
            - initializer (:obj:`dict`): config for module parameters initialization

        """
        super(ConvFC, self).__init__(inplanes, num_classes, cfg)

        inplanes = self.inplanes
        self.relu = nn.ReLU(inplace=True)

        self.fc6_list = nn.ModuleList()
        self.fc7_list = nn.ModuleList()
        self.fc_rcnn_cls_list = nn.ModuleList()
        self.fc_rcnn_loc_list = nn.ModuleList()
        self.conv_list = nn.ModuleList()
        cls_out_channel = num_classes if self.cls_loss.activation_type == 'softmax' else num_classes - 1
        for i in range(self.num_stage):
            convs = nn.Sequential()
            for j in range(num_conv):
                convs.add_module(
                    f'{j}',
                    conv3x3(inplanes,
                            inplanes,
                            normalize=normalize,
                            drop_block=drop_block,
                            coord_conv=coord_conv))
            self.conv_list.append(convs)
            fc6 = nn.Linear(self.pool_size * self.pool_size * inplanes, feat_planes)
            self.fc6_list.append(fc6)
            fc7 = nn.Linear(feat_planes, feat_planes)
            self.fc7_list.append(fc7)

            # fc_rcnn_cls = nn.Linear(feat_planes, num_classes)
            fc_rcnn_cls = nn.Linear(feat_planes, cls_out_channel)
            self.fc_rcnn_cls_list.append(fc_rcnn_cls)
            if self.cfg.get('share_location', False):
                fc_rcnn_loc = nn.Linear(feat_planes, 4)
            else:
                # fc_rcnn_loc = nn.Linear(feat_planes, num_classes * 4)
                fc_rcnn_loc = nn.Linear(feat_planes, cls_out_channel * 4)
            self.fc_rcnn_loc_list.append(fc_rcnn_loc)

        initialize_from_cfg(self, initializer)
        init_weights_normal(self.fc_rcnn_cls_list, 0.01)
        init_weights_normal(self.fc_rcnn_loc_list, 0.001)

    def forward_net(self, rois, x, stride, stage):
        roipool = self.roipool_list[stage]
        fc6 = self.fc6_list[stage]
        fc7 = self.fc7_list[stage]
        fc_rcnn_cls = self.fc_rcnn_cls_list[stage]
        fc_rcnn_loc = self.fc_rcnn_loc_list[stage]
        conv = self.conv_list[stage]
        x = roipool(rois, x, stride)
        x = conv(x)
        c = x.numel() // x.shape[0]
        x = x.view(-1, c)
        x = self.relu(fc6(x))
        x = self.relu(fc7(x))
        cls_pred = fc_rcnn_cls(x)
        loc_pred = fc_rcnn_loc(x)
        return cls_pred, loc_pred
