import torch
import numpy as np
import torch.nn as nn
from up.utils.general.registry_factory import MODULE_ZOO_REGISTRY
from up.tasks.det_3d.data.data_utils import limit_period
from up.tasks.det_3d.data import box_coder_utils
from up.tasks.det.models.utils.anchor_generator import build_anchor_generator
from up.tasks.det.models.postprocess.roi_supervisor import build_roi_supervisor
from up.tasks.det.models.postprocess.roi_predictor import build_roi_predictor
from up.tasks.det_3d.data.box_coder_utils import build_box_coder
from up.models.losses import build_loss

__all__ = ['AnchorHeadSingle']


@MODULE_ZOO_REGISTRY.register('anchor_head_post')
class AnchorHeadSingle(nn.Module):
    def __init__(
            self,
            dir_offset,
            dir_limit_offset,
            num_dir_bins,
            cfg,
            predict_boxes_when_training=False,
            use_multihead=False,
            prefix=None
    ):
        super(AnchorHeadSingle, self).__init__()
        self.predict_boxes_when_training = predict_boxes_when_training
        self.prefix = prefix if prefix is not None else self.__class__.__name__

        self.num_dir_bins = num_dir_bins
        self.dir_offset = dir_offset
        self.dir_limit_offset = dir_limit_offset
        self.use_multihead = use_multihead

        self.box_coder = build_box_coder(cfg['box_coder'])
        self.anchor_generator = build_anchor_generator(cfg['anchor_generator'])
        self.supervisor = build_roi_supervisor(cfg['roi_supervisor'])
        self.predictor = build_roi_predictor(cfg['roi_predictor'])

        self.cls_loss = build_loss(cfg['cls_loss'])
        self.loc_loss = build_loss(cfg['loc_loss'])
        self.dir_loss = build_loss(cfg['dir_loss'])

    def forward(self, input):
        cls_preds, box_preds, dir_pred = input['cls_preds'], input['box_preds'], input['dir_pred']
        self.class_names, gt_boxes, grid_size, batch_size, self.num_anchors = input['class_names'], \
            input['gt_boxes'], input['voxel_infos']['grid_size'], input['batch_size'], input['num_anchors']

        anchors, num_anchors_per_location = self.anchor_generator.get_anchors(grid_size)
        self.anchors = [x.cuda() for x in anchors]
        output = {}
        if self.training:
            targets = self.supervisor.get_targets(self.class_names, self.anchors, gt_boxes, self.box_coder)
            losses = self.get_loss(targets, cls_preds, box_preds, dir_pred)
            output.update(losses)
            if self.predict_boxes_when_training:
                batch_preds = self.generate_predicted_boxes(batch_size, cls_preds, box_preds, dir_pred)
        else:
            batch_preds = self.generate_predicted_boxes(batch_size, cls_preds, box_preds, dir_pred)
            pred_list = self.predictor.predict(input, batch_preds)
            output.update({'pred_list': pred_list})
        return output

    def generate_predicted_boxes(self, batch_size, cls_preds, box_preds, dir_cls_preds=None):
        """
        Args:
            batch_size:
            cls_preds: (N, H, W, C1)
            box_preds: (N, H, W, C2)
            dir_cls_preds: (N, H, W, C3)

        Returns:
            batch_cls_preds: (B, num_boxes, num_classes)
            batch_box_preds: (B, num_boxes, 7+C)

        """
        if isinstance(self.anchors, list):
            if self.use_multihead:
                anchors = torch.cat([anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1])
                                     for anchor in self.anchors], dim=0)
            else:
                anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors
        num_anchors = anchors.view(-1, anchors.shape[-1]).shape[0]
        batch_anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
        batch_cls_preds = cls_preds.view(batch_size, num_anchors, -1).float() \
            if not isinstance(cls_preds, list) else cls_preds
        batch_box_preds = box_preds.view(batch_size, num_anchors, -1) if not isinstance(box_preds, list) \
            else torch.cat(box_preds, dim=1).view(batch_size, num_anchors, -1)
        batch_box_preds = self.box_coder.decode_torch(batch_box_preds, batch_anchors)

        if dir_cls_preds is not None:
            dir_offset = self.dir_offset
            dir_limit_offset = self.dir_limit_offset
            dir_cls_preds = dir_cls_preds.view(batch_size, num_anchors, -1) if not isinstance(dir_cls_preds, list) \
                else torch.cat(dir_cls_preds, dim=1).view(batch_size, num_anchors, -1)
            dir_labels = torch.max(dir_cls_preds, dim=-1)[1]

            period = (2 * np.pi / self.num_dir_bins)
            dir_rot = limit_period(
                batch_box_preds[..., 6] - dir_offset, dir_limit_offset, period
            )
            batch_box_preds[..., 6] = dir_rot + dir_offset + period * dir_labels.to(batch_box_preds.dtype)

        if isinstance(self.box_coder, box_coder_utils.PreviousResidualDecoder):
            batch_box_preds[..., 6] = limit_period(
                -(batch_box_preds[..., 6] + np.pi / 2), offset=0.5, period=np.pi * 2
            )
        return {'batch_cls_preds': batch_cls_preds, 'batch_box_preds': batch_box_preds, 'cls_preds_normalized': False}

    def get_loss(self, targets, cls_preds, box_preds, dir_cls_preds):
        self.num_classes = len(self.class_names)
        box_cls_labels = targets['box_cls_labels']
        box_reg_targets = targets['box_reg_targets']
        output = {}
        cls_loss = self.get_cls_layer_loss(cls_preds, box_cls_labels)
        loc_loss, dir_loss = self.get_box_reg_layer_loss(box_cls_labels, box_reg_targets, box_preds, dir_cls_preds)
        output.update({self.prefix + '.cls_loss': cls_loss, self.prefix
                      + '.loc_loss': loc_loss, self.prefix + '.dir_loss': dir_loss})
        return output

    def get_cls_layer_loss(self, cls_preds, box_cls_labels):
        batch_size = int(cls_preds.shape[0])
        cared = box_cls_labels >= 0  # [N, num_anchors]
        positives = box_cls_labels > 0
        negatives = box_cls_labels == 0
        negative_cls_weights = negatives * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        reg_weights = positives.float()
        if self.num_classes == 1:
            # class agnostic
            box_cls_labels[positives] = 1

        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_targets = box_cls_labels * cared.type_as(box_cls_labels)
        cls_preds = cls_preds.view(batch_size, -1, self.num_classes)
        normalizer_override = cls_targets.shape[0]
        cls_loss = self.cls_loss(
            cls_preds,
            cls_targets,
            normalizer_override=normalizer_override,
            weights=cls_weights)
        return cls_loss

    def get_box_reg_layer_loss(self, box_cls_labels, box_reg_targets, box_preds, box_dir_cls_preds):
        batch_size = int(box_preds.shape[0])
        positives = box_cls_labels > 0
        reg_weights = positives.float()
        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        if isinstance(self.anchors, list):
            if self.use_multihead:
                anchors = torch.cat(
                    [anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1]) for anchor in
                     self.anchors], dim=0)
            else:
                anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors
        anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
        box_preds = box_preds.view(batch_size, -1,
                                   box_preds.shape[-1] // self.num_anchors if not self.use_multihead else
                                   box_preds.shape[-1])
        # sin(a - b) = sinacosb-cosasinb
        box_preds_sin, reg_targets_sin = self.add_sin_difference(box_preds, box_reg_targets)
        normalizer_override = reg_targets_sin.shape[0]
        loc_loss = self.loc_loss(
            box_preds_sin,
            reg_targets_sin,
            normalizer_override=normalizer_override,
            weights=reg_weights)  # [N, M]
        if box_dir_cls_preds is not None:
            dir_targets = self.get_direction_target(
                anchors, box_reg_targets,
                dir_offset=self.dir_offset,
                num_bins=self.num_dir_bins
            )
            dir_logits = box_dir_cls_preds.view(batch_size, -1, self.num_dir_bins)
            weights = positives.type_as(dir_logits)
            weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
            dir_targets = dir_targets.long().argmax(dim=-1)
            dir_logits = dir_logits.permute(0, 2, 1)
            normalizer_override = dir_targets.shape[0]
            dir_loss = self.dir_loss(dir_logits, dir_targets, normalizer_override=normalizer_override, weights=weights)
        return loc_loss, dir_loss

    @staticmethod
    def add_sin_difference(boxes1, boxes2, dim=6):
        assert dim != -1
        rad_pred_encoding = torch.sin(boxes1[..., dim:dim + 1]) * torch.cos(boxes2[..., dim:dim + 1])
        rad_tg_encoding = torch.cos(boxes1[..., dim:dim + 1]) * torch.sin(boxes2[..., dim:dim + 1])
        boxes1 = torch.cat([boxes1[..., :dim], rad_pred_encoding, boxes1[..., dim + 1:]], dim=-1)
        boxes2 = torch.cat([boxes2[..., :dim], rad_tg_encoding, boxes2[..., dim + 1:]], dim=-1)
        return boxes1, boxes2

    @staticmethod
    def get_direction_target(anchors, reg_targets, one_hot=True, dir_offset=0, num_bins=2):
        batch_size = reg_targets.shape[0]
        anchors = anchors.view(batch_size, -1, anchors.shape[-1])
        rot_gt = reg_targets[..., 6] + anchors[..., 6]
        offset_rot = limit_period(rot_gt - dir_offset, 0, 2 * np.pi)
        dir_cls_targets = torch.floor(offset_rot / (2 * np.pi / num_bins)).long()
        dir_cls_targets = torch.clamp(dir_cls_targets, min=0, max=num_bins - 1)
        if one_hot:
            dir_targets = torch.zeros(*list(dir_cls_targets.shape), num_bins, dtype=anchors.dtype,
                                      device=dir_cls_targets.device)
            dir_targets.scatter_(-1, dir_cls_targets.unsqueeze(dim=-1).long(), 1.0)
            dir_cls_targets = dir_targets
        return dir_cls_targets
