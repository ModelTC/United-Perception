# Standard Library
import copy

# Import from third library
import torch
import torch.nn as nn

# Import from eod
from eod.utils.model import accuracy as A
from eod.tasks.det.models.utils.anchor_generator import build_anchor_generator
from eod.models.losses import build_loss
from eod.tasks.det.models.losses.entropy_loss import apply_class_activation
from eod.utils.general.registry_factory import MODULE_ZOO_REGISTRY
from eod.utils.general.context import config
from eod.utils.general.hook_helper import allreduce
from eod.utils.env.dist_helper import env
from .fcos_predictor import build_fcos_predictor
from .fcos_supervisor import build_fcos_supervisor

__all__ = ['FcosPostProcess']


def get_ave_normalizer(loc_mask):
    ave_loc_mask = torch.sum(loc_mask)
    allreduce(ave_loc_mask)
    num_gpus = env.world_size
    ave_normalizer = max(ave_loc_mask.item(), 1) / float(num_gpus)
    return ave_normalizer


def get_fcos_loc_loss(loss_func, loc_pred, loc_target, loss_denorm, weight=None):
    if loss_func.name == 'iou_loss':
        locations = loc_pred.new_zeros(loc_pred.shape)
        pred_bbox = fcos_offset2bbox(locations, loc_pred)
        target_bbox = fcos_offset2bbox(locations, loc_target)
    elif loss_func.name == 'l1_loss':
        pred_bbox = loc_pred
        target_bbox = loc_target
    else:
        raise NotImplementedError

    loss = loss_func(pred_bbox, target_bbox, reduction_override='none')

    if weight is not None:
        weight = weight.float()

    if weight is not None and weight.sum() > 0:
        if loss_func.name == 'l1_loss':
            weight = weight.view(-1, 1).expand_as(loss)
        return (loss * weight).sum() / loss_denorm
    else:
        assert loss.numel() != 0
        return loss.mean()


def fcos_offset2bbox(locations, offset):
    x1 = locations[:, 0] - offset[:, 0]
    y1 = locations[:, 1] - offset[:, 1]
    x2 = locations[:, 0] + offset[:, 2]
    y2 = locations[:, 1] + offset[:, 3]
    return torch.stack([x1, y1, x2, y2], -1)


def compute_centerness_targets(reg_targets):
    left_right = reg_targets[:, [0, 2]]
    top_bottom = reg_targets[:, [1, 3]]
    centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
        top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
    return torch.sqrt(centerness)


@MODULE_ZOO_REGISTRY.register('fcos_post')
class FcosPostProcess(nn.Module):
    def __init__(self, num_classes, dense_points, loc_ranges, cfg, has_mask):
        super(FcosPostProcess, self).__init__()
        self.num_classes = num_classes
        self.dense_points = dense_points
        assert self.num_classes > 1
        self.loc_ranges = loc_ranges
        with config(cfg, 'train') as train_cfg:
            self.supervisor = build_fcos_supervisor(train_cfg['fcos_supervisor'])
        with config(cfg, 'test') as test_cfg:
            self.predictor = build_fcos_predictor(test_cfg['fcos_predictor'])
        self.center_generator = build_anchor_generator(cfg['center_generator'])
        self.cls_loss = build_loss(cfg['cls_loss'])
        self.loc_loss = build_loss(cfg['loc_loss'])
        self.center_loss = build_loss(cfg['center_loss'])
        self.prefix = 'FcosNet'
        self.tocaffe = False
        self.cfg = copy.deepcopy(cfg)
        self.has_mask = has_mask
        self.cls_loss_type = self.cls_loss.activation_type

    @property
    def class_activation(self):
        return self.cls_loss.activation_type

    @property
    def with_background_channel(self):
        return self.class_activation == 'softmax'

    def apply_activation_and_centerness(self, mlvl_preds, remove_background_channel_if_any):
        """apply activation on permuted class prediction
        Arguments:
            - mlvl_pred (list (levels) of tuple (predictions) of Tensor): first element must
            be permuted class prediction with last dimension be class dimension
        """
        mlvl_activated_preds = []
        for lvl_idx, preds in enumerate(mlvl_preds):
            cls_pred = apply_class_activation(preds[0], self.class_activation)
            ctr_pred = preds[2].sigmoid()
            if self.with_background_channel and remove_background_channel_if_any:
                cls_pred = cls_pred[..., 1:]
            cls_pred *= ctr_pred
            mlvl_activated_preds.append((cls_pred, *preds[1:]))
        return mlvl_activated_preds

    def permute_preds(self, mlvl_preds):
        """Permute preds from [B, A*C, H, W] to [B, H*W*A, C] """
        mlvl_permuted_preds, mlvl_shapes, box_feature = [], [], []
        for lvl_idx, preds in enumerate(mlvl_preds):
            if self.has_mask:
                box_feature.append(preds[3])
                preds = preds[:3]
            b, _, h, w = preds[0].shape
            k = self.dense_points * h * w
            preds = [p.permute(0, 2, 3, 1).contiguous().view(b, k, -1) for p in preds]
            mlvl_permuted_preds.append(preds)
            mlvl_shapes.append((h, w, k))
        return mlvl_permuted_preds, mlvl_shapes, box_feature

    def forward(self, input):
        features = input['features']
        strides = input['strides']
        image_info = input['image_info']
        mlvl_preds = input['mlvl_preds']
        # [B, hi*wi*A, :]
        mlvl_preds, mlvl_shapes, box_feature = self.permute_preds(mlvl_preds)
        mlvl_shapes = [(*shp, s) for shp, s in zip(mlvl_shapes, strides)]
        # [hi*wi*A, 4], for C4 there is only one layer, for FPN generate anchors for each layer
        mlvl_locations = self.center_generator.get_anchors(mlvl_shapes, device=features[0].device)
        output = {}

        if self.training:
            targets = self.supervisor.get_targets(mlvl_locations, self.loc_ranges, input, self.has_mask)
            if self.has_mask:
                targets, gt_inds, gt_bitmasks, gt_bitmasks_full = targets[:4], targets[4], targets[5], targets[6]
                output.update({'box_feature': box_feature,
                               'targets': targets,
                               'mlvl_preds': mlvl_preds,
                               'mlvl_locations': mlvl_locations,
                               'gt_inds': gt_inds,
                               'gt_bitmasks': gt_bitmasks,
                               'gt_bitmasks_full': gt_bitmasks_full})
            losses = self.get_loss(targets, mlvl_preds)
            output.update(losses)
        else:
            with torch.no_grad():
                mlvl_preds = self.apply_activation_and_centerness(mlvl_preds, remove_background_channel_if_any=True)
                results = self.predictor.predict(mlvl_locations, mlvl_preds, image_info, strides, self.has_mask)
                output.update(results)
                if self.has_mask:
                    output.update({'box_feature': box_feature,
                                   'mlvl_locations': mlvl_locations,
                                   'mlvl_preds': mlvl_preds})
        return output

    def get_loss(self, targets, mlvl_preds):
        mlvl_cls_pred, mlvl_loc_pred, mlvl_ctr_pred = zip(*mlvl_preds)
        cls_pred = torch.cat(mlvl_cls_pred, dim=1)
        loc_pred = torch.cat(mlvl_loc_pred, dim=1)
        centerness_pred = torch.cat(mlvl_ctr_pred, dim=1)
        del mlvl_cls_pred, mlvl_loc_pred, mlvl_ctr_pred

        cls_target, loc_target, cls_mask, loc_mask = targets

        normalizer = max(get_ave_normalizer(loc_mask), 1.0)
        cls_loss = self.cls_loss(cls_pred, cls_target, normalizer_override=normalizer)
        acc = A.accuracy_v2(cls_pred, cls_target, activation_type=self.cls_loss_type)

        sample_loc_mask = loc_mask.reshape(-1)

        loc_target = loc_target.reshape(-1, 4)[sample_loc_mask]
        loc_pred = loc_pred.reshape(-1, 4)[sample_loc_mask]
        centerness_pred = centerness_pred.reshape(-1)[sample_loc_mask]

        if loc_pred.numel() > 0:
            centerness_targets = compute_centerness_targets(loc_target)
            centerness_loss = self.center_loss(centerness_pred, centerness_targets)
        else:
            centerness_targets = loc_target.sum()
            centerness_loss = centerness_pred.sum()
        loss_denorm = max(get_ave_normalizer(centerness_targets.float()), 1e-6)
        if loc_pred.numel() > 0:
            loc_loss = get_fcos_loc_loss(self.loc_loss, loc_pred, loc_target, loss_denorm, centerness_targets)
        else:
            loc_loss = loc_pred.sum()

        return {
            self.prefix + '.cls_loss': cls_loss,
            self.prefix + '.loc_loss': loc_loss,
            self.prefix + '.centerness_loss': centerness_loss,
            self.prefix + '.accuracy': acc
        }
