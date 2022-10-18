import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from up.utils.env.dist_helper import allreduce, env
from up.utils.general.registry_factory import MODULE_PROCESS_REGISTRY
from up.utils.model import accuracy as A  # noqa F401
from up.tasks.det.models.utils.anchor_generator import build_anchor_generator
from up.models.losses import build_loss
from up.tasks.det.models.losses.entropy_loss import apply_class_activation
from up.tasks.det.models.utils.bbox_helper import offset2bbox, bbox_iou_overlaps
from .onenet_supervisor import build_onenet_supervisor
from .onenet_predictor import build_onenet_predictor

__all__ = ['OnenetPostProcess']


@MODULE_PROCESS_REGISTRY.register('onenet_post')
class OnenetPostProcess(nn.Module):
    def __init__(self,
                 num_classes,
                 net_type,
                 cfg,
                 use_qfl=False,
                 l1_weight=5.0,
                 num_anchors=None,
                 prefix=None,
                 norm_on_bbox=False,
                 all_reduce_norm=True):
        super(OnenetPostProcess, self).__init__()
        assert net_type == "fcos" or net_type == "retina", "only support retina or fcos currently!"
        self.prefix = prefix if prefix is not None else self.__class__.__name__
        self.net_type = net_type
        self.num_classes = num_classes
        self.use_qfl = use_qfl
        self.anchor_generator = build_anchor_generator(cfg['anchor_generator'])
        self.num_anchors = self.anchor_generator.num_anchors if num_anchors is None else num_anchors
        test_cfg = copy.deepcopy(cfg)
        test_cfg.update(test_cfg.get('test', {}))
        train_cfg = copy.deepcopy(cfg)
        train_cfg.update(train_cfg.get('train', {}))
        self.supervisor = build_onenet_supervisor(train_cfg['roi_supervisor'])
        self.predictor = build_onenet_predictor(test_cfg['roi_predictor'])
        self.cls_loss = build_loss(cfg['cls_loss'])
        self.loc_loss = build_loss(cfg['loc_loss'])
        self.norm_on_bbox = norm_on_bbox
        self.l1_weight = l1_weight

        self.cfg = copy.deepcopy(cfg)
        self.all_reduce_norm = all_reduce_norm

    @property
    def class_activation(self):
        return self.cls_loss.activation_type

    @property
    def with_background_channel(self):
        return self.class_activation == 'softmax'

    def apply_activation(self, preds, remove_background_channel_if_any):
        cls_pred = apply_class_activation(preds[0], self.class_activation)
        if self.with_background_channel and remove_background_channel_if_any:
            cls_pred = cls_pred[..., 1:]
        preds[0] = cls_pred
        return preds

    def get_ave_normalizer(self, _mask):
        ave_mask = torch.sum(_mask)
        if env.world_size > 1:
            allreduce(ave_mask)
        num_gpus = env.world_size
        ave_normalizer = max(ave_mask.item(), 1) / float(num_gpus)
        return ave_normalizer

    def permute_preds(self, mlvl_preds):
        """Permute preds from [B, A*C, H, W] to [B, H*W*A, C] """
        mlvl_permuted_preds, mlvl_shapes = [], []
        for lvl_idx, preds in enumerate(mlvl_preds):
            b, _, h, w = preds[0].shape
            k = self.num_anchors * h * w
            preds = [p.permute(0, 2, 3, 1).contiguous().view(b, k, -1) for p in preds]
            mlvl_permuted_preds.append(preds)
            mlvl_shapes.append((h, w, k))
        return mlvl_permuted_preds, mlvl_shapes

    def forward(self, input):
        features = input['features']
        strides = input['strides']
        mlvl_preds = input['mlvl_preds'] if 'mlvl_preds' in input else input['preds']

        mlvl_preds, mlvl_shapes = self.permute_preds(mlvl_preds)
        mlvl_shapes = [(*shp, s) for shp, s in zip(mlvl_shapes, strides)]

        mlvl_predef = self.anchor_generator.get_anchors(mlvl_shapes, device=features[0].device)
        preds = []
        mlvl_cls_pred, mlvl_loc_pred = zip(*(mlvl_preds))
        preds.append(torch.cat(mlvl_cls_pred, dim=1))
        preds.append(torch.cat(mlvl_loc_pred, dim=1))
        del mlvl_preds, mlvl_cls_pred, mlvl_loc_pred

        preds[1] = self.decode_loc(preds, mlvl_predef)

        output = {}

        if self.training:
            targets = self.supervisor.get_targets(preds, input)
            losses = self.get_loss(targets, input)
            output.update(losses)
        else:
            preds = self.apply_activation(preds, remove_background_channel_if_any=True)
            results = self.predictor.predict(preds, input)
            output.update(results)
        return output

    def decode_loc(self, preds, mlvl_predef):
        cls_pred, loc_pred = preds[:2]
        B, K, C = cls_pred.shape
        loc_pred = loc_pred.view(B * K, 4)
        if self.net_type == "fcos":
            if self.norm_on_bbox:
                raise NotImplementedError("TODO!")
            locations = torch.stack([mlvl_predef.clone() for _ in range(B)]).reshape(B * K, 2)
            x1 = locations[:, 0] - loc_pred[:, 0]
            y1 = locations[:, 1] - loc_pred[:, 1]
            x2 = locations[:, 0] - loc_pred[:, 2]
            y2 = locations[:, 1] - loc_pred[:, 3]
            decode_bbox = torch.stack([x1, y1, x2, y2], -1).view(B, K, 4)
        else:
            all_anchors = torch.cat(mlvl_predef, dim=0)
            batch_anchors = all_anchors.view(1, -1, 4).expand((B, K, 4)).reshape(B * K, 4)
            decode_bbox = offset2bbox(batch_anchors, loc_pred).view(B, K, 4)
        return decode_bbox

    def get_acc(self, cls_pred, cls_target):
        acc = A.accuracy_v2(cls_pred, cls_target, self.class_activation)
        return acc

    def get_loss(self, targets, input):
        cls_targets, reg_targets, cls_preds, pos_loc_preds = targets[:4]
        img_whwh, num_boxes, pos_cls_targets, pos_cls_preds, pos_idx = targets[4:]

        if self.all_reduce_norm and dist.is_initialized():
            num_fgs = self.get_ave_normalizer(num_boxes)
        else:
            num_fgs = num_boxes.item()
        num_fgs = max(num_fgs, 1)

        if self.use_qfl:
            B, K, _ = cls_preds.size()
            scores = cls_preds.new_zeros((B, K))
            if num_fgs > 0:
                scores[pos_idx] = bbox_iou_overlaps(pos_loc_preds.detach(), reg_targets, aligned=True)
            cls_loss = self.cls_loss(cls_preds, cls_targets, normalizer_override=num_fgs, scores=scores)
        else:
            cls_loss = self.cls_loss(cls_preds, cls_targets, normalizer_override=num_fgs)

        if num_fgs > 0:
            loc_loss = self.loc_loss(pos_loc_preds, reg_targets, normalizer_override=num_fgs)
        else:
            loc_loss = pos_loc_preds.sum()

        if self.l1_weight > 0:
            if num_fgs > 0:
                l1_loss = F.l1_loss(pos_loc_preds, reg_targets, reduction='none') / img_whwh
                l1_loss = l1_loss.sum() / num_fgs * self.l1_weight
            else:
                l1_loss = pos_loc_preds.sum()
        else:
            l1_loss = torch.tensor(0.0).cuda()

        acc = self.get_acc(pos_cls_preds, pos_cls_targets)

        return {
            self.prefix + '.cls_loss': cls_loss,
            self.prefix + '.loc_loss': loc_loss,
            self.prefix + '.l1_loss': l1_loss,
            self.prefix + '.accuracy': acc
        }
