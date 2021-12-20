import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from eod.utils.model import accuracy as A  # noqa F401
from eod.tasks.det.models.utils.anchor_generator import build_anchor_generator
from eod.models.losses import build_loss
from eod.utils.general.registry_factory import MODULE_ZOO_REGISTRY

from eod.utils.env.dist_helper import allreduce, env
from eod.tasks.det.models.postprocess.roi_supervisor import build_roi_supervisor
from eod.tasks.det.models.postprocess.roi_predictor import build_roi_predictor


__all__ = ['YoloxPostProcess']


@MODULE_ZOO_REGISTRY.register('yolox_post')
class YoloxPostProcess(nn.Module):
    def __init__(self,
                 num_classes,
                 cfg,
                 norm_on_bbox=False,
                 all_reduce_norm=True,
                 use_l1=False):
        super(YoloxPostProcess, self).__init__()
        self.prefix = self.__class__.__name__
        self.num_classes = num_classes
        self.tocaffe = False
        self.norm_on_bbox = norm_on_bbox
        test_cfg = copy.deepcopy(cfg)
        test_cfg.update(test_cfg.get('test', {}))
        train_cfg = copy.deepcopy(cfg)
        train_cfg.update(train_cfg.get('train', {}))
        self.supervisor = build_roi_supervisor(train_cfg['roi_supervisor'])
        self.predictor = build_roi_predictor(test_cfg['roi_predictor'])
        self.point_generator = build_anchor_generator(cfg['anchor_generator'])
        self.dense_points = self.point_generator.dense_points
        self.cls_loss = build_loss(cfg['cls_loss'])
        self.loc_loss = build_loss(cfg['loc_loss'])
        self.obj_loss = build_loss(cfg['obj_loss'])
        self.l1_loss = nn.L1Loss(reduction="none")
        self.use_l1 = use_l1

        # config
        self.cfg = copy.deepcopy(cfg)
        self.all_reduce_norm = all_reduce_norm

    @property
    def class_activation(self):
        return self.cls_loss.activation_type

    def apply_activation(self, mlvl_preds):
        """apply activation on permuted class prediction
        Arguments:
            - mlvl_pred (list (levels) of tuple (predictions) of Tensor): first element must
            be permuted class prediction with last dimension be class dimension
        """
        mlvl_activated_preds = []
        for lvl_idx, preds in enumerate(mlvl_preds):
            cls_pred = preds[0].sigmoid()
            obj_pred = preds[2].sigmoid()
            cls_pred *= obj_pred
            mlvl_activated_preds.append((cls_pred, *preds[1:]))
        return mlvl_activated_preds

    def permute_preds(self, mlvl_preds):
        """Permute preds from [B, A*C, H, W] to [B, H*W*A, C] """
        mlvl_permuted_preds, mlvl_shapes = [], []
        for lvl_idx, preds in enumerate(mlvl_preds):
            b, _, h, w = preds[0].shape
            k = self.dense_points * h * w
            preds = [p.permute(0, 2, 3, 1).contiguous().view(b, k, -1) for p in preds]
            mlvl_permuted_preds.append(preds)
            mlvl_shapes.append((h, w, k))
        return mlvl_permuted_preds, mlvl_shapes

    def forward(self, input):
        features = input['features']
        strides = input['strides']
        mlvl_preds = input['preds']

        # [B, hi*wi*A, :]
        mlvl_preds, mlvl_shapes = self.permute_preds(mlvl_preds)
        mlvl_shapes = [(*shp, s) for shp, s in zip(mlvl_shapes, strides)]

        # [hi*wi*A, 2], for C4 there is only one layer, for FPN generate anchors for each layer
        mlvl_locations = self.point_generator.get_anchors(mlvl_shapes, device=features[0].device)

        mlvl_ori_loc_preds = None
        if self.use_l1:
            mlvl_ori_loc_preds = []
        for l_ix in range(len(mlvl_locations)):
            # add preds for l1 loss
            if self.use_l1:
                mlvl_ori_loc_preds.append(mlvl_preds[l_ix][1].clone())
            mlvl_preds[l_ix][1][..., :2] *= strides[l_ix]
            mlvl_preds[l_ix][1][..., :2] += mlvl_locations[l_ix]
            if self.norm_on_bbox:
                mlvl_preds[l_ix][1][..., 2:4] = F.relu(mlvl_preds[l_ix][1][..., 2:4])  # , inplace=True)
                mlvl_preds[l_ix][1][..., 2:4] *= strides[l_ix]
            else:
                mlvl_preds[l_ix][1][..., 2:4] = torch.exp(mlvl_preds[l_ix][1][..., 2:4]) * strides[l_ix]

        if self.training:
            targets = self.supervisor.get_targets(mlvl_locations, input, mlvl_preds)
            if not self.use_l1:
                losses = self.get_loss(targets, mlvl_preds)
            else:
                losses = self.get_loss(targets, mlvl_preds, mlvl_ori_loc_preds)
            return losses
        else:
            with torch.no_grad():
                mlvl_preds = self.apply_activation(mlvl_preds)
                results = self.predictor.predict(mlvl_preds)
                return results

    def get_acc(self, cls_pred, cls_targets):
        max_value, cls_targets_acc = cls_targets.topk(1, 1, True, True)
        cls_targets_acc += 1
        cls_targets_acc[max_value == 0] = 0
        acc = A.accuracy_v2(cls_pred, cls_targets_acc, activation_type='qfl')
        return acc

    def get_loss(self, targets, mlvl_preds, mlvl_ori_loc_preds=None):
        mlvl_cls_pred, mlvl_loc_pred, mlvl_obj_pred = zip(*mlvl_preds)
        cls_pred = torch.cat(mlvl_cls_pred, dim=1)
        loc_pred = torch.cat(mlvl_loc_pred, dim=1)
        obj_pred = torch.cat(mlvl_obj_pred, dim=1)
        if self.use_l1:
            ori_loc_preds = torch.cat(mlvl_ori_loc_preds, dim=1)
            del mlvl_ori_loc_preds
        del mlvl_cls_pred, mlvl_loc_pred, mlvl_obj_pred

        cls_targets, reg_targets, obj_targets, ori_reg_targets, fg_masks, num_fgs = targets
        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        if self.use_l1:
            ori_reg_targets = torch.cat(ori_reg_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)
        if self.all_reduce_norm and dist.is_initialized():
            num_fgs = self.get_ave_normalizer(fg_masks)
        else:
            num_fgs = fg_masks.sum().item()
        num_fgs = max(num_fgs, 1)

        cls_pred = cls_pred.reshape(-1, self.num_classes - 1)
        cls_loss = self.cls_loss(cls_pred[fg_masks], cls_targets, normalizer_override=num_fgs)

        acc = self.get_acc(cls_pred[fg_masks], cls_targets)

        loc_target = reg_targets.reshape(-1, 4)
        loc_pred = loc_pred.reshape(-1, 4)
        # loc_preds [xc, yc, w, h] -> [x1, y1, x2, y2]
        loc_pred_x1 = loc_pred[:, 0] - loc_pred[:, 2] / 2
        loc_pred_y1 = loc_pred[:, 1] - loc_pred[:, 3] / 2
        loc_pred_x2 = loc_pred[:, 0] + loc_pred[:, 2] / 2
        loc_pred_y2 = loc_pred[:, 1] + loc_pred[:, 3] / 2
        loc_pred = torch.stack([loc_pred_x1, loc_pred_y1, loc_pred_x2, loc_pred_y2], dim=-1)
        obj_pred = obj_pred.reshape(-1)
        if loc_pred[fg_masks].numel() > 0:
            loc_loss = self.loc_loss(loc_pred[fg_masks], loc_target, normalizer_override=num_fgs)
        else:
            loc_loss = loc_pred[fg_masks].sum()
        obj_loss = self.obj_loss(obj_pred, obj_targets, normalizer_override=num_fgs)
        # add l1 loss
        if self.use_l1:
            if ori_loc_preds.numel() > 0:
                l1_loss = (self.l1_loss(ori_loc_preds.view(-1, 4)[fg_masks], ori_reg_targets)).sum() / num_fgs
            else:
                l1_loss = ori_loc_preds.sum()
        else:
            l1_loss = torch.tensor(0.0).cuda()
        return {
            self.prefix + '.cls_loss': cls_loss,
            self.prefix + '.loc_loss': loc_loss,
            self.prefix + '.obj_loss': obj_loss,
            self.prefix + '.l1_loss': l1_loss,
            self.prefix + '.accuracy': acc
        }

    def get_ave_normalizer(self, _mask):
        ave_mask = torch.sum(_mask)
        if env.world_size > 1:
            allreduce(ave_mask)
        num_gpus = env.world_size
        ave_normalizer = max(ave_mask.item(), 1) / float(num_gpus)
        return ave_normalizer
