import torch
import torch.nn as nn

from eod.tasks.det.models.utils.anchor_generator import build_anchor_generator
from eod.models.losses import build_loss
from eod.utils.env.dist_helper import get_world_size
from eod.utils.general.registry_factory import MODULE_ZOO_REGISTRY
from eod.tasks.det.models.postprocess.roi_supervisor import build_roi_supervisor
from eod.tasks.det.models.postprocess.roi_predictor import build_roi_predictor
from eod.tasks.det.models.utils.bbox_helper import xywh2xyxy


__all__ = ['YoloV5PostProcess']


@MODULE_ZOO_REGISTRY.register('yolov5_post')
class YoloV5PostProcess(nn.Module):
    def __init__(self,
                 num_classes,
                 cfg,
                 num_levels=3,
                 num_anchors_per_level=3,
                 out_strides=[8, 16, 32],
                 loss_weights=[0.05, 1.0, 0.5],
                 obj_loss_layer_weights=[4.0, 1.0, 0.4],
                 initializer=None):
        super(YoloV5PostProcess, self).__init__()

        self.prefix = self.__class__.__name__
        self.num_classes = num_classes - 1
        self.single_cls = (num_classes == 2)

        self.tocaffe = False
        self.num_levels = num_levels
        self.num_anchors_per_level = num_anchors_per_level
        # P3-5 or P3-6 loss weight
        self.obj_loss_layer_weights = obj_loss_layer_weights
        # box loss weight, obj loss weight, cls loss weight, the input order is important
        scale = 3 / self.num_levels
        loss_weights = [loss_weights[i] * scale for i in range(3)]
        # obj loss weight is little diff
        loss_weights[1] *= (1.4 if num_levels >= 4 else 1.)
        self.loss_type_weights = loss_weights
        self.strides = out_strides

        cfg['anchor_generator']['kwargs']['num_levels'] = num_levels
        cfg['anchor_generator']['kwargs']['num_anchors_per_level'] = num_anchors_per_level
        cfg['roi_supervisor']['kwargs'] = {}
        cfg['roi_supervisor']['kwargs']['strides'] = out_strides
        cfg['roi_supervisor']['kwargs']['num_classes'] = self.num_classes
        cfg['roi_predictor']['kwargs']['strides'] = out_strides
        cfg['roi_predictor']['kwargs']['single_cls'] = self.single_cls
        self.anchor_generator = build_anchor_generator(cfg['anchor_generator'])
        self.supervisor = build_roi_supervisor(cfg['roi_supervisor'])
        self.test_predictor = build_roi_predictor(cfg['roi_predictor'])

        self.box_loss = build_loss(cfg['box_loss'])
        self.obj_loss = build_loss(cfg['obj_loss'])
        self.cls_loss = build_loss(cfg['cls_loss'])

    @property
    def num_anchors(self):
        return self.anchor_generator.num_anchors

    @property
    def anchor_shapes(self):
        return self.anchor_generator._shapes

    def permute_preds(self, mlvl_preds):
        permute_mlvl_preds = []
        for loc_pred, cls_pred, obj_pred in mlvl_preds:
            bs, ls, ny, nx = loc_pred.shape
            cs = cls_pred.shape[1]
            loc_pred = loc_pred.view(bs, self.num_anchors_per_level,
                                     ls // self.num_anchors_per_level, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            cls_pred = cls_pred.view(bs, self.num_anchors_per_level,
                                     cs // self.num_anchors_per_level, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            obj_pred = obj_pred.view(bs, self.num_anchors_per_level,
                                     1, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            permute_mlvl_preds.append((loc_pred, cls_pred, obj_pred))
        return permute_mlvl_preds

    def forward(self, input):
        raw_mlvl_preds = input['preds']
        mlvl_preds = self.permute_preds(raw_mlvl_preds)

        output = {}
        mlvl_anchors = torch.tensor(self.anchor_shapes.reshape(self.num_levels, -1, 2)).type_as(mlvl_preds[0][0])

        if self.training:
            mlvl_targets = self.supervisor.get_targets(input, mlvl_preds, mlvl_anchors)
            losses = self.get_loss(mlvl_targets, mlvl_preds)
            output.update(losses)
        else:
            results = self.test_predictor.predict(mlvl_preds, mlvl_anchors)
            output.update(results)
        return output

    def get_loss(self, mlvl_targets, mlvl_preds):
        device = mlvl_preds[0][0].device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)

        for lvl_id, (targets, preds) in enumerate(zip(mlvl_targets, mlvl_preds)):
            loc_pred, cls_pred, obj_pred = preds
            b, l, h, w, pl = loc_pred.shape

            xy_pred = loc_pred[..., :2].sigmoid() * 2. - 0.5
            wh_pred = (loc_pred[..., 2:].sigmoid() * 2) ** 2

            anchors = self.anchor_shapes[lvl_id] / self.strides[lvl_id]
            multi_op = torch.tensor(anchors).type_as(loc_pred).reshape(1, anchors.shape[0], 1, 1, 2)
            wh_pred = wh_pred * multi_op

            box_gt, cls_gt, box_mask, cls_mask = targets

            box_pred = torch.cat((xy_pred, wh_pred), -1)     # predicted box

            iou_loss = self.box_loss(xywh2xyxy(box_pred.reshape(-1, 4), stacked=True),
                                     xywh2xyxy(box_gt.reshape(-1, 4), stacked=True)) * box_mask.reshape(-1)

            pos_grid = torch.where(iou_loss)[0]
            iou_loss_keep = iou_loss[pos_grid]
            lbox += iou_loss_keep.sum() / max(1, iou_loss_keep.size(0))

            obj_gt = ((1 - iou_loss).detach() * box_mask.reshape(-1)).clamp(0).type_as(obj_pred).reshape(b, l, h, w)

            lobj += self.obj_loss(obj_pred, obj_gt).mean() * self.obj_loss_layer_weights[lvl_id]    # obj loss

            if not self.single_cls:              # for multi class training, single class do not consider the cls_loss
                cls_loss = self.cls_loss(cls_pred, cls_gt) * cls_mask.reshape(-1)
                cls_loss = cls_loss.reshape(-1, self.num_classes)[pos_grid].reshape(-1)
                lcls += cls_loss.sum() / max(1, cls_loss.size(0))   # cls loss

        bs = mlvl_preds[0][0].shape[0]
        world_size = get_world_size()
        lbox *= (self.loss_type_weights[0] * bs * world_size)
        lobj *= (self.loss_type_weights[1] * bs * world_size)
        lcls *= (self.loss_type_weights[2] * bs * world_size)

        return {
            self.prefix + '.cls_loss': lcls,
            self.prefix + '.obj_loss': lobj,
            self.prefix + '.box_loss': lbox
        }
