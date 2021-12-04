import torch

from eod.tasks.det.models.utils.nms_wrapper import nms
from eod.utils.general.fp16_helper import to_float32
from eod.utils.general.registry_factory import ROI_MERGER_REGISTRY, ROI_PREDICTOR_REGISTRY
from eod.tasks.det.models.postprocess.roi_predictor import build_merger
from eod.tasks.det.models.utils.bbox_helper import xyxy2xywh, xywh2xyxy


__all__ = ['YoloV5Merger', 'YoloV5Predictor']


def norm_croods(data):
    # for yolov5 coords norm
    targets = data['gt_bboxes']
    norm_targets = []
    for batch_id in range(len(targets)):
        gt_bboxes = targets[batch_id]
        gt_bboxes[:, :4] = xyxy2xywh(gt_bboxes[:, :4], stacked=True)
        gt_bboxes[:, 4:5] -= 1
        gt_bboxes = torch.cat([gt_bboxes[:, 4:5], gt_bboxes[:, :4]], 1)
        h1, w1 = data['image'][batch_id].shape[1:]
        gt_bboxes[:, [1, 3]] /= w1
        gt_bboxes[:, [2, 4]] /= h1
        gt_bboxes[:, 1:].clamp_(0, 1)
        norm_targets.append(gt_bboxes)
    data['gt_bboxes'] = norm_targets
    return data


@ROI_MERGER_REGISTRY.register('yolov5')
class YoloV5Merger(object):
    def __init__(self, top_n, nms):
        self.top_n = top_n
        self.nms_cfg = nms

    @torch.no_grad()
    @to_float32
    def merge(self, mlvl_rois):
        """
        Merge rois from different levels together

        Note:
            1. do nms for each class when nms_iou_thresh > 0
            2. keep top_n rois for each image, keep all when top_n <= 0

        Arguments:
            - mlvl_rois (FloatTensor): [N, >=7], (batch_index, x1, y1, x2, y2, score, cls)
        """
        max_wh = 4096  # (pixels) maximum box width and height

        merged_rois = []
        B = int(torch.max(mlvl_rois[:, 0]).item() + 1)
        for bid in range(B):
            img_rois = mlvl_rois[mlvl_rois[:, 0] == bid]
            if img_rois.numel() == 0: continue  # noqa E701

            # batch nms
            cls_hash = img_rois[:, 6:7] * max_wh

            boxes = img_rois[:, 1:5] + cls_hash
            scores = img_rois[:, 5:6]
            res, keep = nms(torch.cat([boxes, scores], 1), self.nms_cfg)

            rois_keep = img_rois[keep]

            # If none remain process next image
            n = rois_keep.shape[0]  # number of boxes
            if not n:
                continue

            if n > self.top_n:
                rois_keep = rois_keep[:self.top_n]
            merged_rois.append(rois_keep)

        if len(merged_rois) > 0:
            merged_rois = torch.cat(merged_rois, dim=0)
        else:
            merged_rois = mlvl_rois.new_zeros((1, 7))
        return merged_rois


@ROI_PREDICTOR_REGISTRY.register('yolov5')
class YoloV5Predictor(object):
    def __init__(self, strides, roi_min_size, conf_thres, single_cls=False, merger=None):

        self.strides = strides
        self.roi_min_size = roi_min_size
        self.conf_thres = conf_thres
        self.single_cls = single_cls

        if merger is not None:
            self.merger = build_merger(merger)
        else:
            self.merger = None

    @torch.no_grad()
    @to_float32
    def predict(self, mlvl_preds, mlvl_anchors):
        mlvl_resutls = []
        for lvl_id, (preds, anchors) in enumerate(zip(mlvl_preds, mlvl_anchors)):
            loc_pred, cls_pred, obj_pred = preds
            results = self.single_level_predict(anchors, self.strides[lvl_id], loc_pred, cls_pred, obj_pred)
            if results is not None:
                mlvl_resutls.append(results)
        if len(mlvl_resutls) > 0:
            results = torch.cat(mlvl_resutls, dim=0)
        else:
            results = mlvl_anchors[0].new_zeros((1, 7))
        if self.merger is not None:
            results = self.merger.merge(results)
        return {'dt_bboxes': results}

    def regression(self, anchors, stride, loc_pred, cls_pred, obj_pred):
        b, _, h, w, pl = loc_pred.shape
        pc = cls_pred.shape[-1]
        anchors = anchors.view(1, -1, 1, 1, 2)

        yv, xv = torch.meshgrid([torch.arange(h), torch.arange(w)])
        grids = torch.stack((xv, yv), 2).to(loc_pred.device).view((1, 1, h, w, 2)).float()

        loc_pred = loc_pred.sigmoid()
        cls_pred = cls_pred.sigmoid()
        obj_pred = obj_pred.sigmoid()
        loc_pred[..., :2] = (loc_pred[..., :2] * 2. - 0.5 + grids) * stride  # xy
        loc_pred[..., 2:] = (loc_pred[..., 2:] * 2) ** 2 * anchors

        loc_pred = loc_pred.view(b, -1, pl)
        cls_pred = cls_pred.view(b, -1, pc)
        obj_pred = obj_pred.view(b, -1)
        return loc_pred, cls_pred, obj_pred

    def single_level_predict(self, anchors, stride, loc_pred, cls_pred, obj_pred):
        loc_pred, cls_pred, obj_pred = self.regression(anchors, stride, loc_pred, cls_pred, obj_pred)
        pred_keeps = obj_pred > self.conf_thres  # candidates

        roi_keeps = []
        bs = loc_pred.shape[0]
        for bid in range(bs):
            b_pred_keeps = pred_keeps[bid]
            box_keep = loc_pred[bid][b_pred_keeps]
            if not box_keep.shape[0]:
                continue
            cls_keep = cls_pred[bid][b_pred_keeps]
            if not self.single_cls:
                # Compute conf
                cls_keep *= obj_pred[bid][b_pred_keeps].view(-1, 1)  # conf = obj_conf * cls_conf

            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            box = xywh2xyxy(box_keep, stacked=True)

            if not self.single_cls:
                box_idx, cls_idx = (cls_keep[:, :] > self.conf_thres).nonzero(as_tuple=False).T
                rois = torch.cat((box[box_idx], cls_keep[box_idx, cls_idx, None],
                                  (cls_idx[:, None].float() + 1)), 1)
            else:
                conf, cls_idx = cls_keep[:, :].max(1, keepdim=True)
                rois = torch.cat((box, conf, cls_idx.float() + 1), 1)[conf.view(-1) > self.conf_thres]

            # If none remain process next image
            n = rois.shape[0]  # number of boxes
            if not n:
                continue
            batch_ix = torch.tensor([bid] * n, device=box.device, dtype=box.dtype).unsqueeze(-1)
            rois = torch.cat([batch_ix, rois], 1)
            roi_keeps.append(rois)
        if roi_keeps:
            return torch.cat(roi_keeps, 0)
        else:
            return None
