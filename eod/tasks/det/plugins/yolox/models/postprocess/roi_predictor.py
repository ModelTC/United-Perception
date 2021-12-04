# Import from third library
import torch
from eod.tasks.det.models.utils.nms_wrapper import nms
from eod.utils.general.registry_factory import ROI_PREDICTOR_REGISTRY


__all__ = ['YoloXPredictor']


@ROI_PREDICTOR_REGISTRY.register('yolox')
class YoloXPredictor(object):
    def __init__(self, num_classes, pre_nms_score_thresh, nms):
        self.pre_nms_score_thresh = pre_nms_score_thresh
        self.nms_cfg = nms
        self.top_n = 300
        self.num_classes = num_classes - 1

    @torch.no_grad()
    def predict(self, preds):
        max_wh = 4096
        preds = [torch.cat(p, dim=2) for p in preds]
        preds = torch.cat(preds, dim=1)
        x1 = preds[..., self.num_classes] - preds[..., self.num_classes + 2] / 2
        y1 = preds[..., self.num_classes + 1] - preds[..., self.num_classes + 3] / 2
        x2 = preds[..., self.num_classes] + preds[..., self.num_classes + 2] / 2
        y2 = preds[..., self.num_classes + 1] + preds[..., self.num_classes + 3] / 2
        preds[..., self.num_classes] = x1
        preds[..., self.num_classes + 1] = y1
        preds[..., self.num_classes + 2] = x2
        preds[..., self.num_classes + 3] = y2

        B = preds.shape[0]
        det_results = []
        for b_ix in range(B):
            pred_per_img = preds[b_ix]
            class_conf, class_pred = torch.max(pred_per_img[:, :self.num_classes], 1, keepdim=True)

            conf_mask = (class_conf.squeeze() >= self.pre_nms_score_thresh).squeeze()
            detections = torch.cat((pred_per_img[:, self.num_classes:-1], class_conf, class_pred.float()), 1)
            detections = detections[conf_mask]
            if not detections.size(0):
                continue

            # batch nms
            cls_hash = detections[:, -1].unsqueeze(-1) * max_wh

            boxes = detections[:, :4] + cls_hash
            scores = detections[:, 4:5]  # .unsqueeze(-1)
            res, keep = nms(torch.cat([boxes, scores], 1), self.nms_cfg)

            rois_keep = detections[keep]

            rois_keep[:, -1] = rois_keep[:, -1] + 1

            # If none remain process next image
            n = rois_keep.shape[0]  # number of boxes
            if not n:
                continue

            if n > self.top_n:
                rois_keep = rois_keep[:self.top_n]

            det_results.append(torch.cat([rois_keep.new_full((rois_keep.shape[0], 1), b_ix), rois_keep], dim=1))
        if len(det_results) == 0:
            det_results.append(preds.new_zeros((1, 7)))
        bboxes = torch.cat(det_results, dim=0)
        return {'dt_bboxes': bboxes}
