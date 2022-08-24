import torch

from up.utils.general.registry_factory import ROI_PREDICTOR_REGISTRY
from up.tasks.det.models.utils.bbox_helper import clip_bbox, filter_by_size

__all__ = ['OnenetPredictor']


@ROI_PREDICTOR_REGISTRY.register('onenet')
class OnenetPredictor(object):
    def __init__(self, topk_boxes=100, min_size=0.0):
        self.topk_boxes = topk_boxes
        self.min_size = min_size

    @torch.no_grad()
    def predict(self, preds, input):
        img_info = input['image_info']
        cls_pred = preds[0].permute(0, 2, 1)
        loc_pred = preds[1].permute(0, 2, 1)

        B = preds[0].shape[0]
        det_results = []

        for i in range(B):
            cls_pred_per_img = cls_pred[i]
            loc_pred_per_img = loc_pred[i]
            topk_score_per_c, topk_ind_per_c = torch.topk(cls_pred_per_img, k=self.topk_boxes)
            topk_boxes_per_c = loc_pred_per_img[:, topk_ind_per_c.reshape(-1)]
            topk_score_all, topk_ind_all = torch.topk(topk_score_per_c.reshape(-1), k=self.topk_boxes)
            topk_boxes_all = topk_boxes_per_c[:, topk_ind_all].transpose(0, 1)

            topk_label_all = topk_ind_all // self.topk_boxes + 1

            detections = torch.cat([topk_boxes_all, topk_score_all.unsqueeze(-1), topk_label_all.unsqueeze(-1)], dim=1)

            detections = clip_bbox(detections, img_info[i])
            detections, _ = filter_by_size(detections, self.min_size)

            det_results.append(torch.cat([detections.new_full((detections.shape[0], 1), i), detections], dim=1))

        if len(det_results):
            det_results.append(cls_pred.new_zeros((1, 7)))
        bboxes = torch.cat(det_results, dim=0)
        return {'dt_bboxes': bboxes}


def build_onenet_predictor(predictor_cfg):
    return ROI_PREDICTOR_REGISTRY.build(predictor_cfg)
