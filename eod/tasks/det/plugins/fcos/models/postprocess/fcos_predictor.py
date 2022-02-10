# Import from third library
import torch
from eod.utils.general.registry_factory import FCOS_PREDICTOR_REGISTRY
from eod.tasks.det.models.postprocess.roi_predictor import RoiPredictor

__all__ = ['FcosPredictor']


@FCOS_PREDICTOR_REGISTRY.register('fcos')
class FcosPredictor(RoiPredictor):
    def __init__(self, pre_nms_score_thresh, pre_nms_top_n, post_nms_top_n,
                 roi_min_size, merger, nms=None, all_in_one=False, norm_on_bbox=False):

        super().__init__(pre_nms_score_thresh, pre_nms_top_n,
                         post_nms_top_n, roi_min_size, merger, nms)
        self.all_in_one = all_in_one
        self.norm_on_bbox = norm_on_bbox

    def predict(self, locations, preds, image_info, strides=None, return_pos_inds=False):
        """
        Arguments:
            - locations (FloatTensor, fp32): [K, 2]
            - preds[0] (cls_pred, FloatTensor, fp32): [B, K, C], C[i] -> class i+1, background class is excluded
            - preds[1] (loc_pred, FloatTensor, fp32): [B, K, 4]
            - image_info (list of FloatTensor): [B, >=2] (image_h, image_w, ...)

        Returns:
            rois (FloatTensor): [N, >=7] (batch_ix, x1, y1, x2, y2, score, cls)
        """
        output = {}
        if self.norm_on_bbox:
            self.strides = strides
        results = super().predict(locations, preds, image_info, return_pos_inds)
        if return_pos_inds:
            output.update({'pos_inds': results['pos_inds']})

        bboxes = results['dt_bboxes']
        if self.all_in_one:
            bboxes[:, -1] = torch.clamp(bboxes[:, -1], max=1)

        output.update({'dt_bboxes': bboxes})
        return output

    def offset2bbox(self, locations, offset):
        x1 = locations[:, 0] - offset[:, 0]
        y1 = locations[:, 1] - offset[:, 1]
        x2 = locations[:, 0] + offset[:, 2]
        y2 = locations[:, 1] + offset[:, 3]
        return torch.stack([x1, y1, x2, y2], -1)

    def regression(self, locations, preds, image_info):
        cls_pred, loc_pred = preds[:2]
        B, K, C = cls_pred.shape
        if self.norm_on_bbox:
            loc_pred *= self.strides[self.lvl]
        concat_locations = torch.stack([locations.clone() for _ in range(B)])
        rois = self.offset2bbox(concat_locations.view(B * K, 2), loc_pred.view(B * K, 4)).view(B, K, 4)
        return rois


def build_fcos_predictor(predictor_cfg):
    return FCOS_PREDICTOR_REGISTRY.build(predictor_cfg)
