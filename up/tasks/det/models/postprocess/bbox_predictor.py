import torch

from up.extensions import nms
from up.tasks.det.models.utils.bbox_helper import (
    clip_bbox,
    offset2bbox,
    box_voting,
    unnormalize_offset,
)
from up.utils.general.fp16_helper import to_float32
from up.utils.general.registry_factory import BBOX_PREDICTOR_REGISTRY
from up.utils.general.log_helper import default_logger as logger

__all__ = ['BboxPredictor', 'BboxPredictorMultiCls']


@BBOX_PREDICTOR_REGISTRY.register('faster')
class BboxPredictor(object):
    def __init__(self, bbox_normalize, bbox_score_thresh, top_n, share_location,
                 nms, bbox_vote=None):
        """
        Args:
            - bbox_normalize: `dict` with `means` and `stds`
            - bbox_score_thresh: score thresh of detected bboxes
            - top_n: maximum number of detected bboxes
            - share_location: class agnostic regression
            - nms: config of nms
            - bbox_vote: config of bbox_vote
        """
        self.offset_means = bbox_normalize['means']
        self.offset_stds = bbox_normalize['stds']
        self.bbox_score_thresh = bbox_score_thresh
        self.top_n = top_n
        self.share_location = share_location
        self.nms_cfg = nms
        self.bbox_vote_cfg = bbox_vote

    @torch.no_grad()
    @to_float32
    def predict(self, proposals, preds, image_info, start_idx):
        """
        Arguments:
            - proposals (FloatTensor, fp32): [N, >=5] (batch_ix, x1, y1, x2, y2, ...)
            - preds:
                - cls_pred (FloatTensor, fp32): [N, C], C is num_classes, including background class
                - loc_pred (FloatTensor, fp32): [N, C*4] or [N, 4]
            - image_info (list of FloatTensor): [B, >=3] (image_h, image_w, scale_factor, ...)

        Returns:
            bboxes (FloatTensor): [R, 7], (batch_ix, x1, y1, x2, y2, score, cls)
        """
        cls_pred, loc_pred = preds

        # acquire all predicted bboxes
        B, (R, C) = len(image_info), cls_pred.shape
        loc_pred = unnormalize_offset(loc_pred.view(-1, 4), self.offset_means, self.offset_stds)
        bboxes = offset2bbox(proposals[:, 1:1 + 4], loc_pred.view(R, -1))
        bboxes = torch.cat([bboxes for _ in range(C)], dim=1) if self.share_location else bboxes
        bboxes = torch.cat([bboxes.view(-1, 4), cls_pred.view(-1, 1)], dim=1).view(R, -1)

        detected_bboxes = []
        for b_ix in range(B):
            img_inds = torch.nonzero(proposals[:, 0] == b_ix).reshape(-1)
            if len(img_inds) == 0:
                continue  # noqa E701
            img_bboxes = bboxes[img_inds]
            img_scores = cls_pred[img_inds]

            # clip bboxes which are out of bounds
            img_bboxes = img_bboxes.view(-1, 5)
            img_bboxes[:, :4] = clip_bbox(img_bboxes[:, :4], image_info[b_ix])
            img_bboxes = img_bboxes.view(-1, C * 5)

            inds_all = img_scores > self.bbox_score_thresh
            result_bboxes = []
            for cls in range(start_idx, C):
                # keep well-predicted bbox
                inds_cls = inds_all[:, cls].nonzero().reshape(-1)
                img_bboxes_cls = img_bboxes[inds_cls, cls * 5:(cls + 1) * 5]
                if len(img_bboxes_cls) == 0:
                    continue  # noqa E701

                # do nms, can choose the nms type, naive or softnms
                order = img_bboxes_cls[:, 4].sort(descending=True)[1]
                img_bboxes_cls = img_bboxes_cls[order]

                img_bboxes_cls_nms, keep_inds = nms(img_bboxes_cls, self.nms_cfg)

                if self.bbox_vote_cfg is not None:
                    img_bboxes_cls_nms = box_voting(img_bboxes_cls_nms,
                                                    img_bboxes_cls,
                                                    self.bbox_vote_cfg.get('vote_th', 0.9),
                                                    scoring_method=self.bbox_vote_cfg.get('scoring_method', 'id'))

                ix = img_bboxes_cls.new_full((img_bboxes_cls_nms.shape[0], 1), b_ix)
                c = img_bboxes_cls.new_full((img_bboxes_cls_nms.shape[0], 1), (cls + 1 - start_idx))
                result_bboxes.append(torch.cat([ix, img_bboxes_cls_nms, c], dim=1))

            if len(result_bboxes) == 0:
                continue  # noqa E701

            # keep the top_n well-predicted bboxes per image
            result_bboxes = torch.cat(result_bboxes, dim=0)
            if self.top_n >= 0 and self.top_n < result_bboxes.shape[0]:
                num_base = result_bboxes.shape[0] - self.top_n + 1
                thresh = torch.kthvalue(result_bboxes[:, 5].cpu(), num_base)[0]
                keep_inds = result_bboxes[:, 5] >= thresh.item()
                result_bboxes = result_bboxes[keep_inds][:self.top_n]
            detected_bboxes.append(result_bboxes)

        if len(detected_bboxes) == 0:
            dt_bboxes = proposals.new_zeros((1, 7))
        else:
            dt_bboxes = torch.cat(detected_bboxes, dim=0)

        return {'dt_bboxes': dt_bboxes}


@BBOX_PREDICTOR_REGISTRY.register('faster_multicls')
class BboxPredictorMultiCls(BboxPredictor):
    def __init__(self,
                 bbox_normalize,
                 bbox_score_thresh,
                 top_n,
                 share_location,
                 nms,
                 bbox_vote=None,
                 pre_nms_top_n=6000):
        super().__init__(bbox_normalize, bbox_score_thresh, top_n, share_location, nms, bbox_vote)
        self.INF = 4096
        self.pre_nms_top_n = pre_nms_top_n
        logger.warning(f"using pre_nms_top_n {pre_nms_top_n} to speed up nms")
        assert self.bbox_vote_cfg is None, "bbox vote is not support!!!"

    @torch.no_grad()
    @to_float32
    def predict(self, proposals, preds, image_info, start_idx):
        """
        Arguments:
            - proposals (FloatTensor, fp32): [N, >=5] (batch_ix, x1, y1, x2, y2, ...)
            - preds:
                - cls_pred (FloatTensor, fp32): [N, C], C is num_classes, including background class
                - loc_pred (FloatTensor, fp32): [N, C*4] or [N, 4]
            - image_info (list of FloatTensor): [B, >=3] (image_h, image_w, scale_factor, ...)

        Returns:
            bboxes (FloatTensor): [R, 7], (batch_ix, x1, y1, x2, y2, score, cls)
        """
        cls_pred, loc_pred = preds

        # acquire all predicted bboxes
        B, (R, C) = len(image_info), cls_pred.shape
        loc_pred = unnormalize_offset(loc_pred.view(-1, 4), self.offset_means, self.offset_stds)
        bboxes = offset2bbox(proposals[:, 1:1 + 4], loc_pred.view(R, -1))
        bboxes = torch.cat([bboxes for _ in range(C)], dim=1) if self.share_location else bboxes
        bboxes = torch.cat([bboxes.view(-1, 4), cls_pred.view(-1, 1)], dim=1).view(R, -1)

        detected_bboxes = []
        for b_ix in range(B):
            img_inds = torch.nonzero(proposals[:, 0] == b_ix).reshape(-1)
            if len(img_inds) == 0: continue  # noqa E701
            img_bboxes = bboxes[img_inds]
            img_scores = cls_pred[img_inds]

            # clip bboxes which are out of bounds
            img_bboxes = img_bboxes.view(-1, 5)
            img_bboxes[:, :4] = clip_bbox(img_bboxes[:, :4], image_info[b_ix])

            img_scores = img_scores.flatten()

            # filter bbox base bbox score thresh
            inds_all = img_scores > self.bbox_score_thresh
            img_scores = img_scores[inds_all]
            img_bboxes = img_bboxes[inds_all]

            # sort scores and get classes idxs
            img_scores, topk_idxs = img_scores.sort(descending=True)
            classes_idxs = (topk_idxs % C).float()

            # exclude background
            foreground_idxs = classes_idxs >= start_idx
            classes_idxs = classes_idxs[foreground_idxs]
            topk_idxs = topk_idxs[foreground_idxs]

            # in order to speed nms, pre_nms_top_n are set
            _pre_nms_top_n = min(self.pre_nms_top_n, img_bboxes.shape[0])
            topk_idxs = topk_idxs[:_pre_nms_top_n]
            img_bboxes = img_bboxes[topk_idxs]
            classes_idxs = classes_idxs[:_pre_nms_top_n]

            # convert bbox base different class
            temp_bboxes = img_bboxes.clone()
            temp_bboxes[:, :4] += (classes_idxs * self.INF).view(-1, 1).expand_as(temp_bboxes[:, :4])

            # batch nms
            _, keep_inds = nms(temp_bboxes, self.nms_cfg)
            del temp_bboxes
            img_bboxes_nms = img_bboxes[keep_inds]
            classes_idxs = classes_idxs[keep_inds]
            ix = img_bboxes_nms.new_full((img_bboxes_nms.shape[0], 1), b_ix)
            result_bboxes = torch.cat([ix, img_bboxes_nms, classes_idxs.view(-1, 1)], dim=1)

            if len(result_bboxes) == 0: continue  # noqa E701

            # keep the top_n well-predicted bboxes per image
            if self.top_n >= 0 and self.top_n < result_bboxes.shape[0]:
                num_base = result_bboxes.shape[0] - self.top_n + 1
                thresh = torch.kthvalue(result_bboxes[:, 5].cpu(), num_base)[0]
                keep_inds = result_bboxes[:, 5] >= thresh.item()
                result_bboxes = result_bboxes[keep_inds][:self.top_n]
            detected_bboxes.append(result_bboxes)

        if len(detected_bboxes) == 0:
            dt_bboxes = proposals.new_zeros((1, 7))
        else:
            dt_bboxes = torch.cat(detected_bboxes, dim=0)

        return {'dt_bboxes': dt_bboxes}


def build_bbox_predictor(predictor_cfg):
    return BBOX_PREDICTOR_REGISTRY.build(predictor_cfg)
