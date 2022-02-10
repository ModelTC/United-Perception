# Import from third library
import torch
from eod.tasks.det.models.utils.nms_wrapper import nms
from eod.utils.general.fp16_helper import to_float32
from eod.utils.general.registry_factory import ROI_MERGER_REGISTRY, ROI_PREDICTOR_REGISTRY
from eod.tasks.det.models.utils.bbox_helper import (
    clip_bbox,
    filter_by_size,
    offset2bbox
)

__all__ = [
    'RoiPredictor',
    'RoiPredictorMultiCls',
    'RpnMerger',
    'RetinaMerger',
    'RetinaMergerMultiCls'
]


@ROI_PREDICTOR_REGISTRY.register('base')
class RoiPredictor(object):
    """Predictor for the first stage
    """

    def __init__(self,
                 pre_nms_score_thresh,
                 pre_nms_top_n,
                 post_nms_top_n,
                 roi_min_size,
                 merger=None,
                 nms=None,
                 clip_box=True):
        self.pre_nms_score_thresh = pre_nms_score_thresh
        self.pre_nms_top_n = pre_nms_top_n
        self.post_nms_top_n = post_nms_top_n
        self.nms_cfg = nms
        self.roi_min_size = roi_min_size
        self.clip_box = clip_box
        if merger is not None:
            self.merger = build_merger(merger)
        else:
            self.merger = None

    @torch.no_grad()
    @to_float32
    def predict(self, mlvl_anchors, mlvl_preds, image_info, return_pos_inds=None):
        mlvl_resutls = []
        num_points = 0
        for lvl, (anchors, preds) in enumerate(zip(mlvl_anchors, mlvl_preds)):
            self.lvl = lvl
            results = self.single_level_predict(anchors, preds, image_info, num_points, return_pos_inds)
            num_points += len(anchors)
            mlvl_resutls.append(results)
        if len(mlvl_resutls) > 0:
            results = torch.cat(mlvl_resutls, dim=0)
        else:
            results = mlvl_anchors[0].new_zeros((1, 8)) if return_pos_inds else mlvl_anchors[0].new_zeros((1, 7))
        if self.merger is not None:
            results = self.merger.merge(results)
        if return_pos_inds:
            results, pos_inds = torch.split(results, [7, 1], dim=1)
            return {'dt_bboxes': results, 'pos_inds': pos_inds}
        return {'dt_bboxes': results}

    def regression(self, anchors, preds, image_info):
        cls_pred, loc_pred = preds[:2]
        B, K = cls_pred.shape[:2]
        concat_anchors = torch.stack([anchors.clone() for _ in range(B)])
        rois = offset2bbox(concat_anchors.view(B * K, 4), loc_pred.view(B * K, 4)).view(B, K, 4)  # noqa
        return rois

    def single_level_predict(self, anchors, preds, image_info, num_points, return_pos_inds=None):
        """
        Arguments:
            - anchors (FloatTensor, fp32): [K, 4]
            - preds[0] (cls_pred, FloatTensor, fp32): [B, K, C], C[i] -> class i+1, background class is excluded
            - preds[1] (loc_pred, FloatTensor, fp32): [B, K, 4]
            - image_info (list of FloatTensor): [B, >=2] (image_h, image_w, ...)

        Returns:
            rois (FloatTensor): [N, >=7] (batch_ix, x1, y1, x2, y2, score, cls)
        """

        rois = self.regression(anchors, preds, image_info)

        cls_pred = preds[0]
        B, K, C = cls_pred.shape
        roi_min_size = self.roi_min_size
        pre_nms_top_n = self.pre_nms_top_n
        pre_nms_top_n = pre_nms_top_n if pre_nms_top_n > 0 else K
        post_nms_top_n = self.post_nms_top_n
        # if featmap size is too large, filter by score thresh to reduce computation
        if K > 120:
            score_thresh = self.pre_nms_score_thresh
        else:
            score_thresh = 0

        batch_rois = []
        for b_ix in range(B):
            # clip rois and filter rois which are too small
            image_rois = rois[b_ix]
            image_inds = torch.arange(K, dtype=torch.int64, device=image_rois.device)
            if self.clip_box:
                image_rois = clip_bbox(image_rois, image_info[b_ix])
            image_rois, filter_inds = filter_by_size(image_rois, roi_min_size)
            image_cls_pred = cls_pred[b_ix][filter_inds]
            image_inds = image_inds[filter_inds]
            if image_rois.numel() == 0:
                continue  # noqa E701

            for cls in range(C):
                cls_rois = image_rois
                scores = image_cls_pred[:, cls]
                cls_inds = image_inds
                assert not torch.isnan(scores).any()
                if score_thresh > 0:
                    # to reduce computation
                    keep_idx = torch.nonzero(scores > score_thresh).reshape(-1)
                    if keep_idx.numel() == 0:
                        continue  # noqa E701
                    cls_rois = cls_rois[keep_idx]
                    scores = scores[keep_idx]
                    cls_inds = cls_inds[keep_idx]
                # do nms per image, only one class
                _pre_nms_top_n = min(pre_nms_top_n, scores.shape[0])
                scores, order = scores.topk(_pre_nms_top_n, sorted=True)
                cls_rois = cls_rois[order, :]
                cls_inds = cls_inds[order]
                cls_rois = torch.cat([cls_rois, scores[:, None], cls_inds[:, None]], dim=1)

                if self.nms_cfg is not None:
                    cls_rois, keep_idx = nms(cls_rois, self.nms_cfg)
                if post_nms_top_n > 0:
                    cls_rois = cls_rois[:post_nms_top_n]

                ix = cls_rois.new_full((cls_rois.shape[0], 1), b_ix)
                c = cls_rois.new_full((cls_rois.shape[0], 1), cls + 1)
                cls_rois, cls_inds = torch.split(cls_rois, [5, 1], dim=1)
                if return_pos_inds:
                    cls_rois = torch.cat([ix, cls_rois, c, cls_inds], dim=1)
                else:
                    cls_rois = torch.cat([ix, cls_rois, c], dim=1)
                batch_rois.append(cls_rois)

        if len(batch_rois) == 0:
            return anchors.new_zeros((1, 8)) if return_pos_inds else anchors.new_zeros((1, 7))
        results = torch.cat(batch_rois, dim=0)
        if return_pos_inds:
            results[:, 7].add_(num_points)
        return results


@ROI_PREDICTOR_REGISTRY.register('base_multicls')
class RoiPredictorMultiCls(RoiPredictor):
    """Predictor for the first stage
    """

    def __init__(self, pre_nms_score_thresh,
                 pre_nms_top_n, post_nms_top_n, roi_min_size, merger=None, nms=None, clip_box=True):
        super().__init__(pre_nms_score_thresh, pre_nms_top_n, post_nms_top_n, roi_min_size, merger, nms, clip_box)
        self.INF = 4096

    def single_level_predict(self, anchors, preds, image_info):
        """
        Arguments:
            - anchors (FloatTensor, fp32): [K, 4]
            - preds[0] (cls_pred, FloatTensor, fp32): [B, K, C], C[i] -> class i+1, background class is excluded
            - preds[1] (loc_pred, FloatTensor, fp32): [B, K, 4]
            - image_info (list of FloatTensor): [B, >=2] (image_h, image_w, ...)

        Returns:
            rois (FloatTensor): [N, >=7] (batch_ix, x1, y1, x2, y2, score, cls)
        """

        rois = self.regression(anchors, preds, image_info)

        cls_pred = preds[0]
        B, K, C = cls_pred.shape
        roi_min_size = self.roi_min_size
        pre_nms_top_n = self.pre_nms_top_n
        pre_nms_top_n = pre_nms_top_n if pre_nms_top_n > 0 else K
        # post_nms_top_n = self.post_nms_top_n
        # if featmap size is too large, filter by score thresh to reduce computation
        if K > 120:
            score_thresh = self.pre_nms_score_thresh
        else:
            score_thresh = 0

        batch_rois = []
        for b_ix in range(B):
            # clip rois and filter rois which are too small
            image_rois = rois[b_ix]
            if self.clip_box:
                image_rois = clip_bbox(image_rois, image_info[b_ix])
            image_rois, filter_inds = filter_by_size(image_rois, roi_min_size)
            image_cls_pred = cls_pred[b_ix][filter_inds]
            if image_rois.numel() == 0:
                continue  # noqa E701

            image_cls_pred = image_cls_pred.flatten()
            _pre_nms_top_n = min(pre_nms_top_n, image_rois.shape[0])
            scores, topk_idxs = image_cls_pred.sort(descending=True)
            scores = scores[:_pre_nms_top_n]
            topk_idxs = topk_idxs[:_pre_nms_top_n]
            if score_thresh > 0:
                # to reduce computation
                keep_idx = torch.nonzero(scores > score_thresh).reshape(-1)
                if keep_idx.numel() == 0:
                    continue  # noqa E701
                scores = scores[keep_idx]
                topk_idxs = topk_idxs[keep_idx]
            anchor_idxs = topk_idxs // C
            classes_idxs = ((topk_idxs % C) + 1).float().view(-1, 1)
            cls_rois = image_rois[anchor_idxs]
            scores = scores.view(-1, 1)

            if self.nms_cfg is not None:
                boxes = cls_rois + (classes_idxs * self.INF).expand_as(cls_rois)
                _, keep_idx = nms(torch.cat((boxes, scores), dim=1), self.nms_cfg)
                cls_rois = cls_rois[keep_idx]
                scores = scores[keep_idx]
                classes_idxs = classes_idxs[keep_idx]
            ix = cls_rois.new_full((cls_rois.shape[0], 1), b_ix)
            cls_rois = torch.cat([ix, cls_rois, scores, classes_idxs], dim=1)
            batch_rois.append(cls_rois)
        if len(batch_rois) == 0:
            return anchors.new_zeros((1, 7))
        return torch.cat(batch_rois, dim=0)


@ROI_MERGER_REGISTRY.register('rpn')
class RpnMerger(object):
    """Concat results from all levels and keep topk
    """

    def __init__(self, top_n, topk_in_a_batch=False):
        self.top_n = top_n
        self.topk_in_a_batch = topk_in_a_batch

    @torch.no_grad()
    @to_float32
    def merge(self, mlvl_rois):
        """Merge rois from all levels by keeping topk rois
        Arguments:
            - mlvl_rois (FloatTensor): [N, >=7], (batch_idx, x1, y1, x2, y2, score, cls)
        """
        merged_rois = []
        # TODO: we choose topk in a batch when training to align performance with maskrcnn-benchmark.
        # This might lead to ~0.5 mAP performace gain.
        if self.topk_in_a_batch:
            topk = min(self.top_n, mlvl_rois.shape[0])
            scores, order = torch.topk(mlvl_rois[:, 5], k=topk, sorted=True)
            return mlvl_rois[order]
        else:
            merged_rois = []
            B = int(torch.max(mlvl_rois[:, 0]).item() + 1)
            for b_ix in range(B):
                rois = mlvl_rois[mlvl_rois[:, 0] == b_ix]
                if rois.numel() == 0:
                    continue  # noqa E701
                topk = min(self.top_n, rois.shape[0])
                scores, order = torch.topk(rois[:, 5], k=topk, sorted=True)
                merged_rois.append(rois[order])
            if len(merged_rois) > 0:
                merged_rois = torch.cat(merged_rois, dim=0)
            else:
                merged_rois = mlvl_rois.new_zeros((1, 7))
            return merged_rois


@ROI_MERGER_REGISTRY.register('retina')
class RetinaMerger(object):
    """Merge results from multi-levels
    1. concat all results
    2. nms
    3. keep topk
    """

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
        merged_rois = []
        B = int(torch.max(mlvl_rois[:, 0]).item() + 1)
        for b_ix in range(B):
            img_rois = mlvl_rois[mlvl_rois[:, 0] == b_ix]
            if img_rois.numel() == 0:
                continue  # noqa E701

            classes = torch.unique(img_rois[:, 6].int()).cpu().numpy().tolist()
            all_cls_rois = []
            for cls in classes:
                cls_rois = img_rois[img_rois[:, 6] == cls]
                if cls_rois.numel() == 0:
                    continue  # noqa E701
                _, indices = nms(cls_rois[:, 1:6], self.nms_cfg)
                all_cls_rois.append(cls_rois[indices])
            if len(all_cls_rois) == 0:
                continue  # noqa E701
            rois = torch.cat(all_cls_rois, dim=0)

            if self.top_n < rois.shape[0]:
                _, inds = torch.topk(rois[:, 5], self.top_n)
                rois = rois[inds]
            merged_rois.append(rois)
        if len(merged_rois) > 0:
            merged_rois = torch.cat(merged_rois, dim=0)
        else:
            merged_rois = mlvl_rois.new_zeros((1, 7))
        return merged_rois


@ROI_MERGER_REGISTRY.register('retina_multicls')
class RetinaMergerMultiCls(RetinaMerger):
    """Merge results from multi-levels
    1. concat all results
    2. nms
    3. keep topk
    """

    def __init__(self, top_n, nms):
        super().__init__(top_n, nms)
        self.INF = 4096

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
        merged_rois = []
        B = int(torch.max(mlvl_rois[:, 0]).item() + 1)
        for b_ix in range(B):
            img_rois = mlvl_rois[mlvl_rois[:, 0] == b_ix]
            if img_rois.numel() == 0:
                continue  # noqa E701
            img_rois = img_rois[img_rois[:, 5].argsort(descending=True)]
            c = img_rois[:, 6]
            boxes = img_rois[:, 1:6].clone()
            boxes[:, :4] += c.view(-1, 1) * self.INF
            _, indices = nms(boxes, self.nms_cfg)
            del boxes
            rois = img_rois[indices]
            if self.top_n < rois.shape[0]:
                rois = rois[:self.top_n]
            merged_rois.append(rois)
        if len(merged_rois) > 0:
            merged_rois = torch.cat(merged_rois, dim=0)
        else:
            merged_rois = mlvl_rois.new_zeros((1, 7))
        return merged_rois


def build_merger(merger_cfg):
    return ROI_MERGER_REGISTRY.build(merger_cfg)


def build_roi_predictor(predictor_cfg):
    return ROI_PREDICTOR_REGISTRY.build(predictor_cfg)
