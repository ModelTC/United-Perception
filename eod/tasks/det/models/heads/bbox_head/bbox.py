# import copy
import torch
# import torch.nn as nn
# import torch.nn.functional as F


# from eod.extensions import nms
from eod.tasks.det.models.utils.bbox_helper import (
    bbox2offset,
    clip_bbox,
    filter_by_size,
    normalize_offset,
    offset2bbox,
    box_voting,
    unnormalize_offset,
)
from eod.tasks.det.models.utils.box_sampler import build_roi_sampler
from eod.tasks.det.models.utils.matcher import build_matcher
from eod.utils.general.fp16_helper import to_float32
from eod.utils.general.log_helper import default_logger as logger
from eod.extensions import nms
# from eod.tasks.det.models.utils.nms_wrapper import nms
from eod.utils.general.registry_factory import BBOX_SUPERVISOR_REGISTRY, BBOX_PREDICTOR_REGISTRY
from eod.utils.general.tocaffe_utils import ToCaffe

__all__ = ['BboxSupervisor', 'BboxPredictor']


@BBOX_SUPERVISOR_REGISTRY.register('faster')
class BboxSupervisor(object):
    """Compute the second stage targets for faster-rcnn methods
    """

    def __init__(self, bbox_normalize, matcher, sampler):
        """
        Arguments:
            - bbox_normalize: :obj:`dict` with :obj:`means` and :obj:`stds` for regression
            - matcher: :obj:`dict` of config used to build box matcher
            - sampler: :obj:`dict` of config used to build box sampler
        """
        self.offset_means = bbox_normalize['means']
        self.offset_stds = bbox_normalize['stds']
        self.matcher = build_matcher(matcher)
        self.sampler = build_roi_sampler(sampler)

    @torch.no_grad()
    @to_float32
    def get_targets(self, input):
        """
        Arguments:
            - input[':obj:`dt_bboxes`] (FloatTensor, fp32): [N, >=5], (batch_idx, x1, y1, x2, y2, ...)
            - input[:obj:`gt_bboxes`] (list of FloatTensor): [B, num_gts, 5], (x1, y1, x2, y2, label)
            - input[:obj:`image_info`] (list of FloatTensor): [B, >=3] (image_h, image_w, scale_factor, ...)
            - input[:obj:`ignore_regions`] (list of FloatTensor): None or [B, num_igs, 4] (x1, y1, x2, y2)

        Returns:
            - batch_sample_record (list of tuple): [B, (pos_inds, pos_target_gt_inds)]
            - rois (FloatTensor): [R, 5] (batch_idx, x1, y1, x2, y2), sampled rois
            - cls_target (LongTensor): [R]
            - loc_target (FloatTensor): [R, 4]
            - loc_weight (FloatTensor): [R, 4], 1 for positive, otherwise 0
        """
        proposals = input['dt_bboxes']
        gt_bboxes = input['gt_bboxes']
        ignore_regions = input.get('gt_ignores', None)
        image_info = input['image_info']

        T = proposals
        B = len(gt_bboxes)
        if ignore_regions is None:
            ignore_regions = [None] * B

        batch_rois = []
        batch_cls_target = []
        batch_loc_target = []
        batch_loc_weight = []
        batch_sample_record = [None] * B
        for b_ix in range(B):
            rois = proposals[proposals[:, 0] == b_ix]
            if rois.numel() > 0:
                # remove batch idx, score & label
                rois = rois[:, 1:1 + 4]
            else:
                rois = rois.view(-1, 4)

            # filter gt_bboxes which are too small
            gt, _ = filter_by_size(gt_bboxes[b_ix], min_size=1)

            # add gts into rois
            if gt.numel() > 0:
                rois = torch.cat([rois, gt[:, :4]], dim=0)

            # clip rois which are out of bounds
            rois = clip_bbox(rois, image_info[b_ix])

            rois_target_gt, overlaps = self.matcher.match(
                rois, gt, ignore_regions[b_ix], return_max_overlaps=True)
            pos_inds, neg_inds = self.sampler.sample(rois_target_gt, overlaps=overlaps)
            P = pos_inds.numel()
            N = neg_inds.numel()
            if P + N == 0:
                continue  # noqa E701

            # save pos inds and related gts' inds for mask/keypoint head
            # sample_record = (pos_inds, rois_target_gt[pos_inds])
            # batch_sample_record[b_ix] = sample_record

            # acquire target cls and target loc for sampled rois
            pos_rois = rois[pos_inds]
            neg_rois = rois[neg_inds]
            pos_target_gt = gt[rois_target_gt[pos_inds]]
            if pos_target_gt.numel() > 0:
                pos_cls_target = pos_target_gt[:, 4].to(dtype=torch.int64)
            else:
                # give en empty tensor if no positive
                pos_cls_target = T.new_zeros((0,), dtype=torch.int64)
            sample_record = (pos_inds, rois_target_gt[pos_inds], pos_rois, pos_cls_target)
            batch_sample_record[b_ix] = sample_record
            neg_cls_target = T.new_zeros((N,), dtype=torch.int64)

            offset = bbox2offset(pos_rois, pos_target_gt)
            pos_loc_target = normalize_offset(offset, self.offset_means, self.offset_stds)

            rois = torch.cat([pos_rois, neg_rois], dim=0)
            b = T.new_full((rois.shape[0], 1), b_ix)
            rois = torch.cat([b, rois], dim=1)
            batch_rois += [rois]
            batch_cls_target += [pos_cls_target, neg_cls_target]
            batch_loc_target += [pos_loc_target, T.new_zeros(N, 4)]

            loc_weight = torch.cat([T.new_ones(P, 4), T.new_zeros(N, 4)])
            batch_loc_weight.append(loc_weight)

        if len(batch_rois) == 0:
            num_rois = 1
            rois = T.new_zeros((num_rois, 5))
            cls_target = T.new_zeros(num_rois).long() - 1  # target cls must be `long` type
            loc_target = T.new_zeros((num_rois, 4))
            loc_weight = T.new_zeros((num_rois, 4))
            logger.warning('no valid proposals, set cls_target to {}'.format(cls_target))
        else:
            rois = torch.cat(batch_rois, dim=0)
            cls_target = torch.cat(batch_cls_target, dim=0).long()
            loc_target = torch.cat(batch_loc_target, dim=0)
            loc_weight = torch.cat(batch_loc_weight, dim=0)

        return batch_sample_record, rois, cls_target, loc_target, loc_weight


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

    @ToCaffe.disable_trace
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


def build_bbox_supervisor(supervisor_cfg):
    return BBOX_SUPERVISOR_REGISTRY.build(supervisor_cfg)


def build_bbox_predictor(predictor_cfg):
    return BBOX_PREDICTOR_REGISTRY.build(predictor_cfg)
