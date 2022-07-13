# Standard Library
import copy

# Import from third library
import torch

# Import from pod
from up.tasks.det.models.postprocess.bbox_predictor import BboxPredictor
from up.tasks.det.models.utils.bbox_helper import (
    bbox2offset,
    clip_bbox,
    filter_by_size,
    normalize_offset,
    offset2bbox,
    unnormalize_offset
)
from up.tasks.det.models.utils.box_sampler import build_roi_sampler
from up.tasks.det.models.utils.matcher import build_matcher
from up.utils.general.fp16_helper import to_float32
from up.utils.general.log_helper import default_logger as logger
from up.utils.general.registry_factory import BBOX_PREDICTOR_REGISTRY, BBOX_SUPERVISOR_REGISTRY

__all__ = ['CascadeSupervisor', 'CascadePredictor']


class SingleSupervisor(object):
    """Compute targets for single stage, and return with gt_flags
    for futher process.
    """

    def __init__(self, bbox_normalize, matcher, sampler):
        self.offset_means = bbox_normalize['means']
        self.offset_stds = bbox_normalize['stds']
        self.matcher = build_matcher(matcher)
        self.sampler = build_roi_sampler(sampler)

    @torch.no_grad()
    @to_float32
    def get_targets(self, proposals, input):
        """
        Arguments:
            - proposals (FloatTensor, fp32): [N, >=5], (batch_idx, x1, y1, x2, y2, ...)
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
        gt_bboxes = input['gt_bboxes']
        ignore_regions = input.get('ignore_regions', None)
        image_info = input['image_info']

        T = proposals
        B = len(gt_bboxes)
        if ignore_regions is None:
            ignore_regions = [None] * B

        batch_rois = []
        batch_gt_flags = []
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
            gt_flags = rois.new_zeros((rois.shape[0], ), dtype=torch.bool)
            if gt.numel() > 0:
                rois = torch.cat([rois, gt[:, :4]], dim=0)
                gt_ones = rois.new_ones(gt.shape[0], dtype=torch.bool)
                gt_flags = torch.cat([gt_flags, gt_ones])

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
            sample_record = (pos_inds, rois_target_gt[pos_inds])
            batch_sample_record[b_ix] = sample_record

            # acquire target cls and target loc for sampled rois
            pos_rois = rois[pos_inds]
            neg_rois = rois[neg_inds]
            pos_target_gt = gt[rois_target_gt[pos_inds]]
            gt_flags = gt_flags[pos_inds]
            if pos_target_gt.numel() > 0:
                pos_cls_target = pos_target_gt[:, 4].to(dtype=torch.int64)
            else:
                # give en empty tensor if no positive
                pos_cls_target = T.new_zeros((0, ), dtype=torch.int64)
            neg_cls_target = T.new_zeros((N, ), dtype=torch.int64)

            offset = bbox2offset(pos_rois, pos_target_gt)
            pos_loc_target = normalize_offset(offset, self.offset_means, self.offset_stds)

            rois = torch.cat([pos_rois, neg_rois], dim=0)
            # for cascade rcnn, add by buxingyuan
            gt_zeros = rois.new_zeros(neg_rois.shape[0], dtype=torch.bool)
            gt_flags = torch.cat([gt_flags, gt_zeros])
            # end
            b = T.new_full((rois.shape[0], 1), b_ix)
            rois = torch.cat([b, rois], dim=1)
            batch_rois += [rois]
            batch_gt_flags += [gt_flags]
            batch_cls_target += [pos_cls_target, neg_cls_target]
            batch_loc_target += [pos_loc_target, T.new_zeros(N, 4)]

            loc_weight = torch.cat([T.new_ones(P, 4), T.new_zeros(N, 4)])
            batch_loc_weight.append(loc_weight)

        if len(batch_rois) == 0:
            num_rois = 1
            rois = T.new_zeros((num_rois, 5))
            gt_flags = T.new_zeros(num_rois).to(dtype=torch.bool)
            cls_target = T.new_zeros(num_rois).long() - 1  # target cls must be `long` type
            loc_target = T.new_zeros((num_rois, 4))
            loc_weight = T.new_zeros((num_rois, 4))
            logger.warning('no valid proposals, set cls_target to {}'.format(cls_target))
        else:
            rois = torch.cat(batch_rois, dim=0)
            gt_flags = torch.cat(batch_gt_flags, dim=0)
            cls_target = torch.cat(batch_cls_target, dim=0).long()
            loc_target = torch.cat(batch_loc_target, dim=0)
            loc_weight = torch.cat(batch_loc_weight, dim=0)

        return batch_sample_record, rois, cls_target, loc_target, loc_weight, gt_flags


@BBOX_SUPERVISOR_REGISTRY.register('cascade')
class CascadeSupervisor(object):
    """Compute targets for cascade heads
    """

    def __init__(self, num_stage, stage_bbox_normalize, stage_matcher, sampler):
        """
        Arguments:
            - num_stage (:obj:`int`): number of stages
            - stage_bbox_normalize (:obj:`dict`): with :obj:`means` and :obj:`stds` of list of num_stage
            - stage_matcher (:obj:`dict`): config of matcher with multiple iou thresholds
            - sampler (:obj:`dict`): config of bbox sampler
        """
        self.supervisor_list = []
        for i in range(num_stage):
            stage_config = self.get_stage_config(i, stage_bbox_normalize, stage_matcher, sampler)
            supervisor = SingleSupervisor(*stage_config)
            self.supervisor_list.append(supervisor)

    def get_stage_config(self, stage, stage_bbox_normalize, stage_matcher, sampler):
        bbox_normalize = {
            'means': stage_bbox_normalize['means'][stage],
            'stds': stage_bbox_normalize['stds'][stage]
        }
        matcher = copy.deepcopy(stage_matcher)
        kwargs = matcher['kwargs']
        for k in ['positive_iou_thresh', 'negative_iou_thresh']:
            kwargs[k] = kwargs[k][stage]
        return bbox_normalize, matcher, sampler

    def get_targets(self, stage, proposals, input):
        return self.supervisor_list[stage].get_targets(proposals, input)


@BBOX_PREDICTOR_REGISTRY.register('cascade')
class CascadePredictor(object):
    """Unified class for refinement and prediction
    """

    def __init__(self, num_stage, stage_bbox_normalize, bbox_score_thresh, top_n, share_location,
                 nms, bbox_vote=None):
        """
        Arguments:
            - num_stage (:obj:`int`): number of stages
            - stage_bbox_normalize (:obj:`dict`): with :obj:`means` and :obj:`stds` of list of num_stage
            - bbox_score_thresh (:obj:`float`): minimum score to keep
            - share_location (:obj:`bool`): class agnostic regression if True
            - nms (:obj:`dict`): config for nms
            - bbox_vote (:obj:`dict`): config for bbox vote
        """

        bbox_normalize = self.get_stage_config(num_stage - 1, stage_bbox_normalize)
        self.final_predictor = BboxPredictor(
            bbox_normalize, bbox_score_thresh, top_n, share_location, nms, bbox_vote)
        self.refiner_list = []
        for i in range(num_stage - 1):
            bbox_normalize = self.get_stage_config(i, stage_bbox_normalize)
            self.refiner_list.append(Refiner(bbox_normalize, share_location))

    def get_stage_config(self, stage, stage_bbox_normalize):
        bbox_normalize = {
            'means': stage_bbox_normalize['means'][stage],
            'stds': stage_bbox_normalize['stds'][stage]
        }
        return bbox_normalize

    def refine(self, i, *args, **kwargs):
        return self.refiner_list[i].refine(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.final_predictor.predict(*args, **kwargs)


class Refiner(object):
    """Regression only with classification
    """

    def __init__(self, bbox_normalize, share_location):
        self.offset_means = bbox_normalize['means']
        self.offset_stds = bbox_normalize['stds']
        self.share_location = share_location

    def refine(self, rois, labels, loc_pred, image_info, gt_flags=None):
        """
        Something likes predict_bboxes, but this is used in cascade rcnn
        Arguments:
            - rois (FloatTensor, fp32): [N, >=5] (batch_ix, x1, y1, x2, y2, ...)
            - roi_labels (FloatTensor, fp32): [N]
            - loc_pred (FloatTensor, fp32): [N, C*4] or [N, 4]
            - image_info (list of FloatTensor): [B, >=3] (image_h, image_w, scale_factor, ...)
            - gt_flags (list of bool): whether the rois is ground truth bboxes

        Returns:
            bboxes (FloatTensor): [R, 5], (batch_ix, x1, y1, x2, y2)
        """
        B = len(image_info)

        # double check loc_pred shape because loc_pred is reduced to [N, 4] in get loss func
        if self.share_location or loc_pred.shape[1] == 4:
            offset = loc_pred
        else:
            inds = labels * 4
            inds = torch.stack((inds, inds + 1, inds + 2, inds + 3), 1)
            offset = torch.gather(loc_pred, 1, inds)
        assert offset.size(1) == 4

        offset = unnormalize_offset(offset, self.offset_means, self.offset_stds)
        bboxes = offset2bbox(rois[:, 1:1 + 4], offset)

        detected_bboxes = []
        for b_ix in range(B):
            rois_ix = torch.nonzero(rois[:, 0] == b_ix).reshape(-1)
            if rois_ix.numel() == 0:
                continue
            pre_bboxes = bboxes[rois_ix]

            # clip bboxes which are out of bounds
            pre_bboxes[:, :4] = clip_bbox(pre_bboxes[:, :4], image_info[b_ix])

            ix = pre_bboxes.new_full((pre_bboxes.shape[0], 1), b_ix)
            post_bboxes = torch.cat([ix, pre_bboxes], dim=1)
            detected_bboxes.append(post_bboxes)
        detected_bboxes = torch.cat(detected_bboxes, dim=0)

        if gt_flags is not None:
            # filter gt bboxes
            assert gt_flags.dtype == torch.bool, gt_flags.dtype
            pos_keep = ~gt_flags
            keep_inds = torch.nonzero(pos_keep).reshape(-1)
            new_rois = detected_bboxes[keep_inds]
            return new_rois
        else:
            return detected_bboxes
