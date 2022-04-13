import torch

from up.tasks.det.models.utils.bbox_helper import (
    bbox2offset,
    clip_bbox,
    filter_by_size,
    normalize_offset,
)
from up.tasks.det.models.utils.box_sampler import build_roi_sampler
from up.tasks.det.models.utils.matcher import build_matcher
from up.utils.general.fp16_helper import to_float32
from up.utils.general.log_helper import default_logger as logger
from up.utils.general.registry_factory import BBOX_SUPERVISOR_REGISTRY

__all__ = ['BboxSupervisor']


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


def build_bbox_supervisor(supervisor_cfg):
    return BBOX_SUPERVISOR_REGISTRY.build(supervisor_cfg)
