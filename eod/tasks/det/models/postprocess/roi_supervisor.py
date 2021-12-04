import torch
from eod.tasks.det.models.utils.box_sampler import build_roi_sampler
from eod.tasks.det.models.utils.matcher import build_matcher
from eod.utils.general.fp16_helper import to_float32
from eod.utils.general.registry_factory import ROI_SUPERVISOR_REGISTRY
from eod.tasks.det.models.utils.bbox_helper import (
    bbox2offset,
    bbox_iou_overlaps,
    filter_by_size,
    offset2bbox
)


__all__ = [
    'RetinaSupervisor',
    'RpnSupervisor',
    'ATSSSupervisor'
]


@ROI_SUPERVISOR_REGISTRY.register('retina')
class RetinaSupervisor(object):
    def __init__(self,
                 allowed_border,
                 matcher,
                 sampler,
                 bbox_encode=True,
                 return_pos_inds=False,
                 filter_outside_before_match=True):
        self.filter_outside_before_match = filter_outside_before_match
        self.border = allowed_border
        self.bbox_encode = bbox_encode
        self.return_pos_inds = return_pos_inds
        self.matcher = build_matcher(matcher)
        self.sampler = build_roi_sampler(sampler)

    @torch.no_grad()
    @to_float32
    def get_targets(self, mlvl_anchors, input, mlvl_preds=None):
        r"""Match anchors with gt bboxes and sample batch samples for training

        Arguments:
           - mlvl_anchors (:obj:`list` of :obj:`FloatTensor`, fp32): [[k_0, 4], ..., [k_n,4]],
             for layer :math:`i` in FPN, k_i = h_i * w_i * A
           - input (:obj:`dict`) with:
               - gt_bboxes (list of FloatTensor): [B, num_gts, 5] (x1, y1, x2, y2, label)
               - image_info (list of FloatTensor): [B, >=3] (image_h, image_w, scale_factor, ...)
               - ignore_regions (list of FloatTensor): None or [B, I, 4] (x1, y1, x2, y2)

        Returns:
            cls_target (LongTensor): [B, K], {-1, 0, 1} for RPN, {-1, 0, 1...C} for retinanet
            loc_target (FloatTensor): [B, K, 4]
            sample_cls_mask (ByteTensor): [B, K], binary mask, 1 for choosed samples, otherwise 0
            sample_loc_mask (ByteTensor): [B, K], binary mask, 1 for choosed positive samples, otherwise 0
        """

        gt_bboxes = input['gt_bboxes']
        ignore_regions = input.get('gt_ignores', None)
        image_info = input['image_info']
        all_anchors = torch.cat(mlvl_anchors, dim=0)
        B = len(gt_bboxes)
        K = all_anchors.shape[0]
        neg_targets = input.get('neg_targets', None)
        if ignore_regions is None:
            ignore_regions = [None] * B
        if neg_targets is None:
            neg_targets = [0] * B

        cls_target = all_anchors.new_full((B, K), -1, dtype=torch.int64)
        loc_target = all_anchors.new_zeros((B, K, 4))
        sample_cls_mask = all_anchors.new_zeros((B, K), dtype=torch.bool)
        sample_loc_mask = all_anchors.new_zeros((B, K), dtype=torch.bool)

        # aux targets
        pos_inds_list, gt_inds_list = [], []
        for b_ix in range(B):
            # filter gt bboxes which are too small
            gt, _ = filter_by_size(gt_bboxes[b_ix], min_size=1)

            # filter anchors which are out of bounds
            img_h, img_w = image_info[b_ix][:2]
            border = self.border
            if border >= 0:
                inside_mask = ((all_anchors[:, 0] > -border)
                               & (all_anchors[:, 1] > -border)
                               & (all_anchors[:, 2] < img_w + border)
                               & (all_anchors[:, 3] < img_h + border))
                # Just for aligning performance to maskrcnn-bencharm with caffe-style ResNet,
                # set to False will lead to ~0.2 mAP gain in coco dataset
                if self.filter_outside_before_match:
                    inside_anchors = all_anchors[inside_mask]
                    inside_inds = torch.nonzero(inside_mask).reshape(-1)
                else:
                    inside_inds = torch.arange(K, dtype=torch.int64, device=all_anchors.device)
                    inside_anchors = all_anchors
            else:
                inside_inds = torch.arange(K, dtype=torch.int64, device=all_anchors.device)
                inside_anchors = all_anchors

            anchor_target_gt, overlaps = self.matcher.match(
                inside_anchors, gt, ignore_regions[b_ix], return_max_overlaps=True)

            if not self.filter_outside_before_match:
                anchor_target_gt[~inside_mask] = self.matcher.IGNORE_TARGET

            pos_inds, neg_inds = self.sampler.sample(anchor_target_gt, overlaps=overlaps)

            # acquire target cls and loc for sampled anchors
            pos_anchors = inside_anchors[pos_inds]
            pos_target_gt = gt[anchor_target_gt[pos_inds]]
            if self.bbox_encode:
                pos_loc_target = bbox2offset(pos_anchors, pos_target_gt)
            else:
                pos_loc_target = pos_target_gt[:, :4]
            if pos_target_gt.numel() > 0:
                pos_cls_target = pos_target_gt[:, 4].to(dtype=torch.int64)
            else:
                pos_cls_target = pos_target_gt.new_zeros((0,), dtype=torch.int64)

            cls_target[b_ix, inside_inds[pos_inds]] = pos_cls_target
            loc_target[b_ix, inside_inds[pos_inds]] = pos_loc_target
            cls_target[b_ix, inside_inds[neg_inds]] = neg_targets[b_ix]

            sample_cls_mask[b_ix, inside_inds[pos_inds]] = True
            sample_cls_mask[b_ix, inside_inds[neg_inds]] = True
            sample_loc_mask[b_ix, inside_inds[pos_inds]] = True
            # aux inds
            pos_inds_list.append(inside_inds[pos_inds].view(-1))
            gt_inds_list.append(anchor_target_gt[pos_inds])

        if self.return_pos_inds:
            return cls_target, loc_target, sample_cls_mask, sample_loc_mask, pos_inds_list, gt_inds_list
        else:
            return cls_target, loc_target, sample_cls_mask, sample_loc_mask


@ROI_SUPERVISOR_REGISTRY.register('rpn')
class RpnSupervisor(RetinaSupervisor):
    """Compute targets for Region Proposal Network
    """

    @torch.no_grad()
    def get_targets(self, mlvl_anchors, input, mlvl_preds=None):
        targets = super(RpnSupervisor, self).get_targets(mlvl_anchors, input, mlvl_preds)
        # for naive rpn, only foreground and background classes
        cls_target = torch.clamp(targets[0], max=1)
        return (cls_target, *targets[1:])


@ROI_SUPERVISOR_REGISTRY.register('atss')
class ATSSSupervisor(object):
    '''
    Compuate targets for Adaptive Training Sample Selection
    '''

    def __init__(self, top_n=9, use_centerness=False, use_iou=False, gt_encode=True, return_gts=False):
        self.top_n = top_n
        self.use_centerness = use_centerness
        self.use_iou = use_iou
        self.gt_encode = gt_encode
        self.return_gts = return_gts

    def compuate_centerness_targets(self, loc_target, anchors):
        gts = offset2bbox(anchors, loc_target)
        anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
        anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2
        _l = anchors_cx - gts[:, 0]
        t = anchors_cy - gts[:, 1]
        r = gts[:, 2] - anchors_cx
        b = gts[:, 3] - anchors_cy
        left_right = torch.stack([_l, r], dim=1)
        top_bottom = torch.stack([t, b], dim=1)
        centerness_target = torch.sqrt((left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
                top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0]))  # noqa E501
        assert not torch.isnan(centerness_target).any()
        return centerness_target

    def compuate_iou_targets(self, loc_target, anchors, loc_pred):
        decode_loc_pred = offset2bbox(anchors, loc_pred)
        decode_loc_target = offset2bbox(anchors, loc_target)
        iou_target = bbox_iou_overlaps(decode_loc_pred, decode_loc_target, aligned=True)
        return iou_target

    @torch.no_grad()
    @to_float32
    def get_targets(self, mlvl_anchors, input, mlvl_preds=None):
        r"""Match anchors with gt bboxes and sample batch samples for training

        Arguments:
           - mlvl_anchors (:obj:`list` of :obj:`FloatTensor`, fp32): [[k_0, 4], ..., [k_n,4]],
             for layer :math:`i` in FPN, k_i = h_i * w_i * A
           - input (:obj:`dict`) with:
               - gt_bboxes (list of FloatTensor): [B, num_gts, 5] (x1, y1, x2, y2, label)
               - image_info (list of FloatTensor): [B, >=3] (image_h, image_w, scale_factor, ...)
               - ignore_regions (list of FloatTensor): None or [B, I, 4] (x1, y1, x2, y2)

        Returns:
            cls_target (LongTensor): [B, K], {-1, 0, 1} for RPN, {-1, 0, 1...C} for retinanet
            loc_target (FloatTensor): [B, K, 4]
            centerness_target (FloatTensor): [B, K], only for use_centerness
            sample_cls_mask (ByteTensor): [B, K], binary mask, 1 for choosed samples, otherwise 0
            sample_loc_mask (ByteTensor): [B, K], binary mask, 1 for choosed positive samples, otherwise 0
        """
        INF = 100000000
        gt_bboxes = input['gt_bboxes']
        ignore_regions = input.get('gt_ignores', None)
        image_info = input.get('image_info')
        num_anchors_per_level = [len(anchor) for anchor in mlvl_anchors]
        all_anchors = torch.cat(mlvl_anchors, dim=0)
        B = len(gt_bboxes)
        K = all_anchors.shape[0]
        neg_targets = input.get('neg_targets', None)
        if ignore_regions is None:
            ignore_regions = [None] * B
        if neg_targets is None:
            neg_targets = [0] * B
        cls_target = all_anchors.new_full((B, K), 0, dtype=torch.int64)
        loc_target = all_anchors.new_zeros((B, K, 4))
        sample_cls_mask = all_anchors.new_zeros((B, K), dtype=torch.bool)
        sample_loc_mask = all_anchors.new_zeros((B, K), dtype=torch.bool)

        anchors_cx = (all_anchors[:, 2] + all_anchors[:, 0]) / 2.0
        anchors_cy = (all_anchors[:, 3] + all_anchors[:, 1]) / 2.0
        anchor_num = anchors_cx.shape[0]
        anchor_points = torch.stack((anchors_cx, anchors_cy), dim=1)
        if self.use_iou:
            mlvl_loc_pred = list(zip(*mlvl_preds))[1]
            loc_pred = torch.cat(mlvl_loc_pred, dim=1)
        for b_ix in range(B):
            gt, _ = filter_by_size(gt_bboxes[b_ix], min_size=1)
            num_gt = gt.shape[0]
            if gt.shape[0] == 0:
                cls_target[b_ix][:] = neg_targets[b_ix]
                continue
            else:
                bbox = gt[:, :4]
                labels = gt[:, 4]
                ious = bbox_iou_overlaps(all_anchors, gt)
                gt_cx = (bbox[:, 2] + bbox[:, 0]) / 2.0
                gt_cy = (bbox[:, 3] + bbox[:, 1]) / 2.0
                gt_cx = torch.clamp(gt_cx, min=0, max=image_info[b_ix][1])
                gt_cy = torch.clamp(gt_cy, min=0, max=image_info[b_ix][0])
                gt_points = torch.stack((gt_cx, gt_cy), dim=1)
                distances = (anchor_points[:, None, :] - gt_points[None, :, :]).pow(2).sum(-1).sqrt()
                # Selecting candidates based on the center distance between anchor box and object
                candidate_idxs = []
                star_idx = 0
                for num in num_anchors_per_level:
                    end_idx = star_idx + num
                    distances_per_level = distances[star_idx:end_idx]
                    topk = min(self.top_n, num)
                    _, topk_idxs_per_level = distances_per_level.topk(topk, dim=0, largest=False)
                    candidate_idxs.append(topk_idxs_per_level + star_idx)
                    star_idx = end_idx
                candidate_idxs = torch.cat(candidate_idxs, dim=0)

                # Using the sum of mean and standard deviation as the IoU threshold to select final positive samples
                candidate_ious = ious[candidate_idxs, torch.arange(num_gt)]
                iou_mean_per_gt = candidate_ious.mean(0)
                iou_std_per_gt = candidate_ious.std(0)
                iou_thresh_per_gt = iou_mean_per_gt + iou_std_per_gt
                is_pos = candidate_ious >= iou_thresh_per_gt[None, :]

                # Limiting the final positive samples’ center to object
                for ng in range(num_gt):
                    candidate_idxs[:, ng] += ng * anchor_num
                e_anchors_cx = anchors_cx.view(1, -1).expand(num_gt, anchor_num).contiguous().view(-1)
                e_anchors_cy = anchors_cy.view(1, -1).expand(num_gt, anchor_num).contiguous().view(-1)
                candidate_idxs = candidate_idxs.view(-1)
                _l = e_anchors_cx[candidate_idxs].view(-1, num_gt) - bbox[:, 0]
                t = e_anchors_cy[candidate_idxs].view(-1, num_gt) - bbox[:, 1]
                r = bbox[:, 2] - e_anchors_cx[candidate_idxs].view(-1, num_gt)
                b = bbox[:, 3] - e_anchors_cy[candidate_idxs].view(-1, num_gt)
                is_in_gts = torch.stack([_l, t, r, b], dim=1).min(dim=1)[0] > 0.01
                is_pos = is_pos & is_in_gts

                # if an anchor box is assigned to multiple gts, the one with the highest IoU will be selected.
                ious_inf = torch.full_like(ious, -INF).t().contiguous().view(-1)
                index = candidate_idxs.view(-1)[is_pos.view(-1)]
                ious_inf[index] = ious.t().contiguous().view(-1)[index]
                ious_inf = ious_inf.view(num_gt, -1).t()

                anchors_to_gt_values, anchors_to_gt_indexs = ious_inf.max(dim=1)

                labels = labels[anchors_to_gt_indexs]
                neg_index = anchors_to_gt_values == -INF
                pos_index = ~neg_index
                labels[anchors_to_gt_values == -INF] = neg_targets[b_ix]
                matched_gts = bbox[anchors_to_gt_indexs]
                if self.gt_encode:
                    reg_targets_per_im = bbox2offset(all_anchors, matched_gts)
                else:
                    reg_targets_per_im = matched_gts
                cls_target[b_ix] = labels
                loc_target[b_ix] = reg_targets_per_im
                sample_cls_mask[b_ix] = pos_index
                sample_loc_mask[b_ix] = pos_index

            if ignore_regions[b_ix] is not None and ignore_regions[b_ix].shape[0] > 0:
                ig_bbox = ignore_regions[b_ix]
                if ig_bbox.sum() > 0:
                    ig_left = anchors_cx[:, None] - ig_bbox[..., 0]
                    ig_right = ig_bbox[..., 2] - anchors_cx[:, None]
                    ig_top = anchors_cy[:, None] - ig_bbox[..., 1]
                    ig_bottom = ig_bbox[..., 3] - anchors_cy[:, None]
                    ig_targets = torch.stack((ig_left, ig_top, ig_right, ig_bottom), -1)
                    ig_inside_bbox_mask = (ig_targets.min(-1)[0] > 0).max(-1)[0]
                    cls_target[b_ix][ig_inside_bbox_mask] = -1
                    sample_cls_mask[b_ix][ig_inside_bbox_mask] = False
                    sample_loc_mask[b_ix][ig_inside_bbox_mask] = False
        if self.use_centerness or self.use_iou:
            batch_anchor = all_anchors.view(1, -1, 4).expand((sample_loc_mask.shape[0], sample_loc_mask.shape[1], 4))
            sample_anchor = batch_anchor[sample_loc_mask].contiguous().view(-1, 4)
            sample_loc_target = loc_target[sample_loc_mask].contiguous().view(-1, 4)
            if sample_loc_target.numel():
                if self.use_iou:
                    centerness_target = self.compuate_iou_targets(sample_loc_target, sample_anchor, loc_pred[sample_loc_mask].view(-1, 4))  # noqa
                else:
                    centerness_target = self.compuate_centerness_targets(
                        sample_loc_target, sample_anchor)
            else:
                centerness_target = sample_loc_target.new_zeros(sample_loc_target.shape[0])
            return cls_target, loc_target, centerness_target, sample_cls_mask, sample_loc_mask

        if self.return_gts:
            return cls_target, loc_target, sample_cls_mask, sample_loc_mask, gt_bboxes
        return cls_target, loc_target, sample_cls_mask, sample_loc_mask


def build_roi_supervisor(supervisor_cfg):
    return ROI_SUPERVISOR_REGISTRY.build(supervisor_cfg)
