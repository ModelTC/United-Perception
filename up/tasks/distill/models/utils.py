import torch

from up.tasks.det.models.utils.assigner import map_rois_to_level
from up.tasks.det.models.utils.bbox_helper import (
    clip_bbox,
    filter_by_size
)


def mlvl_extract_roi_features(rois, x_features, fpn_levels,
                              fpn_strides, base_scale, roi_extractor,
                              return_recover_inds=False):
    x_rois, recover_inds = map_rois_to_level(fpn_levels, base_scale, rois)
    mlvl_rois = []
    mlvl_features = []
    mlvl_strides = []
    for lvl_idx in fpn_levels:
        if x_rois[lvl_idx].numel() > 0:
            mlvl_rois.append(x_rois[lvl_idx])
            mlvl_features.append(x_features[lvl_idx])
            mlvl_strides.append(fpn_strides[lvl_idx])
    assert len(mlvl_rois) > 0, "No rois provided for mimic stage"
    pooled_feats = [roi_extractor(*args) for args in zip(mlvl_rois, mlvl_features, mlvl_strides)]
    if return_recover_inds:
        return torch.cat(pooled_feats, dim=0), recover_inds
    else:
        return torch.cat(pooled_feats, dim=0)


def mlvl_extract_gt_masks(gt_bboxes, fpn_levels, fpn_strides, base_scale, featmap_sizes):
    gt_tensors = []
    for b_ix in range(len(gt_bboxes)):
        gt, _ = filter_by_size(gt_bboxes[b_ix], min_size=1)
        bdx = gt.new_ones(gt.shape[0], 1) * b_ix
        gt_tensors.append(torch.cat([bdx, gt[:, :4]], dim=1))
    gt_bboxes, _ = map_rois_to_level(fpn_levels, base_scale, torch.cat(gt_tensors, dim=0))

    imit_range = [0, 0, 0, 0, 0]
    with torch.no_grad():
        masks = []
        for idx in range(len(featmap_sizes)):
            b, _, h, w = featmap_sizes[idx]
            gt_level = gt_bboxes[idx]
            mask = gt_level.new_zeros(b, h, w)
            for gt in gt_level:
                gt_level_map = gt[1:] / fpn_strides[idx]
                lx = max(int(gt_level_map[0]) - imit_range[idx], 0)
                rx = min(int(gt_level_map[2]) + imit_range[idx], w)
                ly = max(int(gt_level_map[1]) - imit_range[idx], 0)
                ry = min(int(gt_level_map[3]) + imit_range[idx], h)
                if (lx == rx) or (ly == ry):
                    mask[int(gt[0]), ly, lx] += 1
                else:
                    mask[int(gt[0]), ly:ry, lx:rx] += 1
            mask = (mask > 0).type_as(gt_level)
            masks.append(mask)
    return masks


def match_gts(proposals, gt_bboxes, ignore_regions, image_info, matcher):
    B = len(gt_bboxes)
    if ignore_regions is None:
        ignore_regions = [None] * B
    labels = proposals.new_zeros(proposals.shape[0]).long()

    for b_ix in range(B):
        bind = torch.where(proposals[:, 0] == b_ix)[0]
        rois = proposals[bind]
        # rois = proposals[proposals[:, 0] == b_ix]
        if rois.numel() > 0:
            # remove batch idx, score & label
            rois = rois[:, 1:1 + 4]
        else:
            rois = rois.view(-1, 4)
        # filter gt_bboxes which are too small
        gt, _ = filter_by_size(gt_bboxes[b_ix], min_size=1)
        # clip rois which are out of bounds
        rois = clip_bbox(rois, image_info[b_ix])
        rois_target_gt, overlaps = matcher.match(
            rois, gt, ignore_regions[b_ix], return_max_overlaps=True)
        pos_inds = (rois_target_gt >= 0).long()
        labels[bind] = pos_inds
    return labels
