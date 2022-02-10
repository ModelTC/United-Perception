import torch
from eod.utils.general.registry_factory import FCOS_SUPERVISOR_REGISTRY
from eod.tasks.det.models.utils.matcher import build_matcher
from eod.tasks.det.plugins.condinst.models.postprocess.condinst_postprocess import add_bitmasks
from eod.utils.general.registry_factory import MATCHER_REGISTRY

__all__ = ['FcosSupervisor']


@FCOS_SUPERVISOR_REGISTRY.register('fcos')
class FcosSupervisor(object):
    """Compute targets for fcos head
    """

    def __init__(self, matcher, norm_on_bbox=False, mask_out_stride=1):
        self.matcher = build_matcher(matcher)
        self.norm_on_bbox = norm_on_bbox
        self.mask_out_stride = mask_out_stride

    def get_targets(self, locations, loc_ranges, input, has_mask=False):
        """
        Arguments:
            - locations (list of FloatTensor, fp32): {[k_i, 2]}
            - loc_ranges (list of pair): size range
            - input[:obj:`gt_bboxes`] (list of FloatTensor): [B, num_gts, 5], (x1, y1, x2, y2, label)
            - input[:obj:`strides`] (list of int): stride of each layer of FPN
            - input[:obj:`ignore_regions`] (list of FloatTensor): None or [B, num_igs, 4] (x1, y1, x2, y2)
        """
        if has_mask:
            image = input['image']
            gt_masks = input['gt_masks']
            gt_bitmasks, gt_bitmasks_full = add_bitmasks(gt_masks, image, self.mask_out_stride)

        gt_bboxes = input['gt_bboxes']  # [B, num_gts, 5]
        strides = input['strides']
        igs = input.get('gt_ignores', None)
        expanded_loc_ranges = []
        num_points_per = [len(p) for p in locations]
        for i in range(len(locations)):
            expanded_loc_ranges.append(locations[i].new_tensor(loc_ranges[i])[None].expand(len(locations[i]), -1))
        points = torch.cat(locations, dim=0)
        loc_ranges = torch.cat(expanded_loc_ranges, dim=0)

        K = points.size(0)
        B = len(gt_bboxes)
        gt_inds_target = points.new_full((B, K), -1, dtype=torch.int64)
        cls_target = points.new_full((B, K), -1, dtype=torch.int64)
        loc_target = points.new_zeros((B, K, 4))
        sample_cls_mask = points.new_zeros((B, K), dtype=torch.bool)
        sample_loc_mask = points.new_zeros((B, K), dtype=torch.bool)
        for b_ix in range(B):
            gt = gt_bboxes[b_ix]
            if gt.shape[0] > 0:
                ig = None
                if igs is not None:
                    ig = igs[b_ix]
                if has_mask:
                    gt_masks_per_im = gt_bitmasks_full[b_ix]
                    labels, bbox_targets, gt_inds = self.matcher.match(
                        points, gt, loc_ranges, num_points_per, strides, ig, gt_masks_per_im)
                else:
                    labels, bbox_targets, gt_inds = self.matcher.match(
                        points, gt, loc_ranges, num_points_per, strides, ig)
                cls_target[b_ix] = labels
                loc_target[b_ix] = bbox_targets
                gt_inds_target[b_ix] = gt_inds
                sample_cls_mask[b_ix] = labels != -1
                sample_loc_mask[b_ix] = labels > 0

                if self.norm_on_bbox:
                    loc_target_ix = list(loc_target[b_ix].split(num_points_per, 0))
                    for l_ix in range(len(num_points_per)):
                        loc_target_ix[l_ix] /= strides[l_ix]
                    loc_target[b_ix] = torch.cat(loc_target_ix)

        if has_mask:
            return cls_target, loc_target, sample_cls_mask, sample_loc_mask, gt_inds_target, \
                gt_bitmasks, gt_bitmasks_full
        return cls_target, loc_target, sample_cls_mask, sample_loc_mask


@MATCHER_REGISTRY.register('fcos')
class FcosMatcher(object):
    def __init__(self, center_sample=None, pos_radius=1):
        self.center_sample = center_sample
        self.pos_radius = pos_radius

    def match(self, points, gt, loc_ranges, num_points_per,
              strides=[8, 16, 32, 64, 128], ig=None, gt_masks_per_im=None):

        INF = 1e10
        num_gts = gt.shape[0]
        K = points.shape[0]
        gt_labels = gt[:, 4]
        xs, ys = points[:, 0], points[:, 1]
        gt_bboxes = gt[:, :4]
        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0] + 1) * (gt_bboxes[:, 3] - gt_bboxes[:, 1] + 1)

        areas = areas[None].repeat(K, 1)
        loc_ranges = loc_ranges[:, None, :].expand(K, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(K, num_gts, 4)
        gt_xs = xs[:, None].expand(K, num_gts)
        gt_ys = ys[:, None].expand(K, num_gts)

        left = gt_xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - gt_xs
        top = gt_ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - gt_ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        if self.center_sample:
            sample_mask = self.get_sample_region(
                gt_bboxes, strides, num_points_per, gt_xs, gt_ys, bitmasks=gt_masks_per_im)
        else:
            sample_mask = bbox_targets.min(-1)[0] > 0

        max_loc_distance = bbox_targets.max(-1)[0]
        inside_loc_range = (max_loc_distance >= loc_ranges[..., 0]) & (max_loc_distance <= loc_ranges[..., 1])

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[~sample_mask] = INF

        areas[inside_loc_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)
        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = 0
        bbox_targets = bbox_targets[range(K), min_area_inds]

        # ignore
        if ig is not None:
            num_igs = ig.shape[0]
            ig_xs = xs[:, None].expand(K, num_igs)
            ig_ys = ys[:, None].expand(K, num_igs)
            ig_left = ig_xs - ig[..., 0]
            ig_right = ig[..., 2] - ig_xs
            ig_top = ig_ys - ig[..., 1]
            ig_bottom = ig[..., 3] - ig_ys
            ig_targets = torch.stack((ig_left, ig_top, ig_right, ig_bottom), -1)
            ig_inside_gt_bbox_mask = (ig_targets.min(-1)[0] > 0).max(-1)[0]
            labels[ig_inside_gt_bbox_mask] = -1

        return labels, bbox_targets, min_area_inds

    def get_sample_region(self, gt, strides, num_points_per, gt_xs, gt_ys, bitmasks=None):
        if bitmasks is not None:
            _, h, w = bitmasks.size()

            ys = torch.arange(0, h, dtype=torch.float32, device=bitmasks.device)
            xs = torch.arange(0, w, dtype=torch.float32, device=bitmasks.device)

            m00 = bitmasks.sum(dim=-1).sum(dim=-1).clamp(min=1e-6)
            m10 = (bitmasks * xs).sum(dim=-1).sum(dim=-1)
            m01 = (bitmasks * ys[:, None]).sum(dim=-1).sum(dim=-1)
            center_x = m10 / m00
            center_y = m01 / m00
            K = gt.shape[0]
            num_gts = gt.shape[1]
            center_x = center_x[None].expand(K, num_gts)
            center_y = center_y[None].expand(K, num_gts)
        else:
            center_x = (gt[..., 0] + gt[..., 2]) / 2
            center_y = (gt[..., 1] + gt[..., 3]) / 2
        center_gt = gt.new_zeros(gt.shape)
        # no gt
        if center_x[..., 0].float().sum() == 0:
            return gt_xs.new_zeros(gt_xs.shape, dtype=torch.bool)
        beg = 0
        #   num_points_per = [len(p) for p in locations]
        for level, n_p in enumerate(num_points_per):
            end = beg + n_p
            stride = strides[level] * self.pos_radius
            xmin = center_x[beg:end] - stride
            ymin = center_y[beg:end] - stride
            xmax = center_x[beg:end] + stride
            ymax = center_y[beg:end] + stride
            # limit sample region in gt
            center_gt[beg:end, :, 0] = torch.where(xmin > gt[beg:end, :, 0], xmin, gt[beg:end, :, 0])
            center_gt[beg:end, :, 1] = torch.where(ymin > gt[beg:end, :, 1], ymin, gt[beg:end, :, 1])
            center_gt[beg:end, :, 2] = torch.where(xmax > gt[beg:end, :, 2], gt[beg:end, :, 2], xmax)
            center_gt[beg:end, :, 3] = torch.where(ymax > gt[beg:end, :, 3], gt[beg:end, :, 3], ymax)
            beg = end
        left = gt_xs - center_gt[..., 0]
        right = center_gt[..., 2] - gt_xs
        top = gt_ys - center_gt[..., 1]
        bottom = center_gt[..., 3] - gt_ys
        center_bbox = torch.stack((left, top, right, bottom), -1)
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        return inside_gt_bbox_mask


def build_fcos_supervisor(supervisor_cfg):
    return FCOS_SUPERVISOR_REGISTRY.build(supervisor_cfg)
