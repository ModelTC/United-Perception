# Import from third library
import torch
import torch.nn.functional as F
from eod.tasks.det.models.utils.matcher import build_matcher
from eod.utils.general.registry_factory import MATCHER_REGISTRY, ROI_SUPERVISOR_REGISTRY
from eod.tasks.det.models.utils.bbox_helper import bbox_iou_overlaps
from eod.utils.general.log_helper import default_logger as logger


__all__ = ['OTASupervisor', 'OTAMatcher']


@ROI_SUPERVISOR_REGISTRY.register('ota')
class OTASupervisor(object):
    def __init__(self, num_classes, matcher, norm_on_bbox=False, return_gts=False):
        self.matcher = build_matcher(matcher)
        self.norm_on_bbox = norm_on_bbox
        self.return_gts = return_gts
        self.center_sample = True
        self.num_classes = num_classes - 1  # 80

    def get_l1_target(self, gts_xyxy, points_stride, strides, eps=1e-8):
        gts = gts_xyxy.clone()
        gts[:, 0] = (gts_xyxy[:, 0] + gts_xyxy[:, 2]) / 2
        gts[:, 1] = (gts_xyxy[:, 1] + gts_xyxy[:, 3]) / 2
        gts[:, 2] = gts_xyxy[:, 2] - gts_xyxy[:, 0]
        gts[:, 3] = gts_xyxy[:, 3] - gts_xyxy[:, 1]
        points = points_stride.clone() / strides.unsqueeze(-1)
        x_shifts = points[:, 0]
        y_shifts = points[:, 1]
        l1_target = gts.new_zeros((gts.shape[0], 4))
        l1_target[:, 0] = gts[:, 0] / strides - x_shifts
        l1_target[:, 1] = gts[:, 1] / strides - y_shifts
        l1_target[:, 2] = torch.log(gts[:, 2] / strides + eps)
        l1_target[:, 3] = torch.log(gts[:, 3] / strides + eps)
        return l1_target

    @torch.no_grad()
    def get_targets(self, locations, input, mlvl_preds):
        cls_targets = []
        reg_targets = []
        obj_targets = []
        ori_reg_targets = []
        fg_masks = []

        # process mlvl_preds: --> [B, A, 85]
        mlvl_preds = [torch.cat(preds, dim=-1) for preds in mlvl_preds]
        mlvl_preds = torch.cat(mlvl_preds, dim=1)  # [B, A, 85]

        gt_bboxes = input['gt_bboxes']
        strides = input['strides']
        # center points
        num_points_per_level = [len(p) for p in locations]
        points = torch.cat(locations, dim=0)  # [A, 2]

        # batch size
        B = len(gt_bboxes)

        num_gts = 0.0
        num_fgs = 0.0
        for b_ix in range(B):
            gts = gt_bboxes[b_ix]    # [G, 5]
            num_gts += len(gts)
            preds = mlvl_preds[b_ix]  # [A, 85]
            if gts.shape[0] > 0:
                try:
                    img_size = input['image_info'][b_ix][:2]
                    num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds, fg_mask, expanded_strides = \
                        self.matcher.match(gts, preds, points, num_points_per_level, strides, img_size=img_size)
                except RuntimeError:
                    logger.info(
                        "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                           CPU mode is applied in this batch. If you want to avoid this issue, \
                           try to reduce the batch size or image size."
                    )
                    torch.cuda.empty_cache()
                    num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds, fg_mask, expanded_strides = \
                        self.matcher.match(gts, preds, points, num_points_per_level, strides, mode='cpu')
                torch.cuda.empty_cache()
                if num_fg == 0:
                    fg_mask = preds.new_zeros(preds.shape[0], dtype=torch.bool)
                    cls_target = preds.new_zeros((0, self.num_classes))
                    reg_target = preds.new_zeros((0, 4))
                    obj_target = preds.new_zeros((fg_mask.shape[0], 1)).to(torch.bool)
                    l1_target = preds.new_zeros((0, 4))
                else:
                    cls_target = F.one_hot(
                        (gt_matched_classes - 1).to(torch.int64), self.num_classes
                    ) * pred_ious_this_matching.unsqueeze(-1)
                    reg_target = gts[matched_gt_inds, :4]
                    obj_target = fg_mask.unsqueeze(-1)
                    l1_target = self.get_l1_target(gts[matched_gt_inds, :4], points[fg_mask], expanded_strides[0][fg_mask])  # noqa
                num_fgs += num_fg
            else:
                fg_mask = preds.new_zeros(preds.shape[0], dtype=torch.bool)
                cls_target = preds.new_zeros((0, self.num_classes))
                reg_target = preds.new_zeros((0, 4))
                obj_target = preds.new_zeros((fg_mask.shape[0], 1)).to(torch.bool)
                l1_target = preds.new_zeros((0, 4))

            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target)
            ori_reg_targets.append(l1_target)
            fg_masks.append(fg_mask)

        return cls_targets, reg_targets, obj_targets, ori_reg_targets, fg_masks, num_fgs


@MATCHER_REGISTRY.register('ota')
class OTAMatcher(object):
    def __init__(self, num_classes, center_sample=True, pos_radius=1, candidate_k=10, radius=2.5):
        self.pos_radius = pos_radius
        self.center_sample = center_sample
        self.num_classes = num_classes - 1  # 80
        self.candidate_k = candidate_k
        self.radius = radius

    @torch.no_grad()
    def match(self, gts, preds, points, num_points_per_level, strides, mode='cuda', img_size=[]):
        ''' points: [A, 2] || gts: [G, 5] || preds: [A, 85]
        num_points_per_level:  [15808, 3952, 988, 247, 70]
        strides: [8, 16, 32, 64, 128]'''
        G = len(gts)
        if mode == 'cpu':
            logger.info('------------CPU Mode for This Batch-------------')
            gts = gts.cpu()
            points = points.cpu()

        gt_labels = gts[:, 4]   # [G, 1]
        gt_bboxes = gts[:, :4]  # [G, 4]

        if self.center_sample:
            fg_mask, is_in_boxes_and_center, expanded_strides = self.get_sample_region(
                gt_bboxes, strides, points, num_points_per_level, img_size)
        else:
            assert "Not implement"

        if fg_mask.sum() <= 1e-6:
            logger.info('no gt in center')
            return 0, None, None, None, None, None

        masked_preds = preds[fg_mask]
        if mode == 'cpu':
            masked_preds = masked_preds.cpu()

        preds_cls = masked_preds[:, :self.num_classes]
        preds_box = masked_preds[:, self.num_classes:-1]  # [A', 4]
        preds_obj = masked_preds[:, -1:]

        def decode_ota_box(preds):
            x1 = preds[:, 0] - preds[:, 2] / 2
            y1 = preds[:, 1] - preds[:, 3] / 2
            x2 = preds[:, 0] + preds[:, 2] / 2
            y2 = preds[:, 1] + preds[:, 3] / 2
            return torch.stack([x1, y1, x2, y2], -1)

        with torch.cuda.amp.autocast(enabled=False):
            decoded_preds_box = decode_ota_box(preds_box.float())

            pair_wise_ious = bbox_iou_overlaps(
                gt_bboxes, decoded_preds_box)
            gt_labels_ = gt_labels - 1
            gt_cls_per_image = (
                F.one_hot(gt_labels_.to(torch.int64), self.num_classes).float()
                .unsqueeze(1).repeat(1, masked_preds.shape[0], 1)
            )
            pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)
            cls_preds_ = (
                preds_cls.float().unsqueeze(0).repeat(G, 1, 1).sigmoid_()
                * preds_obj.float().unsqueeze(0).repeat(G, 1, 1).sigmoid_()
            )
            pair_wise_cls_loss = F.binary_cross_entropy(
                cls_preds_.sqrt_(),
                gt_cls_per_image,
                reduction="none").sum(-1)
        del cls_preds_

        cost = (
            pair_wise_cls_loss
            + 3.0 * pair_wise_ious_loss
            + 100000.0 * (~is_in_boxes_and_center)
        )

        (num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds
         ) = self.dynamic_k_matching(cost, pair_wise_ious, gt_labels, G, fg_mask)
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        if mode == "cpu":
            gt_matched_classes = gt_matched_classes.cuda()
            fg_mask = fg_mask.cuda()
            pred_ious_this_matching = pred_ious_this_matching.cuda()
            matched_gt_inds = matched_gt_inds.cuda()
            expanded_strides = expanded_strides.cuda()

        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds, fg_mask, expanded_strides

    def get_sample_region(self, gt_bboxes, strides, points, num_points_per_level, img_size):
        G = gt_bboxes.shape[0]  # num_gts
        A = points.shape[0]     # num_anchors
        expanded_strides = torch.zeros(1, points.shape[0]).to(dtype=points.dtype, device=points.device)

        beg = 0
        for l_ix in range(len(num_points_per_level)):
            end = beg + num_points_per_level[l_ix]
            expanded_strides[0][beg:end] = strides[l_ix]
            beg += num_points_per_level[l_ix]

        x_center = points[:, 0].unsqueeze(0).repeat(G, 1) + 0.5 * expanded_strides  # [G, A]
        y_center = points[:, 1].unsqueeze(0).repeat(G, 1) + 0.5 * expanded_strides  # [G, A]

        # x1, x2, y1, y2
        gt_bboxes_l = (gt_bboxes[:, 0]).unsqueeze(1).repeat(1, A)  # [G, A]
        gt_bboxes_r = (gt_bboxes[:, 2]).unsqueeze(1).repeat(1, A)  # [G, A]
        gt_bboxes_t = (gt_bboxes[:, 1]).unsqueeze(1).repeat(1, A)  # [G, A]
        gt_bboxes_b = (gt_bboxes[:, 3]).unsqueeze(1).repeat(1, A)  # [G, A]

        b_l = x_center - gt_bboxes_l  # [G, A]
        b_r = gt_bboxes_r - x_center  # [G, A]
        b_t = y_center - gt_bboxes_t  # [G, A]
        b_b = gt_bboxes_b - y_center  # [G, A]
        bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)   # [G, A, 4]
        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0   # [G, A]
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0         # [A]

        center_radius = self.radius

        gt_bboxes_cx = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2
        gt_bboxes_cy = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2

        if len(img_size) > 0:
            gt_bboxes_cx = torch.clamp(gt_bboxes_cx, min=0, max=img_size[1])
            gt_bboxes_cy = torch.clamp(gt_bboxes_cy, min=0, max=img_size[0])

        gt_bboxes_l = gt_bboxes_cx.unsqueeze(1).repeat(1, A) - center_radius * expanded_strides
        gt_bboxes_r = gt_bboxes_cx.unsqueeze(1).repeat(1, A) + center_radius * expanded_strides
        gt_bboxes_t = gt_bboxes_cy.unsqueeze(1).repeat(1, A) - center_radius * expanded_strides
        gt_bboxes_b = gt_bboxes_cy.unsqueeze(1).repeat(1, A) + center_radius * expanded_strides

        c_l = x_center - gt_bboxes_l
        c_r = gt_bboxes_r - x_center
        c_t = y_center - gt_bboxes_t
        c_b = gt_bboxes_b - y_center
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        is_in_centers_all = is_in_centers.sum(dim=0) > 0

        # in boxes and in centers
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all

        is_in_boxes_and_center = (
            is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
        )
        return is_in_boxes_anchor, is_in_boxes_and_center, expanded_strides

    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        matching_matrix = torch.zeros_like(cost)  # [G, A']

        ious_in_boxes_matrix = pair_wise_ious    # [G, A']
        n_candidate_k = min(self.candidate_k, ious_in_boxes_matrix.size(1))
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)  # , max=A_-1)
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
            )
            matching_matrix[gt_idx][pos_idx] = 1.0

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)    # [A']  wether matched
        if (anchor_matching_gt > 1).sum() > 0:
            cost_min, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0.0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
        fg_mask_inboxes = matching_matrix.sum(0) > 0.0
        num_fg = fg_mask_inboxes.sum().item()

        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[fg_mask_inboxes]

        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds
