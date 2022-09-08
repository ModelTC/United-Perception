# Import from third library
import torch

from up.utils.general.registry_factory import MATCHER_REGISTRY
from up.utils.general.log_helper import default_logger as logger
from up.tasks.det.models.utils.bbox_helper import offset2bbox
import torch.nn.functional as F
# Import from local
from .bbox_helper import bbox_iof_overlaps, bbox_iou_overlaps

GPU_MEMORY = None

__all_ = ['MaxIoUMatcher', 'RetinaOTAMatcher']


def cuda_memory_available(device, needed_memory):
    if torch.cuda.is_available():
        global GPU_MEMORY
        gbytes = 1024.0**3
        if GPU_MEMORY is None:
            GPU_MEMORY = torch.cuda.get_device_properties(device.index).total_memory
        alloated_memory = torch.cuda.memory_allocated()
        spare_memory = 0.5 * gbytes
        available_memory = GPU_MEMORY - alloated_memory - spare_memory
        if needed_memory < available_memory:
            return True
    return False


@MATCHER_REGISTRY.register('max_iou')
class MaxIoUMatcher(object):
    NEGATIVE_TARGET = -1
    IGNORE_TARGET = -2

    def __init__(self,
                 negative_iou_thresh,
                 positive_iou_thresh,
                 ignore_iou_thresh,
                 allow_low_quality_match,
                 low_quality_thresh=0):

        self.negative_iou_thresh = negative_iou_thresh
        self.positive_iou_thresh = positive_iou_thresh
        self.ignore_iou_thresh = ignore_iou_thresh
        self.allow_low_quality_match = allow_low_quality_match
        self.low_quality_thresh = low_quality_thresh

    def match(self, candidate_boxes, gt_bboxes, gt_ignores=None, return_max_overlaps=False):
        """
        Match roi to gt

        .. note::

            This function involves some temporarily tensors.

            - overlaps (``FloatTensor``): [N, M], ious between bounding boxes [N, 4] and gt boxes (M, 4),
              boxes format is (x1, y1, x2, y2)
            - ignore_overlaps (``FloatTensor``): [N, K], ious between bounding boxes [N, 4] with
              ignore regions [M, 4]

        Arguments:
            - candidate_boxes (``FloatTensor``): [N, 4] (x1, y1, x2, y2)
            - gt_bboxes (``FloatTensor``): [M, 5] (x1, y1, x2, y2, label)
            - gt_ignores (``FloatTensor`` or ``None``): [G, 4] (x1, y1, x2, y2)

        Returns:
            - target (LongTensor): [N], matched gt index for each RoI.

        .. note::

            1. if a roi is positive, target = matched gt index (>=0)
            2. if a roi is negative, target = -1,
            3. if a roi is ignored,  target = -2;
        """
        device = candidate_boxes.device
        # we use cpu tensor if there are too many bboxes, gts and ignores
        N, M = candidate_boxes.shape[0], gt_bboxes.shape[0]
        if gt_ignores is not None:
            M += gt_ignores.shape[0]
        if not cuda_memory_available(candidate_boxes.device, N * M * 2 * 4 * 4):
            candidate_boxes = candidate_boxes.cpu()
            gt_bboxes = gt_bboxes.cpu()
            if gt_ignores is not None:
                gt_ignores = gt_ignores.cpu()

        N = candidate_boxes.shape[0]
        M = gt_bboxes.shape[0]

        # check M > 0 for no-gt support
        if M == 0:
            overlaps = candidate_boxes.new_zeros(N, 1)
        else:
            overlaps = bbox_iou_overlaps(candidate_boxes, gt_bboxes)

        target = candidate_boxes.new_full((N,), self.IGNORE_TARGET, dtype=torch.int64)
        dt_to_gt_max, dt_to_gt_argmax = overlaps.max(dim=1)

        # rule 1: negative if maxiou < negative_iou_thresh:
        neg_mask = dt_to_gt_max < self.negative_iou_thresh
        target[neg_mask] = self.NEGATIVE_TARGET

        # rule 2: positive if maxiou > pos_iou_thresh
        pos_mask = dt_to_gt_max > self.positive_iou_thresh
        target[pos_mask] = dt_to_gt_argmax[pos_mask]

        # rule 3: positive if a dt has highest iou with any gt
        if self.allow_low_quality_match and M > 0:
            overlaps = overlaps.t()  # IMPORTANT, for faster caculation
            gt_to_dt_max, _ = overlaps.max(dim=1)
            temp_mask = (overlaps >= gt_to_dt_max[:, None] - 1e-3)
            if self.low_quality_thresh > 0:
                temp_mask = temp_mask * (overlaps > self.low_quality_thresh)
            lqm_dt_inds = torch.nonzero(temp_mask.any(dim=0)).reshape(-1)
            if lqm_dt_inds.numel() > 0:
                target[lqm_dt_inds] = dt_to_gt_argmax[lqm_dt_inds]
                pos_mask[lqm_dt_inds] = 1

        del overlaps
        ignore_overlaps = None
        if gt_ignores is not None and gt_ignores.numel() > 0:
            ignore_overlaps = bbox_iof_overlaps(candidate_boxes, gt_ignores)

        # rule 4: dt has high iou with ignore regions may not supposed to be negative
        if ignore_overlaps is not None and ignore_overlaps.numel() > 0:
            dt_to_ig_max, _ = ignore_overlaps.max(dim=1)
            ignored_dt_mask = dt_to_ig_max > self.ignore_iou_thresh
            # remove positives from ignored
            ignored_dt_mask = (ignored_dt_mask ^ (ignored_dt_mask & pos_mask))
            target[ignored_dt_mask] = self.IGNORE_TARGET

        if return_max_overlaps:
            return target.to(device), dt_to_gt_max
        else:
            return target.to(device)


@MATCHER_REGISTRY.register('retina_ota')
class RetinaOTAMatcher(object):
    def __init__(self,
                 num_classes,
                 use_SimOTA=True,
                 center_sample=True,
                 pos_radius=1,
                 candidate_k=10,
                 radius=2.5):
        self.pos_radius = pos_radius
        self.center_sample = center_sample
        self.num_classes = num_classes - 1  # 80
        self.candidate_k = candidate_k
        self.radius = radius
        self.use_SimOTA = use_SimOTA

    @torch.no_grad()
    def match(self, gts, preds, points, all_anchors, num_points_per_level,
              strides, mode='cuda', img_size=[], use_centerness_score=False):
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
            # 59500 [12, 1232](12是这个img的gt的数量) [1,59500]
            fg_mask, is_in_boxes_and_center, _ = self.get_sample_region(
                gt_bboxes, strides, points, num_points_per_level, img_size)
        else:
            assert "Not implement"

        if fg_mask.sum() <= 1e-6:
            logger.info('no gt in center')
            return 0, None, None, None, None

        masked_preds = preds[fg_mask]
        masked_all_anchors = all_anchors[fg_mask]

        if mode == 'cpu':
            masked_preds = masked_preds.cpu()
            masked_all_anchors = masked_all_anchors.cpu()

        preds_cls = masked_preds[:, :self.num_classes]
        preds_box = masked_preds[:, self.num_classes:-1]  # [A', 4]
        if use_centerness_score:
            preds_box = masked_preds[:, self.num_classes:-1]
            preds_centerness = masked_preds[:, -1:]
        else:
            preds_box = masked_preds[:, self.num_classes:]

        with torch.cuda.amp.autocast(enabled=False):
            decoded_preds_box = offset2bbox(masked_all_anchors, preds_box.float())

            pair_wise_ious = bbox_iou_overlaps(
                gt_bboxes, decoded_preds_box)
            gt_labels_ = gt_labels - 1
            gt_cls_per_image = (
                F.one_hot(gt_labels_.to(torch.int64), self.num_classes).float()
                .unsqueeze(1).repeat(1, masked_preds.shape[0], 1)
            )
            pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)
            if use_centerness_score:
                cls_preds_ = (
                    preds_cls.float().unsqueeze(0).repeat(G, 1, 1).sigmoid_()
                    * preds_centerness.float().unsqueeze(0).repeat(G, 1, 1).sigmoid_()
                )
                cls_preds_ = cls_preds_.sqrt_()
            else:
                cls_preds_ = preds_cls.float().unsqueeze(0).repeat(G, 1, 1).sigmoid_()
            pair_wise_cls_loss = F.binary_cross_entropy(
                cls_preds_,
                gt_cls_per_image,
                reduction="none").sum(-1)
            if not self.use_SimOTA:
                cls_loss_bg = F.binary_cross_entropy(
                    cls_preds_,
                    torch.zeros_like(cls_preds_),
                    reduction="none").sum(-1)

        del cls_preds_

        cost = (
            pair_wise_cls_loss
            + 3.0 * pair_wise_ious_loss
            + 100000.0 * (~is_in_boxes_and_center)
        )

        if not self.use_SimOTA:
            cost = torch.cat([cost, cls_loss_bg.unsqueeze(0)], dim=0)
            num_fg, gt_matched_classes, matched_gt_inds = self.dynamic_k_estimation(
                cost, pair_wise_ious, gt_labels, G, fg_mask)
        else:
            num_fg, gt_matched_classes, matched_gt_inds = self.dynamic_k_matching(
                cost, pair_wise_ious, gt_labels, G, fg_mask)
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        if mode == "cpu":
            gt_matched_classes = gt_matched_classes.cuda()
            fg_mask = fg_mask.cuda()
            # pred_ious_this_matching = pred_ious_this_matching.cuda()
            matched_gt_inds = matched_gt_inds.cuda()
            # expanded_strides = expanded_strides.cuda()

        return num_fg, gt_matched_classes, matched_gt_inds, fg_mask

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

    def dynamic_k_estimation(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        num_anchor = cost.shape[-1]
        # matching_matrix = torch.zeros_like(cost)
        ious_in_boxes_matrix = pair_wise_ious
        n_candidate_k = min(self.candidate_k, ious_in_boxes_matrix.size(1))
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)
        mu = pair_wise_ious.new_ones(num_gt + 1)
        mu[:-1] = torch.clamp(topk_ious.sum(1).int(), min=1).float()
        mu[-1] = num_anchor - mu[:-1].sum()
        nu = pair_wise_ious.new_ones(num_anchor)
        _, pi = self.sinkhorn(mu, nu, cost)

        # Rescale pi so that the max pi for each gt equals to 1.
        rescale_factor, _ = pi.max(dim=1)
        pi = pi / rescale_factor.unsqueeze(1)

        max_assigned_units, matched_gt_inds = torch.max(pi, dim=0)
        fg_mask_inboxes = matched_gt_inds != num_gt
        num_fg = fg_mask_inboxes.sum().item()
        fg_mask[fg_mask.clone()] = fg_mask_inboxes
        gt_matched_classes = gt_classes[matched_gt_inds]

        return num_fg, gt_matched_classes, matched_gt_inds

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

        return num_fg, gt_matched_classes, matched_gt_inds


class SinkhornDistance(torch.nn.Module):
    r"""
        Given two empirical measures each with :math:`P_1` locations
        :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
        outputs an approximation of the regularized OT cost for point clouds.
        Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
        'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
        'mean': the sum of the output will be divided by the number of
        elements in the output, 'sum': the output will be summed. Default: 'none'
        Shape:
            - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
            - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """

    def __init__(self, eps=1e-3, max_iter=100, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, mu, nu, C):
        u = torch.ones_like(mu)
        v = torch.ones_like(nu)

        # Sinkhorn iterations
        for i in range(self.max_iter):
            v = self.eps * \
                (torch.log(
                    nu + 1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            u = self.eps * \
                (torch.log(
                    mu + 1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(
            self.M(C, U, V)).detach()
        # Sinkhorn distance
        cost = torch.sum(
            pi * C, dim=(-2, -1))
        return cost, pi

    def M(self, C, u, v):
        '''
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / epsilon$"
        '''
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps


def build_matcher(cfg):
    return MATCHER_REGISTRY.build(cfg)


def match(boxes, gt, cfg, gt_ignores=None):
    matcher = build_matcher(cfg)
    return matcher.match(boxes, gt, gt_ignores, return_max_overlaps=True)
