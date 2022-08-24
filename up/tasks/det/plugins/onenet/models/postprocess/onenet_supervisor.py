# Import from third library

import torch

from scipy.optimize import linear_sum_assignment

from up.tasks.det.models.utils.matcher import build_matcher
from up.utils.general.registry_factory import MATCHER_REGISTRY, ROI_SUPERVISOR_REGISTRY
from up.tasks.det.models.utils.bbox_helper import generalized_box_iou, filter_by_size


__all__ = ['OnenetSupervisor', 'OnenetMatcher']


@ROI_SUPERVISOR_REGISTRY.register('onenet')
class OnenetSupervisor(object):
    def __init__(self, matcher):
        self.matcher = build_matcher(matcher)

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_targets(self, mlvl_preds, input):
        '''
        Get the targets of classification and regression.

        Args:
            mlvl_preds: a list contains cls_preds and loc_preds (xyxy)
                cls_preds.shape: [B, K, C], loc_preds.shape: [B, K, 4]
        Returns:

        '''
        cls_targets = []
        reg_targets = []

        gt_bboxes = input['gt_bboxes']
        img_info = input['image_info']
        cls_preds = mlvl_preds[0]
        loc_preds = mlvl_preds[1]
        B, K, _ = cls_preds.shape

        targets_ori = []
        for i in range(len(gt_bboxes)):
            tgt = {}
            gts, _ = filter_by_size(gt_bboxes[i], min_size=1)

            tgt['labels'] = gts[:, 4] - 1
            tgt['bboxes'] = gts[:, :4]
            tgt['img_whwh'] = gts.new_tensor([*img_info[i][:2], *img_info[i][:2]][::-1])
            targets_ori.append(tgt)

        indices = self.matcher.match(mlvl_preds, targets_ori)

        pos_idx = self._get_src_permutation_idx(indices)
        cls_targets = gt_bboxes[0].new_full((B, K), 0, dtype=torch.int64)
        cls_targets_o = torch.cat([t['labels'][J] + 1 for t, (_, J) in zip(targets_ori, indices)]).long()
        cls_targets[pos_idx] = cls_targets_o

        pos_loc_preds = loc_preds[pos_idx]
        reg_targets = torch.cat([t['bboxes'][J] for t, (_, J) in zip(targets_ori, indices)], dim=0)

        img_whwh = torch.cat(
            [t['img_whwh'].unsqueeze(0).expand(len(J), 4) for t, (_, J) in zip(targets_ori, indices)], dim=0
        )

        num_boxes = sum(len(t['labels']) for t in targets_ori)
        num_boxes = torch.Tensor([num_boxes]).type_as(loc_preds).to(loc_preds.device)

        pos_cls_targets = cls_targets[pos_idx]
        pos_cls_preds = cls_preds[pos_idx]

        return [cls_targets.detach(),
                reg_targets.detach(),
                cls_preds,
                pos_loc_preds,
                img_whwh.detach(),
                num_boxes.detach(),
                pos_cls_targets.detach(),
                pos_cls_preds.detach(),
                pos_idx]


@MATCHER_REGISTRY.register('onenet')
class OnenetMatcher(object):
    def __init__(self, cost_cls, cost_l1, cost_giou, alpha=0.25, gamma=2, easy_match=True):
        self.cost_cls = cost_cls
        self.cost_l1 = cost_l1
        self.cost_giou = cost_giou
        self.alpha = alpha
        self.gamma = gamma
        self.easy_match = easy_match

    @torch.no_grad()
    def match(self, outputs, targets):
        B, K, _ = outputs[0].shape
        batch_pred_logits = outputs[0].sigmoid()
        batch_pred_bboxes = outputs[1]

        indices = []
        for i in range(B):
            tgt_ids = targets[i]['labels'].long()
            if tgt_ids.shape[0] == 0:
                indices.append((torch.as_tensor([]), torch.as_tensor([])))
                continue

            tgt_bboxes = targets[i]['bboxes']
            pred_logits = batch_pred_logits[i]
            pred_bboxes = batch_pred_bboxes[i]
            neg_cost_cls = (1 - self.alpha) * (pred_logits ** self.gamma) * (-(1 - pred_logits + 1e-8).log())
            pos_cost_cls = self.alpha * ((1 - pred_logits) ** self.gamma) * (-(pred_logits + 1e-8).log())
            cost_cls = pos_cost_cls[:, tgt_ids] - neg_cost_cls[:, tgt_ids]

            img_size_i = targets[i]['img_whwh']
            pred_bboxes_ = pred_bboxes / img_size_i
            tgt_bboxes_ = tgt_bboxes / img_size_i

            cost_l1 = torch.cdist(pred_bboxes_, tgt_bboxes_, p=1)

            if (pred_bboxes[:, 2:] < pred_bboxes[:, :2]).any():
                mask = pred_bboxes[:, 2:] < pred_bboxes[:, :2]
                mask = torch.logical_or(mask[:, 0], mask[:, 1])
                pred_bboxes[mask] = torch.zeros_like(pred_bboxes[mask])

            cost_giou = -generalized_box_iou(pred_bboxes, tgt_bboxes)

            cost_matrix = self.cost_cls * cost_cls + self.cost_l1 * cost_l1 + self.cost_giou * cost_giou

            if not self.easy_match:
                pred_ind, tgt_ind = linear_sum_assignment(cost_matrix.cpu())
            else:
                _, pred_ind = torch.min(cost_matrix, dim=0)
                tgt_ind = torch.arange(len(tgt_ids)).to(pred_ind)
            indices.append((pred_ind, tgt_ind))

        return [
            (torch.as_tensor(i, dtype=torch.int64, device=outputs[0].device),
             torch.as_tensor(j, dtype=torch.int64, device=outputs[0].device)) for i, j in indices
        ]


def build_onenet_supervisor(supervisor_cfg):
    return ROI_SUPERVISOR_REGISTRY.build(supervisor_cfg)
