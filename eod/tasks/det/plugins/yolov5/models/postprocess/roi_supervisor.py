import torch

from eod.utils.general.registry_factory import ROI_SUPERVISOR_REGISTRY
from eod.tasks.det.models.utils.bbox_helper import xyxy2xywh


__all__ = ['YoloV5Supervisor']


def norm_croods(data):
    # for yolov5 coords norm
    targets = data['gt_bboxes']
    norm_targets = []
    for batch_id in range(len(targets)):
        gt_bboxes = targets[batch_id]
        gt_bboxes[:, :4] = xyxy2xywh(gt_bboxes[:, :4], stacked=True)
        gt_bboxes[:, 4:5] -= 1
        gt_bboxes = torch.cat([gt_bboxes[:, 4:5], gt_bboxes[:, :4]], 1)
        h1, w1 = data['image'][batch_id].shape[1:]
        gt_bboxes[:, [1, 3]] /= w1
        gt_bboxes[:, [2, 4]] /= h1
        gt_bboxes[:, 1:].clamp_(0, 1)
        norm_targets.append(gt_bboxes)
    data['gt_bboxes'] = norm_targets
    return data


@ROI_SUPERVISOR_REGISTRY.register('yolov5')
class YoloV5Supervisor(object):
    def __init__(self,
                 num_classes,
                 strides,
                 cls_normalizer=1.,
                 iou_normalizer=1.,
                 anchor_thresh=4.0,
                 pick_bias=0.5,
                 class_normalizer=None):

        self.cls_normalizer = cls_normalizer
        self.iou_normalizer = iou_normalizer
        self.strides = strides
        self.anchor_thresh = anchor_thresh
        self.num_classes = num_classes
        self.class_normalizer = class_normalizer if class_normalizer is not None \
            else [1. for _ in range(self.num_classes)]

        self.pos = 1.0
        self.neg = 0.0

        self.pick_bias = pick_bias
        self.offset = torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]]).float() * pick_bias

    def get_targets(self, input, mlvl_preds, mlvl_anchors):
        input = norm_croods(input)
        targets = input['gt_bboxes']
        outs = []
        device = mlvl_preds[0][0].device

        if self.offset.device != device:
            self.offset = self.offset.to(device)
            self.class_normalizer = torch.tensor(self.class_normalizer, device=device)

        for lvl_id, (preds, anchors) in enumerate(zip(mlvl_preds, mlvl_anchors)):
            loc_pred, cls_pred = preds[:2]
            b, _, h, w, _ = loc_pred.shape

            stride = self.strides[lvl_id]
            anchors = anchors / stride
            anchor_nums = anchors.shape[0]

            box_mask = mlvl_anchors.new_zeros((b, anchor_nums, h, w, 1))
            cls_mask = mlvl_anchors.new_zeros((b, anchor_nums, h, w, self.num_classes))
            box_gt = mlvl_anchors.new_zeros((b, anchor_nums, h, w, 4))
            cls_gt = mlvl_anchors.new_zeros((b, anchor_nums, h, w, self.num_classes))

            gt = []
            bbox_nums = 0
            for batch_id in range(len(targets)):
                item = targets[batch_id]
                item_size = item.shape[0]
                bbox_nums += item_size

                # yolov5 Encoding
                gt_cls = (item[:, 0]).view(item_size, 1)
                gt_x = (item[:, 1] * w).view(item_size, 1)
                gt_y = (item[:, 2] * h).view(item_size, 1)
                gt_w = (item[:, 3] * w).view(item_size, 1)
                gt_h = (item[:, 4] * h).view(item_size, 1)
                item_bid = gt_x.new_full((item_size, 1), batch_id)
                gt.append(torch.cat((gt_x, gt_y, gt_w, gt_h, gt_cls, item_bid), 1))

            gt = torch.cat(gt, 0)
            if not gt.shape[0]:
                outs.append([box_gt, cls_gt, box_mask, cls_mask])
                continue
            anchor_index = torch.arange(anchor_nums, device=device).float().view(anchor_nums, 1).repeat(1, bbox_nums)
            expand_gt = torch.cat((gt.repeat(anchor_nums, 1, 1),
                                   anchor_index[:, :, None]), 2)   # x, y, w, h, cls, bid, aid

            ratio = expand_gt[:, :, 2:4] / anchors[:, None]
            size_keep = torch.max(ratio, 1. / ratio).max(2)[0] < self.anchor_thresh

            gt_with_anchors = expand_gt[size_keep]
            if not gt_with_anchors.shape[0]:
                outs.append([box_gt, cls_gt, box_mask, cls_mask])
                continue

            # Offsets
            grid_xy = gt_with_anchors[:, :2]
            grid_xy_inverse = torch.tensor([h, w], device=device).float() - grid_xy
            left_pos, top_pos = ((grid_xy % 1. < self.pick_bias) & (grid_xy > 1.)).T
            right_pos, bottom_pos = ((grid_xy_inverse % 1. < self.pick_bias) & (grid_xy_inverse > 1.)).T
            all_pos = torch.stack((torch.ones_like(left_pos), left_pos, top_pos, right_pos, bottom_pos))
            gt_with_anchors = gt_with_anchors.repeat((5, 1, 1))[all_pos]
            offsets = (torch.zeros_like(grid_xy)[None] + self.offset[:, None])[all_pos]

            # Define
            grid_xy = gt_with_anchors[:, :2]  # grid xy
            grid_wh = gt_with_anchors[:, 2:4]  # grid wh
            grid_ij = (grid_xy - offsets).long()
            grid_i, grid_j = grid_ij.T  # grid xy indices
            grid_i = grid_i.clamp_(0, w - 1)
            grid_j = grid_j.clamp_(0, h - 1)

            # Append
            cls = gt_with_anchors[:, 4].long()  # cls
            bid = gt_with_anchors[:, 5].long()  # batch index
            aid = gt_with_anchors[:, 6].long()  # anchor index

            box_gt[bid, aid, grid_j, grid_i, :] = torch.cat((grid_xy - grid_ij, grid_wh), 1)
            cls_gt[bid, aid, grid_j, grid_i] = cls_gt[bid, aid, grid_j, grid_i]. \
                scatter_(-1, cls.unsqueeze(1), self.pos)

            box_mask[bid, aid, grid_j, grid_i, :] = self.class_normalizer[cls].unsqueeze(-1)
            cls_mask[bid, aid, grid_j, grid_i, :] = self.class_normalizer[cls].unsqueeze(-1).repeat(1, self.num_classes)

            outs.append([box_gt, cls_gt, box_mask, cls_mask])
        return outs
