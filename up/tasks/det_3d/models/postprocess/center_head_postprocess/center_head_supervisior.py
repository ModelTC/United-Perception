import numpy as np
import torch
from up.utils.general.registry_factory import ROI_SUPERVISOR_REGISTRY
from up.tasks.det_3d.models.utils.center_utils import gaussian_radius

__all__ = ['BaseCenterSupervisor', 'CenterSupervisor']


class BaseCenterSupervisor(object):
    def __init__(self, feature_map_stride, num_max_objs, gaussian_hm=True, radius=None):
        super().__init__()
        self.feature_map_stride = feature_map_stride
        self.num_max_objs = num_max_objs
        self.gaussian_hm = gaussian_hm
        self.radius = radius

    def get_targets(self, class_names, voxel_infos, gt_boxes, feature_map_size):
        """
        Args:
            gt_boxes: (B, M, 8)
            range_image_polar: (B, 3, H, W)
            feature_map_size: (2) [H, W]
            spatial_cartesian: (B, 4, H, W)
        Returns:

        """
        self.voxel_size = voxel_infos['voxel_size']
        self.point_cloud_range = voxel_infos['point_cloud_range']
        feature_map_size = feature_map_size[::-1]  # [H, W][200, 176] ==> [x, y][176, 200]
        batch_size = gt_boxes.shape[0]
        all_names = np.array(['bg', *class_names])
        heatmap_list, target_boxes_list, inds_list, masks_list = [], [], [], []
        for bs_idx in range(batch_size):
            cur_gt_boxes = gt_boxes[bs_idx]
            gt_class_names = all_names[cur_gt_boxes[:, -1].cpu().long().numpy()]
            gt_boxes_single_head = []
            for idx, name in enumerate(gt_class_names):
                if name not in class_names:
                    continue
                temp_box = cur_gt_boxes[idx]
                temp_box[-1] = class_names.index(name) + 1
                gt_boxes_single_head.append(temp_box[None, :])
            if len(gt_boxes_single_head) == 0:
                gt_boxes_single_head = cur_gt_boxes[:0, :]
            else:
                gt_boxes_single_head = torch.cat(gt_boxes_single_head, dim=0)

            heatmap, ret_boxes, inds, mask = self.assign_target_of_single_head(
                num_classes=len(class_names), gt_boxes=gt_boxes_single_head.cpu(),
                feature_map_size=feature_map_size, feature_map_stride=self.feature_map_stride,
                num_max_objs=self.num_max_objs
            )
            heatmap_list.append(heatmap.to(gt_boxes_single_head.device))
            target_boxes_list.append(ret_boxes.to(gt_boxes_single_head.device))
            inds_list.append(inds.to(gt_boxes_single_head.device))
            masks_list.append(mask.to(gt_boxes_single_head.device))
        heatmaps_target = torch.stack(heatmap_list, dim=0)
        boxes_target = torch.stack(target_boxes_list, dim=0)
        inds_all = torch.stack(inds_list, dim=0)
        masks_all = torch.stack(masks_list, dim=0)
        return heatmaps_target, boxes_target, inds_all, masks_all

    def assign_target_of_single_head(
            self, num_classes, gt_boxes, feature_map_size, feature_map_stride, num_max_objs):
        """
        Args:
            gt_boxes: (N, 8)
            feature_map_size: (2), [x, y]

        Returns:

        """
        heatmap = gt_boxes.new_zeros(num_classes, feature_map_size[1], feature_map_size[0])
        ret_boxes = gt_boxes.new_zeros((num_max_objs, gt_boxes.shape[-1] - 1 + 1))
        inds = gt_boxes.new_zeros(num_max_objs).long()
        mask = gt_boxes.new_zeros(num_max_objs).long()
        x, y, z = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2]
        coord_x = (x - self.point_cloud_range[0]) / self.voxel_size[0] / feature_map_stride
        coord_y = (y - self.point_cloud_range[1]) / self.voxel_size[1] / feature_map_stride
        coord_x = torch.clamp(coord_x, min=0, max=feature_map_size[0] - 0.5)
        coord_y = torch.clamp(coord_y, min=0, max=feature_map_size[1] - 0.5)
        center = torch.cat((coord_x[:, None], coord_y[:, None]), dim=-1)
        center_int = center.int()
        center_int_float = center_int.float()
        dx, dy, _ = gt_boxes[:, 3], gt_boxes[:, 4], gt_boxes[:, 5]
        dx = dx / self.voxel_size[0] / feature_map_stride
        dy = dy / self.voxel_size[1] / feature_map_stride

        if self.gaussian_hm:
            radius = gaussian_radius(dx, dy, self.gaussian_overlap)
            radius = torch.clamp_min(radius.int(), min=self.min_radius)
        else:
            radius = self.radius

        for k in range(min(num_max_objs, gt_boxes.shape[0])):
            if dx[k] <= 0 or dy[k] <= 0:
                continue
            if not (0 <= center_int[k][0] <= feature_map_size[0] and 0 <= center_int[k][1]
                    <= feature_map_size[1]):
                continue
            cur_class_id = (gt_boxes[k, -1] - 1).long()
            radius_k = radius[k].item() if self.gaussian_hm else radius
            self.get_heatmap(heatmap[cur_class_id], center[k], radius_k)

            inds[k] = center_int[k, 1] * feature_map_size[0] + center_int[k, 0]
            mask[k] = 1

            ret_boxes[k, 0:2] = center[k] - center_int_float[k].float()
            ret_boxes[k, 2] = z[k]
            ret_boxes[k, 3:6] = gt_boxes[k, 3:6].log()
            ret_boxes[k, 6] = torch.cos(gt_boxes[k, 6])
            ret_boxes[k, 7] = torch.sin(gt_boxes[k, 6])
            if gt_boxes.shape[1] > 8:
                ret_boxes[k, 8:] = gt_boxes[k, 7:-1]
        return heatmap, ret_boxes, inds, mask

    def get_heatmap(self):
        raise NotImplementedError


@ROI_SUPERVISOR_REGISTRY.register('center_head')
class CenterSupervisor(BaseCenterSupervisor):
    def __init__(self, feature_map_stride, num_max_objs, gaussian_hm=True, gaussian_overlap=0.1, min_radius=2):
        super(CenterSupervisor, self).__init__(feature_map_stride, num_max_objs, gaussian_hm)
        self.gaussian_overlap = gaussian_overlap
        self.min_radius = min_radius

    def get_heatmap(self, heatmap, center, radius, k=1, valid_mask=None):
        diameter = 2 * radius + 1
        gaussian = self.gaussian2D((diameter, diameter), sigma=diameter / 6)

        x, y = int(center[0]), int(center[1])
        height, width = heatmap.shape[0:2]
        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)

        masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = torch.from_numpy(
            gaussian[radius - top:radius + bottom, radius - left:radius + right]
        ).to(heatmap.device).float()
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
            if valid_mask is not None:
                cur_valid_mask = valid_mask[y - top:y + bottom, x - left:x + right]
                masked_gaussian = masked_gaussian * cur_valid_mask.float()
            torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
        return heatmap

    def gaussian2D(self, shape, sigma=1):
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return h
