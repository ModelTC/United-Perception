import torch
import numpy as np
import json
from up.utils.general.registry_factory import ANCHOR_GENERATOR_REGISTRY


@ANCHOR_GENERATOR_REGISTRY.register('3d')
class AnchorGenerator(object):
    def __init__(self, anchor_range, base_anchors_file):
        super().__init__()
        with open(base_anchors_file, 'r') as f:
            self.anchor_generator_cfg = np.array(json.load(f))
        self.anchor_range = anchor_range
        self.anchor_sizes = [config['anchor_sizes'] for config in self.anchor_generator_cfg]
        self.anchor_rotations = [config['anchor_rotations'] for config in self.anchor_generator_cfg]
        self.anchor_heights = [config['anchor_bottom_heights'] for config in self.anchor_generator_cfg]
        self.align_center = [config.get('align_center', False) for config in self.anchor_generator_cfg]

        assert len(self.anchor_sizes) == len(self.anchor_rotations) == len(self.anchor_heights)
        self.num_of_anchor_sets = len(self.anchor_sizes)

    def generate_anchors(self, grid_sizes):
        assert len(grid_sizes) == self.num_of_anchor_sets
        all_anchors = []
        num_anchors_per_location = []
        for grid_size, anchor_size, anchor_rotation, anchor_height, align_center in zip(
                grid_sizes, self.anchor_sizes, self.anchor_rotations, self.anchor_heights, self.align_center):

            num_anchors_per_location.append(len(anchor_rotation) * len(anchor_size) * len(anchor_height))
            if align_center:
                x_stride = (self.anchor_range[3] - self.anchor_range[0]) / grid_size[0]
                y_stride = (self.anchor_range[4] - self.anchor_range[1]) / grid_size[1]
                x_offset, y_offset = x_stride / 2, y_stride / 2
            else:
                x_stride = (self.anchor_range[3] - self.anchor_range[0]) / (grid_size[0] - 1)
                y_stride = (self.anchor_range[4] - self.anchor_range[1]) / (grid_size[1] - 1)
                x_offset, y_offset = 0, 0
            x_shifts = torch.arange(
                self.anchor_range[0] + x_offset, self.anchor_range[3] + 1e-5, step=x_stride, dtype=torch.float32,
            ).cuda()
            y_shifts = torch.arange(
                self.anchor_range[1] + y_offset, self.anchor_range[4] + 1e-5, step=y_stride, dtype=torch.float32,
            ).cuda()
            z_shifts = x_shifts.new_tensor(anchor_height)

            num_anchor_size, num_anchor_rotation = anchor_size.__len__(), anchor_rotation.__len__()
            anchor_rotation = x_shifts.new_tensor(anchor_rotation)
            anchor_size = x_shifts.new_tensor(anchor_size)
            x_shifts, y_shifts, z_shifts = torch.meshgrid([
                x_shifts, y_shifts, z_shifts
            ])
            anchors = torch.stack((x_shifts, y_shifts, z_shifts), dim=-1)
            anchors = anchors[:, :, :, None, :].repeat(1, 1, 1, anchor_size.shape[0], 1)
            anchor_size = anchor_size.view(1, 1, 1, -1, 3).repeat([*anchors.shape[0:3], 1, 1])
            anchors = torch.cat((anchors, anchor_size), dim=-1)
            anchors = anchors[:, :, :, :, None, :].repeat(1, 1, 1, 1, num_anchor_rotation, 1)
            anchor_rotation = anchor_rotation.view(
                1, 1, 1, 1, -1, 1).repeat([*anchors.shape[0:3], num_anchor_size, 1, 1])
            anchors = torch.cat((anchors, anchor_rotation), dim=-1)

            anchors = anchors.permute(2, 1, 0, 3, 4, 5).contiguous()
            anchors[..., 2] += anchors[..., 5] / 2
            all_anchors.append(anchors)
        return all_anchors, num_anchors_per_location

    def get_anchors(self, grid_size, anchor_ndim=7):
        feature_map_size = [grid_size[:2] // config['feature_map_stride'] for config in self.anchor_generator_cfg]
        anchors_list, num_anchors_per_location_list = self.generate_anchors(feature_map_size)

        if anchor_ndim != 7:
            for idx, anchors in enumerate(anchors_list):
                pad_zeros = anchors.new_zeros([*anchors.shape[0:-1], anchor_ndim - 7])
                new_anchors = torch.cat((anchors, pad_zeros), dim=-1)
                anchors_list[idx] = new_anchors

        return anchors_list, num_anchors_per_location_list
