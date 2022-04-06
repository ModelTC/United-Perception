import torch
import torch.nn as nn
import torch.nn.functional as F
from up.utils.general.registry_factory import MODULE_ZOO_REGISTRY


class VFETemplate(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def get_output_feature_dim(self):
        raise NotImplementedError

    def forward(self, **kwargs):
        """
        Args:
            **kwargs:

        Returns:
            batch_dict:
                ...
                vfe_features: (num_voxels, C)
        """
        raise NotImplementedError


@MODULE_ZOO_REGISTRY.register('pillar_vfe')
class PillarVFE(VFETemplate):
    def __init__(self, num_point_features, num_filters, with_distance=False, use_absolute_xyz=True, use_norm=True):
        super().__init__()
        self.with_distance = with_distance
        self.use_absolute_xyz = use_absolute_xyz
        self.use_norm = use_norm

        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = num_filters
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)
        self.out_planes = num_filters[-1]

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def get_f_center(self, voxel_features, coords, voxel_size, point_cloud_range):
        voxel_x = voxel_size[0]
        voxel_y = voxel_size[1]
        voxel_z = voxel_size[2]
        x_offset = voxel_x / 2 + point_cloud_range[0]
        y_offset = voxel_y / 2 + point_cloud_range[1]
        z_offset = voxel_z / 2 + point_cloud_range[2]

        f_center = torch.zeros_like(voxel_features[:, :, :3])
        f_center[:, :, 0] = voxel_features[:, :, 0] - \
            (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * voxel_x + x_offset)
        f_center[:, :, 1] = voxel_features[:, :, 1] - \
            (coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * voxel_y + y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - \
            (coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * voxel_z + z_offset)
        return f_center

    def get_pillar_features(self, voxel_infos, voxel_features, coords, voxel_num_points):
        voxel_size, point_cloud_range = voxel_infos['voxel_size'], voxel_infos['point_cloud_range']
        points_mean = torch.sum(voxel_features[:, :, :3], dim=1, keepdim=True) / \
            voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        f_cluster = voxel_features[:, :, :3] - points_mean
        f_center = self.get_f_center(voxel_features, coords, voxel_size, point_cloud_range)
        if self.use_absolute_xyz:
            features = [voxel_features, f_cluster, f_center]
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]
        if self.with_distance:
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
            features.append(points_dist)

        features = torch.cat(features, dim=-1)
        voxel_count = features.shape[1]
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        features *= mask

        for pfn in self.pfn_layers:
            features = pfn(features)
        features = features.squeeze()
        return features

    def scatter(self, voxel_infos, pillar_features, coords):
        self.nx, self.ny, self.nz = voxel_infos['grid_size']
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.out_planes,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.out_planes * self.nz, self.ny, self.nx)
        return batch_spatial_features

    def forward(self, input):
        voxel_infos = input['voxel_infos']
        voxel_features, coords, voxel_num_points = input['voxels'], input['voxel_coords'], input['voxel_num_points']
        pillar_features = self.get_pillar_features(voxel_infos, voxel_features, coords, voxel_num_points)
        spatial_features = self.scatter(voxel_infos, pillar_features, coords)
        return {'spatial_features': spatial_features}

    def get_outplanes(self):
        return self.out_planes


class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()

        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.part = 50000
        self.out_planes = out_channels

    def forward(self, inputs):
        if inputs.shape[0] > self.part:
            # nn.Linear performs randomly when batch size is too large
            num_parts = inputs.shape[0] // self.part
            part_linear_out = [self.linear(inputs[num_part * self.part:(num_part + 1) * self.part])
                               for num_part in range(num_parts + 1)]
            x = torch.cat(part_linear_out, dim=0)
        else:
            x = self.linear(inputs)
        torch.backends.cudnn.enabled = False
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else x
        torch.backends.cudnn.enabled = True
        x = F.relu(x)
        x_max = torch.max(x, dim=1, keepdim=True)[0]

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated

    def get_outplanes(self):
        return self.out_planes
