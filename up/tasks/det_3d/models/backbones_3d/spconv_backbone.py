from functools import partial
import torch.nn as nn
import numpy as np
from up.utils.general.registry_factory import MODULE_ZOO_REGISTRY
from up.tasks.det_3d.models.backbones_3d.vfe.mean_vfe import build_vfe
from up.tasks.det_3d.models.backbones_3d.map_to_bev import build_map_to_bev
from up.utils.general.log_helper import default_logger as logger
from typing import Set
try:
    try:
        import spconv.pytorch as spconv
        from spconv.pytorch import SparseModule
    except BaseException:
        import spconv as spconv
        from spconv import SparseModule
except Exception as err:  # ugly code, fake spconv module to avoid error
    logger.warning(f"import error {err}; If you need spconv, you should install spconv !!!. Or just ignore this error")
    spconv = None
    SparseModule = nn.Module

__all__ = ["VoxelResBackBone8x", "VoxelBackBone8x"]


def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):
    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m


class SparseBasicBlock(SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None, indice_key=None):
        super(SparseBasicBlock, self).__init__()
        assert norm_fn is not None
        bias = norm_fn is not None
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))
        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))
        return out


@MODULE_ZOO_REGISTRY.register('voxel_8x')
class VoxelBackBone8x(nn.Module):
    def __init__(self, input_channels, point_cloud_range, voxel_size, cfg, last_pad=0):
        super().__init__()
        self.vfe = build_vfe(cfg['vfe'])
        self.map_to_bev = build_map_to_bev(cfg['map_to_bev'])
        self.num_point_features = 128

        grid_size = (np.array(point_cloud_range[3:6])
                     - np.array(point_cloud_range[0:3])) / np.array(voxel_size)
        grid_size = np.round(grid_size).astype(np.int64)

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.last_pad = last_pad
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        # Submanifold Sparse Convolution
        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(64, self.num_point_features, (3, 1, 1), stride=(2, 1, 1), padding=self.last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 64
        }
        self.outplanes = self.num_point_features * 2

    def get_outplanes(self):
        return self.outplanes

    def forward_net(self, voxel_features, voxel_coords, batch_size):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)
        output = {}
        output.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        output.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })
        output.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })
        return output

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            output:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_coords, batch_size = batch_dict['voxel_coords'], batch_dict['batch_size']
        voxel_features = self.vfe(batch_dict)
        feature_dict = self.forward_net(voxel_features, voxel_coords, batch_size)
        output = self.map_to_bev(feature_dict)
        return output


@MODULE_ZOO_REGISTRY.register('voxel_res_8x')
class VoxelResBackBone8x(nn.Module):
    def __init__(self, input_channels, point_cloud_range, voxel_size, cfg, last_pad=0):
        super().__init__()
        self.vfe = build_vfe(cfg['vfe'])
        self.map_to_bev = build_map_to_bev(cfg['map_to_bev'])
        self.num_point_features = 128

        grid_size = (np.array(point_cloud_range[3:6])
                     - np.array(point_cloud_range[0:3])) / np.array(voxel_size)
        grid_size = np.round(grid_size).astype(np.int64)

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.last_pad = last_pad
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        # Submanifold Sparse Convolution
        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'),
        )

        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(128, self.num_point_features, (3, 1, 1), stride=(2, 1, 1), padding=self.last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 128
        }
        self.outplanes = self.num_point_features * 2

    def get_outplanes(self):
        return self.outplanes

    def forward_net(self, voxel_features, voxel_coords, batch_size):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)
        output = {}
        output.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        output.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })

        return output

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            output:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_coords, batch_size = batch_dict['voxel_coords'], batch_dict['batch_size']
        voxel_features = self.vfe(batch_dict)
        feature_dict = self.forward_net(voxel_features, voxel_coords, batch_size)
        output = self.map_to_bev(feature_dict)
        return output


def find_all_spconv_keys(model: nn.Module, prefix="") -> Set[str]:
    """
    Finds all spconv keys that need to have weight's transposed
    """
    found_keys: Set[str] = set()
    for name, child in model.named_children():
        new_prefix = f"{prefix}.{name}" if prefix != "" else name

        if isinstance(child, spconv.conv.SparseConvolution):
            new_prefix = f"{new_prefix}.weight"
            found_keys.add(new_prefix)

        found_keys.update(find_all_spconv_keys(child, prefix=new_prefix))

    return found_keys


def replace_feature(out, new_features):
    if "replace_feature" in out.__dir__():
        # spconv 2.x behaviour
        return out.replace_feature(new_features)
    else:
        out.features = new_features
        return out
