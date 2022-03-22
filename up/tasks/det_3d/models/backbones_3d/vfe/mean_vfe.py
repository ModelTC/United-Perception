import torch
from .pillar_vfe import VFETemplate
from up.utils.general.registry_factory import MODULE_ZOO_REGISTRY


@MODULE_ZOO_REGISTRY.register('mean_vfe')
class MeanVFE(VFETemplate):
    def __init__(self):
        super().__init__()

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                voxels: (num_voxels, max_points_per_voxel, C)
                voxel_num_points: optional (num_voxels)
            **kwargs:

        Returns:
            vfe_features: (num_voxels, C)
        """
        voxel_features, voxel_num_points = batch_dict['voxels'], batch_dict['voxel_num_points']
        points_mean = voxel_features[:, :, :].sum(dim=1, keepdim=False)
        normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxel_features)
        points_mean = points_mean / normalizer
        voxel_features = points_mean.contiguous()
        return voxel_features


def build_vfe(cfg_vfe):
    return MODULE_ZOO_REGISTRY.build(cfg_vfe)
