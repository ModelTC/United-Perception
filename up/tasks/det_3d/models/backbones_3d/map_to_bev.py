import torch.nn as nn
from up.utils.general.registry_factory import MODULE_ZOO_REGISTRY


@MODULE_ZOO_REGISTRY.register('height_compression')
class HeightCompression(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, feature_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        encoded_spconv_tensor = feature_dict['encoded_spconv_tensor']
        spatial_features = encoded_spconv_tensor.dense()
        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)
        feature_dict.update({'spatial_features': spatial_features})
        feature_dict.update({'spatial_features_stride': feature_dict['encoded_spconv_tensor_stride']})
        return feature_dict


def build_map_to_bev(cfg_map_to_bev):
    return MODULE_ZOO_REGISTRY.build(cfg_map_to_bev)
