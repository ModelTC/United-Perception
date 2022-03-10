import numpy as np
import torch
import torch.nn as nn
from up.utils.general.registry_factory import MODULE_ZOO_REGISTRY
from up.utils.model.normalize import build_norm_layer


@MODULE_ZOO_REGISTRY.register('base_bev')
class BaseBEVBackbone(nn.Module):
    def __init__(self,
                 inplanes,
                 layer_nums=None,
                 layer_strides=None,
                 num_filters=None,
                 upsample_strides=None,
                 num_upsample_filters=None,
                 normalize={'type': 'solo_bn'}):
        super().__init__()
        if layer_nums is not None:
            assert len(layer_nums) == len(layer_strides) == len(num_filters)
        else:
            layer_nums = layer_strides = num_filters = []

        if upsample_strides is not None:
            assert len(upsample_strides) == len(num_upsample_filters)
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        c_in_list = [inplanes, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                build_norm_layer(num_filters[idx], normalize)[1],
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    build_norm_layer(num_filters[idx], normalize)[1],
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        build_norm_layer(num_upsample_filters[idx], normalize)[1],
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        build_norm_layer(num_upsample_filters[idx], normalize)[1],
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                build_norm_layer(c_in, normalize)[1],
                nn.ReLU(),
            ))
        self.out_planes = c_in

    def get_outplanes(self):
        return self.out_planes

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']
        ups = []
        ret_dict = {}
        x = spatial_features
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)
        data_dict['spatial_features_2d'] = x

        return data_dict
