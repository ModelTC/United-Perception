import torch.nn as nn
from up.utils.general.registry_factory import MODULE_ZOO_REGISTRY
from up.utils.model.normalize import build_norm_layer
from up.utils.model.initializer import init_weights_msra, init_bias_constant

__all__ = ['CenterHead']


class SeparateHead(nn.Module):
    def __init__(self, inplanes, sep_head_dict, init_bias=-2.19, use_bias=False, normalize={'type': 'solo_bn'}):
        super(SeparateHead, self).__init__()
        self.prefix = self.__class__.__name__

        if isinstance(inplanes, list):
            inplanes_length = len(inplanes)
            for i in range(1, inplanes_length):
                if inplanes[i] != inplanes[0]:
                    raise ValueError('list inplanes elements are inconsistent with {}'.format(inplanes[i]))
            inplanes = inplanes[0]
        assert isinstance(inplanes, int)
        self.inplanes = inplanes
        self.sep_head_dict = sep_head_dict

        for cur_name in self.sep_head_dict:
            output_channels = self.sep_head_dict[cur_name]['out_channels']
            num_conv = self.sep_head_dict[cur_name]['num_conv']

            fc_list = []
            for k in range(num_conv - 1):
                fc_list.append(nn.Sequential(
                    nn.Conv2d(self.inplanes, self.inplanes, kernel_size=3, stride=1, padding=1, bias=use_bias),
                    build_norm_layer(self.inplanes, normalize)[1],
                    nn.ReLU()
                ))
            fc_list.append(nn.Conv2d(self.inplanes, output_channels, kernel_size=3, stride=1, padding=1, bias=True))
            fc = nn.Sequential(*fc_list)

            if 'hm' in cur_name:
                # fc[-1].bias.data.fill_(init_bias)
                init_bias_constant(fc[-1], init_bias)
            else:
                init_weights_msra(fc)
            self.__setattr__(cur_name, fc)

    def forward(self, x):
        ret_dict = {}
        for cur_name in self.sep_head_dict.keys():
            ret_dict[cur_name] = self.__getattr__(cur_name)(x)
        return ret_dict


@MODULE_ZOO_REGISTRY.register('center_head')
class CenterHead(nn.Module):
    def __init__(self, inplanes, class_names, shared_conv_channel, num_hm_conv,
                 sep_head_dict, use_bias_before_norm=True, normalize={'type': 'solo_bn'}):
        super().__init__()
        self.shared_conv = nn.Sequential(
            nn.Conv2d(
                inplanes, shared_conv_channel, 3, stride=1, padding=1,
                bias=use_bias_before_norm,
            ),
            build_norm_layer(shared_conv_channel, normalize)[1],
            nn.ReLU(),
        )
        self.heads_list = nn.ModuleList()
        hm_head = {'out_channels': len(class_names), 'num_conv': num_hm_conv}
        sep_head_dict.update({'hm': hm_head})
        self.heads_list.append(
            SeparateHead(
                inplanes=shared_conv_channel,
                sep_head_dict=sep_head_dict,
                init_bias=-2.19,
                use_bias=use_bias_before_norm
            )
        )

    def forward(self, input):
        spatial_features_2d = input['spatial_features_2d']
        x = self.shared_conv(spatial_features_2d)
        pred_dict = {}
        for head in self.heads_list:
            pred_dict.update(head(x))
        return {'pred_dict': pred_dict}
