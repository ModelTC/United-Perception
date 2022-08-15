# Import from third library
import torch
import torch.nn as nn
import torch.nn.functional as F

from up.utils.model.normalize import build_conv_norm
from up.utils.model.initializer import initialize_from_cfg
from up.utils.general.registry_factory import MODULE_ZOO_REGISTRY

__all__ = ['SSDNeck']

@MODULE_ZOO_REGISTRY.register('SSDNeck')
class SSDNeck(nn.Module):
    def __init__(self,
                 inplanes,
                 outplanes,
                 out_strides,
                 extern_level,
                 extern_outplanes,
                 normalize=None,
                 initializer=None):

        super(SSDNeck, self).__init__()

        self.inplanes = inplanes
        self.outplanes = outplanes
        self.extern_outplanes = extern_outplanes
        self.outstrides = out_strides
        self.extern_level = extern_level
        self.normalize = normalize

        self.add_extras()
        initialize_from_cfg(self, initializer)

    def add_extras(self):
        # Extra layers
        in_channels = self.inplanes[-1]#用最后一个feature的out channel做此层的输入channel
        cfg = self.extern_outplanes
        normalize = self.normalize

        flag = False
        extern_layer_idx = 0
        for k, v in enumerate(cfg):
            if in_channels != 'S':
                if v == 'S':
                    self.add_module(
                        self.get_extern_name(extern_layer_idx),
                        build_conv_norm(in_channels, cfg[k + 1], 
                             kernel_size=(1, 3)[flag], stride=2, padding=1, normalize=normalize)
                        )
                else:
                    self.add_module(
                        self.get_extern_name(extern_layer_idx),
                        build_conv_norm(in_channels, v, 
                             kernel_size=(1, 3)[flag], normalize=normalize)
                        )
                extern_layer_idx = extern_layer_idx + 1
                flag = not flag
            in_channels = v

        assert extern_layer_idx  == 2 * self.extern_level

    def forward(self, input):
        """
        .. note::
            - For SSDNet, get C3-C5

        Arguments:
            - input (:obj:`dict`): output of ``Backbone``

        Returns:
            - out (:obj:`dict`):

        Input example::

            {
                'features': [],
                'strides': []
            }

        Output example::

            {
                'features': [], # list of tenosr
                'strides': []   # list of int
            }
        """
        features = input['features']
        x = features[-1] #backbone_last_layer_out

        for idx in range(self.extern_level):
            x = self.get_extern(idx * 2 + 1)(self.get_extern(idx * 2)(x))
            features.append(x)

        return {'features': features, 'strides': self.get_outstrides()}

    def get_extern_name(self, idx):
        return 'extern_{}'.format(idx)

    def get_extern(self, idx):
        return getattr(self, self.get_extern_name(idx))

    def get_outplanes(self):
        """
        Return:
            - outplanes (:obj:`list` of :obj:`int`)
        """
        return self.outplanes

    def get_outstrides(self):
        return torch.tensor(self.outstrides, dtype=torch.int)
