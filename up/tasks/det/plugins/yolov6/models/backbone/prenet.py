import math
import torch.nn as nn
from up.utils.general.registry_factory import MODULE_ZOO_REGISTRY

__all__ = ["Yolov6Pre"]


@MODULE_ZOO_REGISTRY.register('yolov6_pre')
class Yolov6Pre(nn.Module):
    def __init__(
        self,
        depth_multiple=0.33,
        width_multiple=0.50,
        backbone_base_repeats=[1, 6, 12, 18, 6],
        backbone_base_channels=[64, 128, 256, 512, 1024],
        neck_base_repeats=[12, 12, 12, 12],
        neck_base_channels=[256, 128, 128, 256, 256, 512]
    ):
        super(Yolov6Pre, self).__init__()
        self.num_repeats = [
            (max(round(i * depth_multiple), 1) if i > 1 else i) for i in (backbone_base_repeats + neck_base_repeats)
        ]
        self.channels_list = [
            math.ceil(i * width_multiple / 8) * 8 for i in (backbone_base_channels + neck_base_channels)
        ]

    def get_outplanes(self):
        '''
        Return the out channels the backbone and neck need
        '''
        return self.channels_list

    def get_repeats(self):
        return self.num_repeats

    def forward(self, x):
        return x
