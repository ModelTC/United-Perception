# Import from third library
import torch
import torch.nn as nn

from eod.utils.model.initializer import initialize_from_cfg
from eod.utils.general.registry_factory import MODULE_ZOO_REGISTRY
from ..components import ConvBnAct, Bottleneck, Focus, BottleneckCSP, SPP

__all__ = [
    'DarkNetv5', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x'
]


@MODULE_ZOO_REGISTRY.register('DarkNetv5')
class DarkNetv5(nn.Module):
    def __init__(self,
                 init_planes,
                 csp1_block_nums,
                 in_planes=3,
                 focus_type='v4_focus',
                 out_layers=[2, 3, 4],
                 out_strides=[8, 16, 32],
                 normalize={'type': 'solo_bn'},
                 act_fn={'type': 'Hardswish'},
                 initializer=None):
        super(DarkNetv5, self).__init__()

        assert init_planes % 8 == 0, 'init_planes must be a multiple of 8'

        csp1_block_nums_length = len(csp1_block_nums)
        assert 4 <= csp1_block_nums_length <= 7, 'csp1_block_nums length must be greater than 4 and less than 7'

        plane = init_planes
        self.out_planes = []

        if focus_type == 'v5_focus':
            self.focus = Focus(in_planes, plane, kernel_size=3, normalize=normalize, act_fn=act_fn)
        elif focus_type == 'v4_focus':
            # for focus replace
            self.focus = nn.Sequential(ConvBnAct(in_planes, plane // 2, 3, normalize=normalize, act_fn=act_fn),
                                       ConvBnAct(plane // 2, plane, 3, 2, normalize=normalize, act_fn=act_fn),
                                       Bottleneck(plane, plane, normalize=normalize, act_fn=act_fn))
        elif focus_type == 'stem_focus':
            self.focus = nn.Sequential(ConvBnAct(in_planes, plane // 2, 3, 2, normalize=normalize, act_fn=act_fn),
                                       ConvBnAct(plane // 2, plane // 2, 3, normalize=normalize, act_fn=act_fn),
                                       nn.Conv2d(plane // 2, plane, kernel_size=3, stride=1, padding=1, bias=False))
        else:
            raise NotImplementedError
        self.out_planes.append(plane)

        self.csp_modules = nn.ModuleList()
        for i in range(csp1_block_nums_length):
            layers = []
            if i < 4:
                conv_block = ConvBnAct(plane, plane * 2, 3, 2, normalize=normalize, act_fn=act_fn)
                plane *= 2
            else:
                conv_block = ConvBnAct(plane, plane, 3, 2, normalize=normalize, act_fn=act_fn)
            layers.append(conv_block)

            if i == csp1_block_nums_length - 1:
                # last backbone layer
                spp = SPP(plane, plane, normalize=normalize, act_fn=act_fn)
                csp2_1 = BottleneckCSP(plane, plane, nums=csp1_block_nums[i],
                                       shortcut=False, normalize=normalize, act_fn=act_fn)
                layers.append(spp)
                layers.append(csp2_1)
            else:
                csp_blocks = BottleneckCSP(plane, plane, nums=csp1_block_nums[i], normalize=normalize, act_fn=act_fn)
                layers.append(csp_blocks)

            self.csp_modules.append(nn.Sequential(*layers))
            self.out_planes.append(plane)

        self.out_layers = out_layers
        self.out_strides = out_strides
        self.out_planes = [self.out_planes[i] for i in self.out_layers]

        # Init weights, biases
        if initializer is not None:
            initialize_from_cfg(self, initializer)

    def get_outplanes(self):
        """
        get dimension of the output tensor
        """
        return self.out_planes

    def forward(self, x):
        x = x['image']
        features = []
        x = self.focus(x)   # P1/2
        features.append(x)
        for i, m in enumerate(self.csp_modules):
            x = m(x)
            features.append(x)   # P2/4 ~ P5/32
        features = [features[_] for _ in self.out_layers]
        return {'features': features, 'strides': self.get_outstrides()}

    def get_outstrides(self):
        return torch.tensor(self.out_strides, dtype=torch.int)


@MODULE_ZOO_REGISTRY.register('yolov5s')
def yolov5s(**kwargs):
    """
    Constructs a yolov5s model.

    Arguments:
    """
    model = DarkNetv5(32, [1, 3, 3, 1], **kwargs)
    return model


@MODULE_ZOO_REGISTRY.register('yolov5m')
def yolov5m(**kwargs):
    """
    Constructs a yolov5s model.

    Arguments:
    """
    model = DarkNetv5(48, [2, 6, 6, 2], **kwargs)
    return model


@MODULE_ZOO_REGISTRY.register('yolov5l')
def yolov5l(**kwargs):
    """
    Constructs a yolov5s model.

    Arguments:
    """
    model = DarkNetv5(64, [3, 9, 9, 3], **kwargs)
    return model


@MODULE_ZOO_REGISTRY.register('yolov5x')
def yolov5x(**kwargs):
    """
    Constructs a yolov5s model.

    Arguments:
    """
    model = DarkNetv5(80, [4, 12, 12, 4], **kwargs)
    return model
