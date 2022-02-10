# Import from third library
import torch
import torch.nn as nn

from eod.utils.model.initializer import initialize_from_cfg
from eod.utils.general.registry_factory import MODULE_ZOO_REGISTRY
from eod.tasks.det.plugins.yolov5.models.components import ConvBnAct, Focus, SPP
from eod.tasks.det.plugins.yolov5.models.components import Bottleneck as Bottleneckv5
from eod.utils.model.normalize import build_conv_norm

__all__ = [
    'yolox_l', 'yolox_m', 'yolox_s', 'yolox_x', 'yolox_tiny', 'yolox_dyn'
]


class SPP_dila(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self,
                 in_planes,
                 out_planes,
                 kernels=(5, 9, 13),
                 normalize={'type': 'solo_bn'},
                 act_fn={'type': 'Hardswish'},
                 dilations=(1, 3, 5)):
        super(SPP_dila, self).__init__()
        hidden_planes = in_planes // 2  # hidden channels
        self.conv_block1 = ConvBnAct(in_planes, hidden_planes, 1, 1, normalize=normalize, act_fn=act_fn)
        self.conv_block2 = ConvBnAct(hidden_planes * (len(kernels) + 1), out_planes, 1, 1,
                                     normalize=normalize, act_fn=act_fn)
        self.pooling_blocks = nn.ModuleList([build_conv_norm(
            hidden_planes, hidden_planes, 3,
            padding=d, dilation=d, normalize=normalize, activation=True) for d in dilations])

    def forward(self, x):
        x = self.conv_block1(x)
        x = torch.cat([x] + [m(x) for m in self.pooling_blocks], 1)
        x = self.conv_block2(x)
        return x


class DWConv(nn.Module):
    """Depthwise Conv + Conv"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, act_fn="silu", normalize={'type': 'solo_bn'}):
        super().__init__()
        self.dconv = ConvBnAct(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=in_channels,
            act_fn=act_fn,
            normalize=normalize
        )
        self.pconv = ConvBnAct(
            in_channels, out_channels, kernel_size=1, stride=1, groups=1, act_fn=act_fn, normalize=normalize
        )

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(
        self, in_channels, out_channels, shortcut=True,
        expansion=0.5, depthwise=False, act="Silu", normalize={'type': 'solo_bn'}
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        # BaseConv
        Conv = DWConv if depthwise else ConvBnAct
        self.conv1 = ConvBnAct(in_channels, hidden_channels, 1, stride=1, act_fn=act, normalize=normalize)
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act_fn=act, normalize=normalize)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


class CSPLayer(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""

    def __init__(
        self, in_channels, out_channels, n=1,
        shortcut=True, expansion=0.5, depthwise=False, act="Silu", normalize={'type': 'solo_bn'}
    ):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = ConvBnAct(in_channels, hidden_channels, 1, stride=1, act_fn=act, normalize=normalize)
        self.conv2 = ConvBnAct(in_channels, hidden_channels, 1, stride=1, act_fn=act, normalize=normalize)
        self.conv3 = ConvBnAct(2 * hidden_channels, out_channels, 1, stride=1, act_fn=act, normalize=normalize)
        module_list = [
            Bottleneck(hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act, normalize=normalize)
            for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)


class CSPDarknet(nn.Module):
    def __init__(self,
                 dep_mul,
                 wid_mul,
                 out_layers=[2, 3, 4],
                 out_strides=[8, 16, 32],
                 normalize={'type': 'solo_bn'},
                 act_fn={'type': 'Silu'},
                 depthwise=False,
                 initializer=None,
                 use_spp_pool=True,
                 input_channels=3,
                 focus_type='v5_focus'):
        super(CSPDarknet, self).__init__()
        Conv = DWConv if depthwise else ConvBnAct  # BaseConv
        self.use_spp_pool = use_spp_pool

        base_channels = int(wid_mul * 64)  # 64
        base_depth = max(round(dep_mul * 3), 1)  # 3

        self.out_planes = []
        # stem
        if focus_type == 'v5_focus':
            self.stem = Focus(input_channels, base_channels,
                              kernel_size=3, normalize=normalize, act_fn=act_fn)
        elif focus_type == 'v4_focus':
            self.stem = nn.Sequential(
                ConvBnAct(
                    input_channels, base_channels // 2, 3,
                    normalize=normalize, act_fn=act_fn),
                ConvBnAct(
                    base_channels // 2, base_channels, 3, 2,
                    normalize=normalize, act_fn=act_fn),
                Bottleneckv5(base_channels, base_channels, normalize=normalize, act_fn=act_fn))
        elif focus_type == 'stem_focus':
            self.stem = nn.Sequential(
                ConvBnAct(
                    input_channels, base_channels // 2, 3, 2,
                    normalize=normalize, act_fn=act_fn),
                ConvBnAct(
                    base_channels // 2, base_channels // 2, 3,
                    normalize=normalize, act_fn=act_fn),
                ConvBnAct(
                    base_channels // 2, base_channels, kernel_size=3,
                    normalize=normalize, act_fn=act_fn, stride=1))
        self.out_planes.append(base_channels)

        # dark2
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, kernel_size=3, stride=2, normalize=normalize, act_fn=act_fn),
            CSPLayer(
                base_channels * 2, base_channels * 2,
                n=base_depth, depthwise=depthwise, act=act_fn, normalize=normalize
            )
        )
        self.out_planes.append(base_channels * 2)

        # dark3
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, normalize=normalize, act_fn=act_fn),
            CSPLayer(
                base_channels * 4, base_channels * 4,
                n=base_depth * 3, depthwise=depthwise, act=act_fn, normalize=normalize
            )
        )
        self.out_planes.append(base_channels * 4)

        # dark4
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, kernel_size=3, stride=2, normalize=normalize, act_fn=act_fn),
            CSPLayer(
                base_channels * 8, base_channels * 8,
                n=base_depth * 3, depthwise=depthwise, act=act_fn, normalize=normalize
            )
        )
        self.out_planes.append(base_channels * 8)

        # dark5
        self.dark5 = None
        if not self.use_spp_pool:
            self.dark5 = nn.Sequential(
                Conv(
                    base_channels * 8,
                    base_channels * 16,
                    kernel_size=3,
                    stride=2,
                    normalize=normalize,
                    act_fn=act_fn),
                SPP_dila(base_channels * 16, base_channels * 16, act_fn=act_fn, normalize=normalize),
                CSPLayer(
                    base_channels * 16, base_channels * 16, n=base_depth,
                    shortcut=False, depthwise=depthwise, act=act_fn, normalize=normalize
                )
            )
        else:
            self.dark5 = nn.Sequential(
                Conv(
                    base_channels * 8,
                    base_channels * 16,
                    kernel_size=3,
                    stride=2,
                    normalize=normalize,
                    act_fn=act_fn),
                SPP(base_channels * 16, base_channels * 16, act_fn=act_fn, normalize=normalize),
                CSPLayer(
                    base_channels * 16, base_channels * 16, n=base_depth,
                    shortcut=False, depthwise=depthwise, act=act_fn, normalize=normalize
                )
            )
        self.out_planes.append(base_channels * 16)

        self.out_layers = out_layers
        self.out_strides = out_strides
        self.out_planes = [self.out_planes[i] for i in self.out_layers]

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
        # stem
        x = self.stem(x)
        features.append(x)
        # dark2
        x = self.dark2(x)
        features.append(x)
        # dark3
        x = self.dark3(x)
        features.append(x)
        # dark4
        x = self.dark4(x)
        features.append(x)
        # dark5
        x = self.dark5(x)
        features.append(x)

        # [dark3, dark4, dark5]
        features = [features[_] for _ in self.out_layers]
        return {'features': features, 'strides': self.get_outstrides()}

    def get_outstrides(self):
        return torch.tensor(self.out_strides, dtype=torch.int)


@MODULE_ZOO_REGISTRY.register('yolox')
def yolox_dyn(**kwargs):
    """
    Constructs a yolox_tiny model.

    Arguments:
    """
    model = CSPDarknet(**kwargs)
    return model


@MODULE_ZOO_REGISTRY.register('yolox_tiny')
def yolox_tiny(**kwargs):
    """
    Constructs a yolox_tiny model.

    Arguments:
    """
    model = CSPDarknet(dep_mul=0.33, wid_mul=0.375, **kwargs)
    return model


@MODULE_ZOO_REGISTRY.register('yolox_s')
def yolox_s(**kwargs):
    """
    Constructs a yolox_tiny model.

    Arguments:
    """
    model = CSPDarknet(dep_mul=0.33, wid_mul=0.50, **kwargs)
    return model


@MODULE_ZOO_REGISTRY.register('yolox_m')
def yolox_m(**kwargs):
    """
    Constructs a yolox_tiny model.

    Arguments:
    """
    model = CSPDarknet(dep_mul=0.67, wid_mul=0.75, **kwargs)
    return model


@MODULE_ZOO_REGISTRY.register('yolox_l')
def yolox_l(**kwargs):
    """
    Constructs a yolox_tiny model.

    Arguments:
    """
    model = CSPDarknet(dep_mul=1.0, wid_mul=1.0, **kwargs)
    return model


@MODULE_ZOO_REGISTRY.register('yolox_x')
def yolox_x(**kwargs):
    """
    Constructs a yolox_tiny model.

    Arguments:
    """
    model = CSPDarknet(dep_mul=1.33, wid_mul=1.25, **kwargs)
    return model


@MODULE_ZOO_REGISTRY.register('yolox_nano')
def yolox_nano(**kwargs):
    """
    Constructs a yolox_tiny model.

    Arguments:
    """
    model = CSPDarknet(dep_mul=0.33, wid_mul=0.25, **kwargs)
    return model
