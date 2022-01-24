# Standard Library
import math

# Import from third library
import torch.nn as nn

# Import from pod
from up.utils.model.initializer import initialize_from_cfg
from up.utils.model.normalize import build_norm_layer

__all__ = ["mobilenetv2"]


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_bn(inp, oup, stride, normalize):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        build_norm_layer(oup, normalize)[1],
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup, dilation=1, normalize={'type': 'solo_bn'}):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, dilation=dilation, bias=False),
        build_norm_layer(oup, normalize)[1],
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, dilation=1, normalize={'type': 'solo_bn'}):
        super(InvertedResidual, self).__init__()
        self.stride = stride

        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            build_norm_layer(inp * expand_ratio, normalize)[1],
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(inp * expand_ratio,
                      inp * expand_ratio,
                      3,
                      stride,
                      padding=dilation,
                      groups=inp * expand_ratio,
                      bias=False,
                      dilation=dilation),
            build_norm_layer(inp * expand_ratio, normalize)[1],
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            build_norm_layer(oup, normalize)[1],
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNet(nn.Module):
    """
    """
    def __init__(self,
                 out_strides,
                 width_mult=1.0,
                 round_nearest=8,
                 c4_output=False,
                 normalize={'type': 'freeze_bn'},
                 initializer=None):
        """
        Arguments:
            - out_strides (:obj:`list`): strides of output features
            - width_multi (:obj:`float`): width factor of the network
            - c4_output: (:obj:`bool`): output stage4 feature with stride 16
            - normalize (:obj:`dict`): Config of Normalization Layer (see Configuration#Normalization).
        """
        super(MobileNet, self).__init__()

        self.out_strides = out_strides
        ss = 1
        dilation = 1
        max_stride = max(out_strides)
        self.interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        input_channel = _make_divisible(32 * width_mult, round_nearest)
        self.last_channel = _make_divisible(1280 * max(1.0, width_mult), round_nearest)
        ss *= 2
        self.features = [conv_bn(3, input_channel, 2, normalize=normalize)]
        layer_index = [-1]  # last layer before downsample
        output_channels = [-1]

        # building inverted residual blocks
        for t, c, n, s in self.interverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            if s == 2:
                if ss >= max_stride:
                    if c4_output:
                        dilation = 2
                        s = 1
                    else:
                        break
                else:
                    layer_index.append(len(self.features) - 1)
                    output_channels.append(input_channel)
                    ss *= 2
            for i in range(n):
                if i == 0:
                    self.features.append(InvertedResidual(input_channel,
                                                          output_channel,
                                                          s,
                                                          t,
                                                          dilation=dilation,
                                                          normalize=normalize))
                else:
                    self.features.append(InvertedResidual(input_channel,
                                                          output_channel,
                                                          1,
                                                          t,
                                                          normalize=normalize))
                input_channel = output_channel
        if ss == max_stride or c4_output:
            # building last several layers
            self.features.append(conv_1x1_bn(input_channel, self.last_channel, normalize=normalize))
            layer_index.append(len(self.features) - 1)
            output_channels.append(self.last_channel)

        self.features = nn.Sequential(*self.features)
        if not c4_output:
            self.out_planes = [output_channels[int(math.log(s, 2))] for s in self.out_strides]
            self.out_layer_index = [layer_index[int(math.log(s, 2))] for s in self.out_strides]
        else:
            self.out_planes = [self.last_channel]
            self.out_layer_index = layer_index[-1:]

        if initializer is not None:
            initialize_from_cfg(self, initializer)

    def get_outplanes(self):
        """
        """
        return self.out_planes

    def get_outstrides(self):
        return self.out_strides

    def forward(self, input):
        """
        """
        x = input['image']
        output = []
        k = 0
        for m_idx, module in enumerate(self.features._modules.values()):
            x = module(x)
            if k < len(self.out_layer_index) and m_idx == self.out_layer_index[k]:
                output.append(x)
                k += 1
        assert len(output) == len(self.out_layer_index), '{} vs {}'.format(len(output), len(self.out_layer_index))
        return {'features': output, 'strides': self.get_outstrides()}


def mobilenetv2(**kwargs):
    model = MobileNet(**kwargs)
    return model
