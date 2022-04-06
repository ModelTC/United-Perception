# Standard Library
import math

# Import from third library
import torch.nn as nn
import torch.nn.functional as F

try:
    import spring.linklink as link
except:  # noqa
    from up.utils.general.fake_linklink import link

# Import from pod
from up.utils.model.initializer import initialize_from_cfg
from up.utils.model.normalize import build_norm_layer

__all__ = ["mobilenetv3"]


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


def conv_bn(inp, oup, stride, activation=nn.ReLU, normalize={'type': 'solo_bn'}):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        build_norm_layer(oup, normalize)[1],
        activation(inplace=True)
    )


def conv_1x1_bn(inp, oup, activation=nn.ReLU, dilation=1, normalize={'type': 'solo_bn'}):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, dilation=dilation, bias=False),
        build_norm_layer(oup, normalize)[1],
        activation(inplace=True)
    )


class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.


class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            Hsigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Identity(nn.Module):
    def __init__(self, channel):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, kernel, stride, exp, se=False, nl='RE', normalize={'type': 'solo_bn'}, dilation=1):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]
        assert kernel in [3, 5]
        padding = (kernel - 1) // 2
        self.use_res_connect = stride == 1 and inp == oup

        if nl == 'RE':
            activation = nn.ReLU
        elif nl == 'HS':
            activation = Hswish
        else:
            raise NotImplementedError

        SELayer = SEModule if se else Identity

        layers = []
        if inp != exp:
            # pw
            layers.extend([
                nn.Conv2d(inp, exp, 1, 1, 0, bias=False),
                build_norm_layer(exp, normalize)[1],
                activation(inplace=True),
            ])
        else:
            print("no need!!!!!!!!!!1")
        layers.extend([
            # dw
            nn.Conv2d(exp, exp, kernel, stride,
                      padding, groups=exp, bias=False, dilation=dilation),
            build_norm_layer(exp, normalize)[1],
            SELayer(exp),
            activation(inplace=True),
            # pw-linear
            nn.Conv2d(exp, oup, 1, 1, 0, bias=False),
            build_norm_layer(oup, normalize)[1],
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3(nn.Module):
    """
    MobileNet V3 main class, based on
    `"Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>`_
    """
    def __init__(self,
                 out_strides,
                 width_mult=1.0,
                 dropout=0.8,
                 round_nearest=8,
                 mode='small',
                 c4_output=False,
                 normalize={'type': 'freeze_bn'},
                 initializer=None):
        r"""
        Arguments:
            - num_classes (:obj:`int`): Number of classes
            - width_mult (:obj:`float`): Width multiplier, adjusts number of channels in each layer by this amount
            - dropout (:obj:`float`): Dropout rate
            - round_nearest (:obj:`int`): Round the number of channels in each layer to be a multiple of this number
              Set to 1 to turn off rounding
            - mode (:obj:`string`): model type, 'samll' or 'large'
            - c4_output: (:obj:`bool`): output stage4 feature with stride 16
            - normalize (:obj:`dict`): Config of Normalization Layer (see Configuration#Normalization).
        """
        super(MobileNetV3, self).__init__()

        self.out_strides = out_strides
        ss = 1
        dilation = 1
        max_stride = max(out_strides)

        input_channel = 16
        last_channel = 1280
        if mode == 'large':
            mobile_setting = [
                [3, 16,  16,  False, 'RE', 1], # noqa
                [3, 64,  24,  False, 'RE', 2], # noqa
                [3, 72,  24,  False, 'RE', 1], # noqa
                [5, 72,  40,  True,  'RE', 2], # noqa
                [5, 120, 40,  True,  'RE', 1], # noqa
                [5, 120, 40,  True,  'RE', 1], # noqa
                [3, 240, 80,  False, 'HS', 2], # noqa
                [3, 200, 80,  False, 'HS', 1], # noqa
                [3, 184, 80,  False, 'HS', 1], # noqa
                [3, 184, 80,  False, 'HS', 1], # noqa
                [3, 480, 112, True,  'HS', 1], # noqa
                [3, 672, 112, True,  'HS', 1], # noqa
                [5, 672, 160, True,  'HS', 2], # noqa
                [5, 960, 160, True,  'HS', 1], # noqa
                [5, 960, 160, True,  'HS', 1], # noqa
            ]
        elif mode == 'small':
            mobile_setting = [
                [3, 16,  16,  True,  'RE', 2], # noqa
                [3, 72,  24,  False, 'RE', 2], # noqa
                [3, 88,  24,  False, 'RE', 1], # noqa
                [5, 96,  40,  True,  'HS', 2], # noqa
                [5, 240, 40,  True,  'HS', 1], # noqa
                [5, 240, 40,  True,  'HS', 1], # noqa
                [5, 120, 48,  True,  'HS', 1], # noqa
                [5, 144, 48,  True,  'HS', 1], # noqa
                [5, 288, 96,  True,  'HS', 2], # noqa
                [5, 576, 96,  True,  'HS', 1], # noqa
                [5, 576, 96,  True,  'HS', 1], # noqa
            ]
        else:
            raise NotImplementedError

        # building first layer
        last_channel = _make_divisible(
            last_channel * width_mult, round_nearest) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2, activation=Hswish, normalize=normalize)]
        ss *= 2
        layer_index = [-1]  # last layer before downsample
        output_channels = [-1]

        # building mobile blocks
        for k, exp, c, se, nl, s in mobile_setting:
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
            output_channel = _make_divisible(c * width_mult, round_nearest)
            exp_channel = _make_divisible(exp * width_mult, round_nearest)
            self.features.append(InvertedResidual(
                input_channel, output_channel, k, s, exp_channel, se, nl, normalize=normalize, dilation=dilation))
            input_channel = output_channel

        # building last several layers
        if mode == 'large':
            last_conv = _make_divisible(960 * width_mult, round_nearest)
            self.features.append(conv_1x1_bn(
                input_channel, last_conv, activation=Hswish, normalize=normalize))
            self.features.append(nn.AdaptiveAvgPool2d(1))
            self.features.append(nn.Conv2d(last_conv, last_channel, 1, 1, 0))
            self.features.append(Hswish(inplace=True))
            output_channels.append(last_channel)
            layer_index.append(len(self.features) - 1)
        elif mode == 'small':
            last_conv = _make_divisible(576 * width_mult, round_nearest)
            self.features.append(conv_1x1_bn(
                input_channel, last_conv, activation=Hswish, normalize=normalize))
            self.features.append(nn.AdaptiveAvgPool2d(1))
            self.features.append(nn.Conv2d(last_conv, last_channel, 1, 1, 0))
            self.features.append(Hswish(inplace=True))
            output_channels.append(last_channel)
            layer_index.append(len(self.features) - 1)
        else:
            raise NotImplementedError
        self.features = nn.Sequential(*self.features)
        if not c4_output:
            self.out_planes = [output_channels[int(math.log(s, 2))] for s in self.out_strides]
            self.out_layer_index = [layer_index[int(math.log(s, 2))] for s in self.out_strides]
        else:
            self.out_planes = [self.last_channel]
            self.out_layer_index = layer_index[-1:]
        if initializer is not None:
            initialize_from_cfg(self, initializer)
        else:
            self.init_params()

    def get_outplanes(self):
        """
        """
        return self.out_planes

    def get_outstrides(self):
        return self.out_strides

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, link.nn.SyncBatchNorm2d) or isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

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


def mobilenetv3(**kwargs):
    """
    Constructs a MobileNet-V3 model.
    """
    model = MobileNetV3(**kwargs)
    return model
