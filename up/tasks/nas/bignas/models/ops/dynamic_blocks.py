import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from collections import OrderedDict

from up.tasks.nas.bignas.utils.registry_factory import DYNAMIC_BLOCKS_REGISTRY
# from up.utils.model.initializer import trunc_normal_

from up.tasks.nas.bignas.models.ops.dynamic_utils import int2list, build_activation, make_divisible, \
    get_middle_channel, get_fake_input_for_module

from up.tasks.nas.bignas.models.ops.dynamic_ops import DynamicConv2d, DynamicLinear, \
    Identity, build_dynamic_norm_layer


@DYNAMIC_BLOCKS_REGISTRY.register('DynamicLinearBlock')
class DynamicLinearBlock(nn.Module):

    def __init__(self, in_features_list, out_features_list, bias=True, dropout_rate=0, act_func=''):
        super(DynamicLinearBlock, self).__init__()

        self.in_features_list = int2list(in_features_list)
        self.out_features_list = int2list(out_features_list)
        self.bias = bias
        self.dropout_rate = dropout_rate
        self.act_func = act_func

        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(self.dropout_rate, inplace=True)
        else:
            self.dropout = None
        self.linear = DynamicLinear(
            in_features=max(self.in_features_list), out_features=max(self.out_features_list), bias=self.bias
        )
        if act_func != '':
            self.act = build_activation(self.act_func, inplace=False)
        self.active_out_channel = max(self.out_features_list)
        self.active_in_channel = max(self.in_features_list)
        self.active_block = True

    def forward(self, x):
        if not self.active_block:
            return x
        self.linear.active_out_channel = self.active_out_channel
        self.active_in_channel = x.size(1)
        self.active_out_channel = min(self.active_out_channel, max(self.out_features_list))

        if self.dropout is not None:
            x = self.dropout(x)
        x = self.linear(x)
        if self.act_func != '':
            x = self.act(x)
        return x

    @property
    def module_str(self):
        return 'DyLinear(I_%d, O_%d)' % (self.active_in_channel, self.active_out_channel)

    def get_fake_input(self, input_shape=None):
        if not input_shape:
            input_shape = [1, self.active_in_channel]
        else:
            input_shape = input_shape
        return get_fake_input_for_module(self, input_shape)


@DYNAMIC_BLOCKS_REGISTRY.register('DynamicStemBlock')
class DynamicStemBlock(nn.Module):

    def __init__(self, in_channel_list, out_channel_list,
                 kernel_size_list=3, expand_ratio_list=0.5, stride=1, act_func='relu',
                 KERNEL_TRANSFORM_MODE=False, divisor=8,
                 normalize={'type': 'dynamic_solo_bn'}):
        super(DynamicStemBlock, self).__init__()

        self.in_channel_list = int2list(in_channel_list)
        self.out_channel_list = int2list(out_channel_list)

        self.kernel_size_list = int2list(kernel_size_list)
        self.expand_ratio_list = int2list(expand_ratio_list)

        self.stride = stride
        self.act_func = act_func

        self.KERNEL_TRANSFORM_MODE = True if KERNEL_TRANSFORM_MODE and len(self.kernel_size_list) > 1 else False
        self.divisor = divisor

        norm_layer1 = norm_layer2 = build_dynamic_norm_layer(normalize)
        if 'switch' in normalize['type']:
            middle_channel = get_middle_channel(self.out_channel_list, self.expand_ratio_list, self.divisor)
            norm_layer1 = partial(norm_layer1, width_list=middle_channel)
            norm_layer2 = partial(norm_layer2, width_list=self.out_channel_list)

        # build modules
        max_middle_channel = make_divisible(max(self.out_channel_list) * max(self.expand_ratio_list), self.divisor)
        self.point_conv1 = nn.Sequential(OrderedDict([
            ('conv', DynamicConv2d(max(self.in_channel_list), max_middle_channel, kernel_size=self.kernel_size_list,
                                   stride=self.stride, groups=1,
                                   KERNEL_TRANSFORM_MODE=KERNEL_TRANSFORM_MODE)),
            ('bn', norm_layer1(max_middle_channel)),
            ('act', build_activation(self.act_func, inplace=False)),
        ]))

        self.normal_conv = nn.Sequential(OrderedDict([
            ('conv',
             DynamicConv2d(max_middle_channel, max_middle_channel, self.kernel_size_list, groups=1,
                           KERNEL_TRANSFORM_MODE=KERNEL_TRANSFORM_MODE)),
            ('bn', norm_layer1(max_middle_channel)),
            ('act', build_activation(self.act_func, inplace=False))
        ]))

        self.point_conv2 = nn.Sequential(OrderedDict([
            ('conv', DynamicConv2d(max_middle_channel, max(self.out_channel_list), kernel_size=self.kernel_size_list,
                                   groups=1, KERNEL_TRANSFORM_MODE=KERNEL_TRANSFORM_MODE)),
            ('bn', norm_layer2(max(self.out_channel_list))),
            ('act', build_activation(self.act_func, inplace=False))
        ]))

        self.active_kernel_size = max(self.kernel_size_list)
        self.active_expand_ratio = max(self.expand_ratio_list)
        self.active_out_channel = max(self.out_channel_list)
        self.active_in_channel = max(self.in_channel_list)
        self.active_block = True

    def forward(self, x):
        if not self.active_block:
            return x
        in_channel = x.size(1)
        self.active_in_channel = in_channel
        self.active_out_channel = min(self.active_out_channel, max(self.out_channel_list))

        mid_channel = make_divisible(round(self.active_out_channel * self.active_expand_ratio), self.divisor)

        self.point_conv1.conv.active_out_channel = mid_channel

        self.normal_conv.conv.active_kernel_size = self.active_kernel_size
        self.normal_conv.conv.active_out_channel = mid_channel

        self.point_conv2.conv.active_out_channel = self.active_out_channel

        x = self.point_conv1(x)
        x = self.normal_conv(x)
        x = self.point_conv2(x)

        return x

    @property
    def module_str(self):
        if self.active_in_channel == self.active_out_channel and self.stride == 1:
            return 'StemBlock(I_%d, O_%d, E_%.2f, K_%d, S_%d, Transform_%s)' % (
                self.active_in_channel, self.active_out_channel, self.active_expand_ratio, self.active_kernel_size,
                self.stride, self.KERNEL_TRANSFORM_MODE)
        else:
            return 'StemBlock(I_%d, O_%d, E_%.2f, K_%d, S_%d, Transform_%s)' % (
                self.active_in_channel, self.active_out_channel, self.active_expand_ratio, self.active_kernel_size,
                self.stride, self.KERNEL_TRANSFORM_MODE)

    def get_fake_input(self, input_shape=None):
        if not input_shape:
            input_shape = [1, self.active_in_channel, 56, 56]
        else:
            input_shape = input_shape
        return get_fake_input_for_module(self, input_shape)


@DYNAMIC_BLOCKS_REGISTRY.register('DynamicBasicBlock')
class DynamicBasicBlock(nn.Module):

    def __init__(self, in_channel_list, out_channel_list,
                 kernel_size_list=3, expand_ratio_list=1, stride=1, act_func='relu',
                 KERNEL_TRANSFORM_MODE=False, divisor=8, IS_STAGE_BLOCK=False,
                 normalize={'type': 'dynamic_solo_bn'}):
        super(DynamicBasicBlock, self).__init__()

        self.in_channel_list = int2list(in_channel_list)
        self.out_channel_list = int2list(out_channel_list)

        self.kernel_size_list = int2list(kernel_size_list)
        self.expand_ratio_list = int2list(expand_ratio_list)

        self.stride = stride
        self.act_func = act_func

        self.KERNEL_TRANSFORM_MODE = True if KERNEL_TRANSFORM_MODE and len(self.kernel_size_list) > 1 else False
        self.divisor = divisor

        norm_layer1 = norm_layer2 = build_dynamic_norm_layer(normalize)
        if 'switch' in normalize['type']:
            middle_channel = get_middle_channel(self.out_channel_list, self.expand_ratio_list, self.divisor)
            norm_layer1 = partial(norm_layer1, width_list=middle_channel)
            norm_layer2 = partial(norm_layer2, width_list=self.out_channel_list)

        # build modules default is 1
        max_middle_channel = make_divisible(max(self.out_channel_list) * max(self.expand_ratio_list), self.divisor)
        self.normal_conv1 = nn.Sequential(OrderedDict([
            ('conv', DynamicConv2d(max(self.in_channel_list), max_middle_channel, self.kernel_size_list,
                                   stride=self.stride, groups=1, KERNEL_TRANSFORM_MODE=KERNEL_TRANSFORM_MODE)),
            ('bn', norm_layer1(max_middle_channel)),
            ('act', build_activation(self.act_func, inplace=False))
        ]))

        self.normal_conv2 = nn.Sequential(OrderedDict([
            ('conv',
             DynamicConv2d(max_middle_channel, max(self.out_channel_list), self.kernel_size_list, groups=1,
                           KERNEL_TRANSFORM_MODE=KERNEL_TRANSFORM_MODE)),
            ('bn', norm_layer2(max(self.out_channel_list))),
        ]))

        self.active_kernel_size = max(self.kernel_size_list)
        self.active_expand_ratio = max(self.expand_ratio_list)
        self.active_out_channel = max(self.out_channel_list)
        self.active_in_channel = max(self.in_channel_list)
        self.active_block = True

        if max(self.in_channel_list) == max(self.out_channel_list) and self.stride == 1 and not IS_STAGE_BLOCK:
            self.shortcut = Identity()
        else:
            self.shortcut = DynamicConv2d(max(self.in_channel_list), max(self.out_channel_list), kernel_size=1,
                                          groups=1, stride=stride, KERNEL_TRANSFORM_MODE=KERNEL_TRANSFORM_MODE)
            self.shortcutbn = norm_layer2(max(self.out_channel_list))
        self.act2 = build_activation(self.act_func, inplace=False)

    def forward(self, x):
        if not self.active_block:
            return x
        identity = x
        in_channel = x.size(1)
        self.active_in_channel = in_channel
        self.active_out_channel = min(self.active_out_channel, max(self.out_channel_list))

        mid_channel = make_divisible(round(self.active_out_channel * self.active_expand_ratio), self.divisor)
        self.normal_conv1.conv.active_kernel_size = self.active_kernel_size
        self.normal_conv1.conv.active_out_channel = mid_channel

        self.normal_conv2.conv.active_kernel_size = self.active_kernel_size
        self.normal_conv2.conv.active_out_channel = self.active_out_channel

        x = self.normal_conv1(x)
        x = self.normal_conv2(x)
        if in_channel == self.active_out_channel and self.stride == 1:
            x += identity
        else:
            self.shortcut.active_out_channel = self.active_out_channel
            x += self.shortcutbn(self.shortcut(identity))
        return self.act2(x)

    @property
    def module_str(self):
        if self.active_in_channel == self.active_out_channel and self.stride == 1:
            return 'Basic(I_%d, O_%d, E_%.2f, K_%d, S_%d, Shortcut_%s, Transform_%s)' % (
                self.active_in_channel, self.active_out_channel, self.active_expand_ratio, self.active_kernel_size,
                self.stride, 'identity', self.KERNEL_TRANSFORM_MODE)
        else:
            return 'Basic(I_%d, O_%d, E_%.2f, K_%d, S_%d, Shortcut_%s, Transform_%s)' % (
                self.active_in_channel, self.active_out_channel, self.active_expand_ratio, self.active_kernel_size,
                self.stride, 'DyConv', self.KERNEL_TRANSFORM_MODE)

    def get_fake_input(self, input_shape=None):
        if not input_shape:
            input_shape = [1, self.active_in_channel, 56, 56]
        else:
            input_shape = input_shape
        return get_fake_input_for_module(self, input_shape)


@DYNAMIC_BLOCKS_REGISTRY.register('DynamicConvBlock')
class DynamicConvBlock(nn.Module):

    def __init__(self, in_channel_list, out_channel_list, kernel_size_list=3, stride=1, dilation=1,
                 use_bn=True, act_func='relu', KERNEL_TRANSFORM_MODE=False, bias=False,
                 normalize={'type': 'dynamic_solo_bn'}):
        super(DynamicConvBlock, self).__init__()

        self.in_channel_list = int2list(in_channel_list)
        self.out_channel_list = int2list(out_channel_list)
        self.kernel_size_list = int2list(kernel_size_list)
        self.stride = stride
        self.dilation = dilation
        self.use_bn = use_bn
        self.act_func = act_func
        self.KERNEL_TRANSFORM_MODE = True if KERNEL_TRANSFORM_MODE and len(self.kernel_size_list) > 1 else False

        if normalize is None:
            self.use_bn = False

        self.conv = DynamicConv2d(
            in_channels=max(self.in_channel_list), out_channels=max(self.out_channel_list),
            kernel_size=self.kernel_size_list, groups=1, stride=self.stride, dilation=self.dilation,
            KERNEL_TRANSFORM_MODE=self.KERNEL_TRANSFORM_MODE, bias=bias
        )
        if self.use_bn:
            norm_layer = build_dynamic_norm_layer(normalize)
            if 'switch' in normalize['type']:
                norm_layer = partial(norm_layer, width_list=self.out_channel_list)
            self.bn = norm_layer(max(self.out_channel_list))
        if self.act_func != '':
            self.act = build_activation(self.act_func, inplace=False)

        self.active_out_channel = max(self.out_channel_list)
        self.active_in_channel = max(self.in_channel_list)
        self.active_kernel_size = max(self.kernel_size_list)
        self.active_block = True

    def forward(self, x):
        if not self.active_block:
            return x

        self.active_in_channel = x.size(1)
        self.active_out_channel = min(self.active_out_channel, max(self.out_channel_list))

        self.conv.active_out_channel = self.active_out_channel
        self.conv.active_kernel_size = self.active_kernel_size

        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.act_func != '':
            x = self.act(x)
        return x

    @property
    def module_str(self):
        return 'DyConv(I_%d, O_%d, K_%d, S_%d, Transform_%s)' % (
            self.active_in_channel, self.active_out_channel, self.active_kernel_size, self.stride,
            self.KERNEL_TRANSFORM_MODE)

    def get_fake_input(self, input_shape=None):
        if not input_shape:
            input_shape = [1, self.active_in_channel, 224, 224]
        else:
            input_shape = input_shape
        return get_fake_input_for_module(self, input_shape)


@DYNAMIC_BLOCKS_REGISTRY.register('DynamicRegBottleneckBlock')
class DynamicRegBottleneckBlock(nn.Module):

    def __init__(self, in_channel_list, out_channel_list, kernel_size_list=3, expand_ratio_list=0.25,
                 group_width_list=1, stride=1, expand_input=False,
                 act_func='relu', use_se=False, act_func1='relu', act_func2='sigmoid',
                 KERNEL_TRANSFORM_MODE=False, divisor=8,
                 normalize={'type': 'dynamic_solo_bn'}):
        super(DynamicRegBottleneckBlock, self).__init__()

        self.in_channel_list = int2list(in_channel_list)
        self.out_channel_list = int2list(out_channel_list)

        self.kernel_size_list = int2list(kernel_size_list)
        self.expand_ratio_list = int2list(expand_ratio_list)
        self.group_width_list = int2list(group_width_list)

        self.stride = stride
        self.act_func = act_func
        self.expand_input = expand_input

        self.KERNEL_TRANSFORM_MODE = True if KERNEL_TRANSFORM_MODE and len(self.kernel_size_list) > 1 else False
        self.divisor = divisor

        norm_layer1 = norm_layer2 = build_dynamic_norm_layer(normalize)
        if 'switch' in normalize['type']:
            middle_channel = get_middle_channel(self.out_channel_list, self.expand_ratio_list, self.divisor)
            norm_layer1 = partial(norm_layer1, width_list=middle_channel)
            norm_layer2 = partial(norm_layer2, width_list=self.out_channel_list)

        # build modules
        if self.expand_input:
            max_middle_channel = make_divisible(max(self.in_channel_list) * max(self.expand_ratio_list), self.divisor)
        else:
            max_middle_channel = make_divisible(max(self.out_channel_list) * max(self.expand_ratio_list), self.divisor)
        group_num = max_middle_channel // max(self.group_width_list)
        self.point_conv1 = nn.Sequential(OrderedDict([
            ('conv', DynamicConv2d(max(self.in_channel_list), max_middle_channel, kernel_size=1, groups=1,
                                   KERNEL_TRANSFORM_MODE=KERNEL_TRANSFORM_MODE)),
            ('bn', norm_layer1(max_middle_channel)),
            ('act', build_activation(self.act_func, inplace=False)),
        ]))

        self.group_conv = nn.Sequential(OrderedDict([
            ('conv',
             DynamicConv2d(max_middle_channel, max_middle_channel, self.kernel_size_list, stride=self.stride,
                           groups=group_num, KERNEL_TRANSFORM_MODE=KERNEL_TRANSFORM_MODE)),
            ('bn', norm_layer1(max_middle_channel)),
            ('act', build_activation(self.act_func, inplace=False))
        ]))
        if use_se:
            self.group_conv.add_module('se',
                                       DynamicSEBlock(max_middle_channel, act_func1=act_func1, act_func2=act_func2))

        self.point_conv2 = nn.Sequential(OrderedDict([
            ('conv', DynamicConv2d(max_middle_channel, max(self.out_channel_list), kernel_size=1, groups=1,
                                   KERNEL_TRANSFORM_MODE=KERNEL_TRANSFORM_MODE)),
            ('bn', norm_layer2(max(self.out_channel_list))),
        ]))
        self.act3 = build_activation(self.act_func, inplace=False)

        self.active_kernel_size = max(self.kernel_size_list)
        self.active_expand_ratio = max(self.expand_ratio_list)
        self.active_out_channel = max(self.out_channel_list)
        self.active_in_channel = max(self.in_channel_list)
        self.active_group_width = max(self.group_width_list)
        self.active_block = True

        if max(self.in_channel_list) == max(self.out_channel_list) and self.stride == 1:
            self.shortcut = Identity()
        else:
            self.shortcut = DynamicConv2d(max(self.in_channel_list), max(self.out_channel_list), kernel_size=1,
                                          groups=1, stride=stride, KERNEL_TRANSFORM_MODE=KERNEL_TRANSFORM_MODE)
            self.shortcutbn = norm_layer2(max(self.out_channel_list))

    def forward(self, x):
        if not self.active_block:
            return x
        identity = x
        in_channel = x.size(1)
        self.active_in_channel = in_channel
        self.active_out_channel = min(self.active_out_channel, max(self.out_channel_list))

        if self.expand_input:
            mid_channel = make_divisible(self.active_in_channel * self.active_expand_ratio, self.divisor)
        else:
            mid_channel = make_divisible(self.active_out_channel * self.active_expand_ratio, self.divisor)

        self.point_conv1.conv.active_out_channel = mid_channel

        self.group_conv.conv.active_kernel_size = self.active_kernel_size
        self.group_conv.conv.active_out_channel = mid_channel
        self.group_conv.conv.active_groups = mid_channel // self.active_group_width

        self.point_conv2.conv.active_out_channel = self.active_out_channel

        x = self.point_conv1(x)
        x = self.group_conv(x)
        x = self.point_conv2(x)
        if in_channel == self.active_out_channel and self.stride == 1:
            x += identity
        else:
            self.shortcut.active_out_channel = self.active_out_channel
            x += self.shortcutbn(self.shortcut(identity))
        return self.act3(x)

    @property
    def module_str(self):
        if self.active_in_channel == self.active_out_channel and self.stride == 1:
            return 'RegBottleneck(I_%d, O_%d, E_%d, K_%d, G_%d, S_%d, Shortcut_%s, Transform_%s)' % (
                self.active_in_channel, self.active_out_channel, self.active_expand_ratio, self.active_kernel_size,
                self.active_group_width,
                self.stride, 'identity', self.KERNEL_TRANSFORM_MODE)
        else:
            return 'RegBottleneck(I_%d, O_%d, E_%d, K_%d, G_%d, S_%d, Shortcut_%s, Transform_%s)' % (
                self.active_in_channel, self.active_out_channel, self.active_expand_ratio, self.active_kernel_size,
                self.active_group_width,
                self.stride, 'DyConv', self.KERNEL_TRANSFORM_MODE)

    def get_fake_input(self, input_shape=None):
        if not input_shape:
            input_shape = [1, self.active_in_channel, 56, 56]
        else:
            input_shape = input_shape
        return get_fake_input_for_module(self, input_shape)


class DynamicSEBlock(nn.Module):

    def __init__(self, channel, reduction=4, bias=True, act_func1='relu', act_func2='h_sigmoid',
                 inplace=False, divisor=8):
        super(DynamicSEBlock, self).__init__()
        self.channel = channel
        self.reduction = reduction
        self.bias = bias
        self.act_func1 = act_func1
        self.act_func2 = act_func2
        self.inplace = inplace
        self.divisor = divisor

        num_mid = make_divisible(self.channel // self.reduction, divisor=self.divisor)

        self.fc = nn.Sequential(OrderedDict([
            ('reduce', nn.Conv2d(self.channel, num_mid, 1, 1, 0, bias=bias)),
            ('act1', build_activation(self.act_func1, inplace=False)),
            ('expand', nn.Conv2d(num_mid, self.channel, 1, 1, 0, bias=bias)),
            ('act2', build_activation(self.act_func2, inplace=False)),
        ]))

    def forward(self, x):
        in_channel = x.size(1)
        num_mid = make_divisible(in_channel // self.reduction, divisor=self.divisor)

        y = F.adaptive_avg_pool2d(x, output_size=1)
        # reduce
        reduce_conv = self.fc.reduce
        reduce_filter = reduce_conv.weight[:num_mid, :in_channel, :, :].contiguous()
        reduce_bias = reduce_conv.bias[:num_mid] if reduce_conv.bias is not None else None
        y = F.conv2d(y, reduce_filter, reduce_bias, 1, 0, 1, 1)
        # relu
        y = self.fc.act1(y)
        # expand
        expand_conv = self.fc.expand
        expand_filter = expand_conv.weight[:in_channel, :num_mid, :, :].contiguous()
        expand_bias = expand_conv.bias[:in_channel] if expand_conv.bias is not None else None
        y = F.conv2d(y, expand_filter, expand_bias, 1, 0, 1, 1)
        # hard sigmoid
        y = self.fc.act2(y)

        return x * y

    def get_fake_input(self, input_shape=None):
        if not input_shape:
            input_shape = [1, self.active_in_channel, 56, 56]
        else:
            input_shape = input_shape
        return get_fake_input_for_module(self, input_shape)
