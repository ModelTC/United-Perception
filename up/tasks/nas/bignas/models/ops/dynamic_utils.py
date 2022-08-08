import math
import copy
import torch
import numpy as np
from collections import OrderedDict
from itertools import repeat
import collections.abc

import torch # noqa
import torch.nn as nn
import torch.nn.functional as F
from up.utils.general.log_helper import default_logger as logger


def int2list(val, repeat_time=1):
    if isinstance(val, list) or isinstance(val, np.ndarray):
        return val
    elif isinstance(val, tuple):
        return list(val)
    else:
        return [val for _ in range(repeat_time)]


def sub_filter_start_end(kernel_size, sub_kernel_size):
    center = kernel_size // 2
    dev = sub_kernel_size // 2

    if kernel_size % 2 != 0:
        start, end = center - dev, center + dev + 1
    else:
        start, end = center - dev, center + dev

    assert end - start == sub_kernel_size
    return start, end


def get_same_padding(kernel_size):
    if isinstance(kernel_size, tuple):
        assert len(kernel_size) == 2, 'invalid kernel size: %s' % kernel_size
        p1 = get_same_padding(kernel_size[0])
        p2 = get_same_padding(kernel_size[1])
        return p1, p2
    assert isinstance(kernel_size, int), 'kernel size should be either `int` or `tuple`'
    assert kernel_size % 2 > 0, 'kernel size should be odd number'
    return kernel_size // 2


def make_divisible(v, divisor, min_val=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_val:
    :return:
    """
    if min_val is None:
        min_val = divisor
    new_v = max(min_val, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def build_activation(act_func, inplace=False):
    act_func = act_func.lower()
    if act_func == 'relu':
        return nn.ReLU(inplace=inplace)
    elif act_func == 'leaky_relu':
        return nn.LeakyReLU()
    elif act_func == 'relu6':
        return nn.ReLU6(inplace=inplace)
    elif act_func == 'tanh':
        return nn.Tanh()
    elif act_func == 'sigmoid':
        return nn.Sigmoid()
    elif act_func == 'h_swish' or act_func == 'hard_swish' or act_func == 'hswish':
        return Hswish(inplace=inplace)
    elif act_func == 'h_sigmoid' or act_func == 'hard_sigmoid' or act_func == 'hsigmoid':
        return Hsigmoid(inplace=inplace)
    elif act_func == 'silu' or act_func == 'swish':
        if hasattr(nn, 'SiLU'):
            return nn.SiLU()
        else:
            return SiLU()
    elif act_func == 'memoryefficientswish':
        return MemoryEfficientMish()
    elif act_func == 'Mish':
        return Mish()
    elif act_func == 'memoryefficientmish':
        return MemoryEfficientMish()
    elif act_func == 'frelu':
        return FReLU()
    elif act_func == 'gelu':
        return GELU()
    elif act_func == 'qgelu' or act_func == 'quick_gelu':
        return QGELU()
    elif act_func is None:
        return None
    else:
        raise ValueError('do not support: %s' % act_func)


class GELU(nn.Module):
    """
    Gaussian Error Linear Units, based on
    `"Gaussian Error Linear Units (GELUs)" <https://arxiv.org/abs/1606.08415>`
    """

    def __init__(self, approximate=True):
        super(GELU, self).__init__()
        self.approximate = approximate

    def forward(self, x):
        if self.approximate:
            cdf = 0.5 * (1.0 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
            return x * cdf
        else:
            return x * (torch.erf(x / math.sqrt(2)) + 1) / 2


class QGELU(nn.Module):
    """
    Gaussian Error Linear Units, based on
    `"Gaussian Error Linear Units (GELUs)" <https://arxiv.org/abs/1606.08415>`
    """

    def __init__(self, approximate=True):
        super(QGELU, self).__init__()
        self.approximate = approximate

    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class Hswish(nn.Module):

    def __init__(self, inplace=False):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


class Hsigmoid(nn.Module):

    def __init__(self, inplace=False):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.


class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


class MemoryEfficientSwish(nn.Module):
    class F(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x * torch.sigmoid(x)

        @staticmethod
        def backward(ctx, grad_output):
            x = ctx.saved_tensors[0]
            sx = torch.sigmoid(x)
            return grad_output * (sx * (1 + x * (1 - sx)))

    def forward(self, x):
        return self.F.apply(x)


# Mish https://github.com/digantamisra98/Mish ---------------------------
class Mish(nn.Module):
    @staticmethod
    def forward(x):
        return x * F.softplus(x).tanh()


class MemoryEfficientMish(nn.Module):
    class F(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x.mul(torch.tanh(F.softplus(x)))  # x * tanh(ln(1 + exp(x)))

        @staticmethod
        def backward(ctx, grad_output):
            x = ctx.saved_tensors[0]
            sx = torch.sigmoid(x)
            fx = F.softplus(x).tanh()
            return grad_output * (fx + x * sx * (1 - fx * fx))

    def forward(self, x):
        return self.F.apply(x)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# FReLU https://arxiv.org/abs/2007.11824 -------------------------------
class FReLU(nn.Module):
    def __init__(self, c1, k=3):  # ch_in, kernel
        super().__init__()
        self.conv = nn.Conv2d(c1, c1, k, 1, 1, groups=c1)
        self.bn = nn.BatchNorm2d(c1)

    def forward(self, x):
        return torch.max(x, self.bn(self.conv(x)))


class SEModule(nn.Module):
    REDUCTION = 4

    def __init__(self, channel, divisor=8):
        super(SEModule, self).__init__()

        self.channel = channel
        self.reduction = SEModule.REDUCTION

        num_mid = make_divisible(self.channel // self.reduction, divisor=divisor)

        self.fc = nn.Sequential(OrderedDict([
            ('reduce', nn.Conv2d(self.channel, num_mid, 1, 1, 0, bias=True)),
            ('relu', nn.ReLU(inplace=False)),
            ('expand', nn.Conv2d(num_mid, self.channel, 1, 1, 0, bias=True)),
            ('h_sigmoid', Hsigmoid(inplace=False)),
        ]))

    def forward(self, x):
        y = x.mean(3, keepdim=True).mean(2, keepdim=True)
        y = self.fc(y)
        return x * y


# this is used in Quantization
def gradscale(x, scale):
    yOut = x
    yGrad = x * scale
    y = (yOut - yGrad).detach() + yGrad
    return y


def signpass(x):
    yOut = x.sign()
    yGrad = x
    y = (yOut - yGrad).detach() + yGrad
    return y


def clippass(x, Qn, Qp):
    yOut = F.hardtanh(x, Qn, Qp)
    yGrad = x
    y = (yOut - yGrad).detach() + yGrad
    return y


def roundpass(x):
    yOut = x.round()
    yGrad = x
    y = (yOut - yGrad).detach() + yGrad
    return y


# v is input
# s in learnable step size
# p in the number of bits
def quantize(v, s, p, isActivation):
    if isActivation:
        Qn = 0
        Qp = 2 ** p - 1
        nfeatures = v.nelement()
        gradScaleFactor = 1 / math.sqrt(nfeatures * Qp)

    else:
        Qn = -2 ** (p - 1)
        Qp = 2 ** (p - 1) - 1
        if p == 1:
            Qp = 1
        nweight = v.nelement()
        gradScaleFactor = 1 / math.sqrt(nweight * Qp)

    s = gradscale(s, gradScaleFactor)
    v = v / s.abs()
    v = F.hardtanh(v, Qn, Qp)
    if not isActivation and p == 1:
        vbar = signpass(v)
    else:
        vbar = roundpass(v)
    vhat = vbar * s.abs()
    return vhat


def copy_module_weights(module, super_module):
    logger.info('subnet module has keys {}'.format(module.state_dict().keys()))
    logger.info('supnet module has keys {}'.format(super_module.state_dict().keys()))

    for (k1, v1), (k2, v2) in zip(module.state_dict().items(), super_module.state_dict().items()):
        logger.info('subnet key {} size {} supnet key {}  size {}'.format(k1, v1.size(), k2, v2.size()))
        assert k1.split('.')[-1] == k2.split('.')[-1], '{} and {} has different name'.format(k1, k2)
        assert v1.dim() == v2.dim(), '{} and {} has different dimension'.format(k1, k2)

        if v1.dim() == 0:
            v1.copy_(copy.deepcopy(v2.data))
        elif v1.dim() == 1:
            v1.copy_(copy.deepcopy(v2[:v1.size(0)].data))
        elif v1.dim() == 2:
            v1.copy_(copy.deepcopy(v2[:v1.size(0), :v1.size(1)].data))
        elif v1.dim() == 4:
            start, end = sub_filter_start_end(v2.size(3), v1.size(3))
            v1.copy_(copy.deepcopy(v2[:v1.size(0), :v1.size(1), start:end, start:end].data))


def get_middle_channel(out_channel_list, expand_ratio_list, divisor):
    middle_channel = []
    for out_channel in out_channel_list:
        for expand in expand_ratio_list:
            mid_c = make_divisible(out_channel * expand, divisor)
            if mid_c not in middle_channel:
                middle_channel.append(mid_c)
    middle_channel = sorted(middle_channel)
    return middle_channel


def get_fake_input_for_module(module, input_shape):
    input = torch.zeros(*input_shape)
    device = next(module.parameters(), torch.tensor([])).device
    input = input.to(device)
    for m in module.parameters():
        if m.dtype == torch.float16:
            input = input.half()
    return input


def get_value_with_shape(state_dict1, state_dict2):
    for k1, v1 in state_dict1.items():
        for k2, v2 in state_dict2.items():
            if v1.nelement() == v2.nelement():
                state_dict1[k1] = v2.reshape(*v1.size())
    return state_dict1


def modify_state_dict(state_dict1, state_dict2):
    logger.info('modify state dict')
    if check_state_dict(state_dict1, state_dict2):
        return dict(zip(state_dict1.keys(), state_dict2.values()))
    else:
        for (k1, v1), (k2, v2) in zip(state_dict1.items(), state_dict2.items()):
            left_state_dict1 = {}
            left_state_dict2 = {}
            if v1.size() == v2.size():
                continue
            else:
                left_state_dict1[k1] = v1
                left_state_dict2[k2] = v2
        left_state_dict1 = get_value_with_shape(left_state_dict1, left_state_dict2)
        state_dict1.update(left_state_dict1)
    return state_dict1


def check_state_dict(state_dict1, state_dict2):
    if len(state_dict1.keys()) != len(state_dict2.keys()):
        logger.info('state dict has different keys length with state_dict1 {} and state_dict2 {}'.format(
            len(state_dict1.keys()), len(state_dict2.keys())))
        return False
    elif [v.size() for v in state_dict1.values()] != [v.size() for v in state_dict2.values()]:
        logger.info('state dict has different value shape')
        for (k1, v1), (k2, v2) in zip(state_dict1.items(), state_dict2.items()):
            if v1.size() == v2.size():
                continue
            else:
                logger.info('key1 {} with shape {} and key2 {} with shape {}'.format(k1, list(v1.size()),
                                                                                     k2, list(v2.size())))
        return False
    return True


def get_model_intermediate_feature(model, input):
    out_dict = {}
    handles = []

    def make_hook(module_name):
        def hook(module, data, out):
            out_dict[module_name] = out[0]

        return hook

    for n, m in model.named_modules():
        if sum(map(lambda x: isinstance(m, x), [nn.Linear, nn.Conv2d])):
            handle = m.register_forward_hook(make_hook(n))
            handles.append(handle)
    model(input)
    return out_dict


def check_model_intermediate(supernet, model, input):
    supernet_out = get_model_intermediate_feature(supernet, input)
    model_out = get_model_intermediate_feature(model, input)
    for (k1, v1), (k2, v2) in zip(supernet_out.items(), model_out.items()):
        logger.info('module k1 {} and k2 {} has the same output: {}'.format(k1, k2, torch.equal(v1, v2)))
        if not torch.equal(v1, v2):
            return


def check_model_output(supernet, model, input_shape=None):
    if input_shape is not None:
        input = supernet.get_fake_input(input_shape=input_shape)
    else:
        input = supernet.get_fake_input()
    supernet.eval()
    model.eval()
    y1 = supernet.forward(input)
    y2 = model.forward(input)
    if torch.equal(y1, y2):
        logger.info('supernet and model has the same output')
        check_model_intermediate(supernet, model, input)
        return True
    else:
        return False


def verify_supernet_with_model(supernet, model, input_shape=None):
    if check_state_dict(supernet.state_dict(), model.state_dict()):
        new_state_dict = modify_state_dict(supernet.state_dict(), model.state_dict())
        supernet.load_state_dict(new_state_dict)
        if check_model_output(supernet, model, input_shape):
            logger.info('supernet and model has the same output, they mush be the same')
            return True
        else:
            logger.info('There must something different in the inference!')
            return False
    else:
        logger.info('supernet has different state dict keys or value shape')
        new_state_dict = modify_state_dict(supernet.state_dict(), model.state_dict())
        supernet.load_state_dict(new_state_dict)
        return check_model_output(supernet, model)


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def get_relative_position_index(window_size):
    # get pair-wise relative position index for each token inside the window
    coords_h = torch.arange(window_size[0])
    coords_w = torch.arange(window_size[1])
    coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
    relative_coords[:, :, 1] += window_size[1] - 1
    relative_coords[:, :, 0] *= 2 * window_size[1] - 1
    relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
    return relative_position_index


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple
