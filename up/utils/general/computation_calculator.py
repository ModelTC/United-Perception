# Standard Library
import time
from collections import Iterable, OrderedDict

import numpy as np
import torch
import torch.nn as nn

from up.utils.model.bn_helper import *
from up.utils.env.gene_env import to_device


def get_cur_time():
    torch.cuda.synchronize()
    return time.time()


def get_diff_time(pre_time):
    torch.cuda.synchronize()
    diff_time = time.time() - pre_time
    return diff_time, time.time()


def clever_format(nums, format="%.2f"):
    def _format_func(num):
        num = int(num)
        if num > np.math.pow(2, 40):
            return format % (num / 1e12) + " T"
        elif num > np.math.pow(2, 30):
            return format % (num / 1e9) + " G"
        elif num > np.math.pow(2, 20):
            return format % (num / 1e6) + " M"
        elif num > np.math.pow(2, 10):
            return format % (num / 1e3) + " K"
        else:
            return format % num + " B"

    if not isinstance(nums, Iterable):
        return _format_func(nums)
    elif isinstance(nums, dict):
        return {k: _format_func(v) for k, v in nums}
    else:
        return [_format_func(_) for _ in nums]


def flops_cal(model, input_shape, depth):
    inputs = {
        'image': torch.randn(1, input_shape[0], input_shape[1], input_shape[2]),
        'image_info': [[input_shape[1], input_shape[2], 1, input_shape[1], input_shape[2], False]],
        'filename': ['Test.jpg'],
        'label': torch.LongTensor([[0]]),
    }

    return profile(model, inputs=(to_device(inputs),), depth=depth)


def profile(model, inputs, verbose=True, depth=0):
    handler_collection = []

    def add_hooks(m):

        if len(list(m.children())) > 0:
            return

        m.register_buffer('total_macc', torch.zeros(1))
        m.register_buffer('total_ops', torch.zeros(1))
        m.register_buffer('total_params', torch.zeros(1))

        m_type = type(m)
        fn = None
        if m_type in register_hooks:
            fn = register_hooks[m_type]

        if fn is None:
            if verbose:
                print("No implemented counting method for ", m)
        else:
            handler = m.register_forward_hook(fn)
            handler_collection.append(handler)

    def collect_flops(module):
        if not hasattr(module, 'total_macc'):
            module.register_buffer('total_macc', torch.zeros(1))
            module.register_buffer('total_ops', torch.zeros(1))
            module.register_buffer('total_params', torch.zeros(1))
        if not hasattr(module, 'total_memory'):
            module.register_buffer('total_memory', torch.zeros(1))
        if not hasattr(module, 'total_time'):
            module.register_buffer('total_time', torch.zeros(1))

        for m in module.children():
            module.total_macc += m.total_macc
            module.total_ops += m.total_ops
            module.total_params += m.total_params
            module.total_memory += m.total_memory
        print(module, module.total_macc)

    def register_memory_hooks(module):

        def pre_forward_hook(m, input):
            m.pre_memory = torch.cuda.memory_allocated()
            m.start_time = get_cur_time()

        def forward_hook(m, input, output):
            m.total_memory = torch.cuda.memory_allocated() - m.pre_memory
            torch.cuda.empty_cache()
            m.total_time, _ = get_diff_time(m.start_time)

        handler = module.register_forward_pre_hook(pre_forward_hook)
        handler_collection.append(handler)
        handler = module.register_forward_hook(forward_hook)
        handler_collection.append(handler)

    # original_device = model.parameters().__next__().device
    training = model.training

    model.eval()
    # warmup for time calculate
    with torch.no_grad():
        model(*inputs)

    model.apply(add_hooks)
    model.apply(register_memory_hooks)

    with torch.no_grad():
        model(*inputs)

    model.apply(collect_flops)

    module_flops_dict = OrderedDict()

    for name, module in model.named_modules():
        if name and name.count('.') <= depth:
            module_flops_dict[name] = {
                'macc': module.total_macc,
                'ops': module.total_ops,
                'params': module.total_params,
                'memory': module.total_memory,
                'time': module.total_time,
            }
    module_flops_dict['total'] = {
        'macc': model.total_macc,
        'ops': model.total_ops,
        'params': model.total_params,
        'memory': model.total_memory,
        'time': model.total_time
    }

    # reset model to original status
    model.train(training)
    for handler in handler_collection:
        handler.remove()

    return module_flops_dict


def count_zero(m, x, y):
    m.total_macc = torch.Tensor([0])
    m.total_ops = torch.Tensor([0])
    m.total_params = torch.Tensor([0])


def count_deconv2d(m, x, y):
    return count_conv2d(m, x, y)


def count_conv2d(m, x, y):
    groups = m.groups
    cin = m.in_channels // groups
    cout = m.out_channels // groups
    kh, kw = m.kernel_size
    batch_size = y.size(0)
    out_h = y.size(2)
    out_w = y.size(3)

    kernel_ops = 2 * cin * kh * kw - 1
    kernel_ma = cin * kh * kw
    bias_ops = 1 if m.bias is not None else 0

    output_elements = out_w * out_h * cout
    total_macc = groups * (output_elements * kernel_ma)
    total_ops = groups * (output_elements * kernel_ops + bias_ops * output_elements)
    total_params = groups * (kh * kw * cin * cout + bias_ops * cout)

    m.total_macc += torch.Tensor([int(total_macc)]) * batch_size
    m.total_ops += torch.Tensor([int(total_ops)]) * batch_size
    m.total_params = torch.Tensor([int(total_params)])


def count_bn(m, x, y):
    # During inference, BN is completely merged into the calculation of foregoing Conv/FC layer.
    x = x[0]
    c_out = y.size(1)
    #    nelements = x.numel()
    #    # subtract, divide, gamma, beta
    #    total_macc = 2 * nelements
    #    total_ops = 4 * nelements

    m.total_macc += torch.Tensor([0])
    m.total_ops += torch.Tensor([0])
    m.total_params = torch.Tensor([int(c_out) * 2])


def count_relu(m, x, y):
    x = x[0]
    nelements = x.numel()
    total_ops = nelements

    m.total_macc += torch.Tensor([0])
    m.total_ops += torch.Tensor([int(total_ops)])
    m.total_params = torch.Tensor([0])


def count_avgpool(m, x, y):
    total_add = torch.prod(torch.Tensor([m.kernel_size]))
    total_div = 1
    kernel_ops = total_add + total_div
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements

    m.total_macc += torch.Tensor([0])
    m.total_ops += torch.Tensor([int(total_ops)])
    m.total_params = torch.Tensor([0])


def count_adap_avgpool(m, x, y):
    kernel = torch.Tensor([*(x[0].shape[2:])]) // torch.Tensor(list((m.output_size,))).squeeze()
    total_add = torch.prod(kernel)
    total_div = 1
    kernel_ops = total_add + total_div
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements

    m.total_macc += torch.Tensor([0])
    m.total_ops += torch.Tensor([int(total_ops)])
    m.total_params = torch.Tensor([0])


def count_linear(m, x, y):
    # per output element
    total_mul = m.in_features
    total_add = m.in_features - 1
    num_elements = y.numel()
    total_macc = total_mul * num_elements
    total_ops = (total_mul + total_add) * num_elements

    m.total_macc += torch.Tensor([int(total_macc)])
    m.total_ops += torch.Tensor([int(total_ops)])
    m.total_params = torch.Tensor([m.in_features * m.out_features])


register_hooks = {
    nn.Conv2d: count_conv2d,
    nn.ConvTranspose2d: count_deconv2d,
    nn.BatchNorm2d: count_bn,
    nn.ReLU: count_relu,
    nn.ReLU6: count_relu,
    nn.LeakyReLU: count_relu,
    nn.AvgPool2d: count_avgpool,
    nn.AdaptiveAvgPool2d: count_adap_avgpool,
    nn.Linear: count_linear,
    nn.Dropout: count_zero,
    FrozenBatchNorm2d: count_bn,
    nn.MaxPool2d: count_zero,
    nn.CrossEntropyLoss: count_zero,
    GroupSyncBatchNorm: count_bn
}
