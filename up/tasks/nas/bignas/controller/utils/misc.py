from collections import Iterable
import json
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import spring.linklink as link
except:  # noqa
    from up.utils.general.fake_linklink import link

from up.utils.env.dist_helper import allreduce
from up.utils.env.dist_helper import get_world_size
from up.utils.general.log_helper import default_logger as logger

from up.tasks.nas.bignas.models.ops.dynamic_ops import DynamicConv2d, DynamicLinear
# from up.tasks.nas.bignas.models.ops.dynamic_blocks import DynamicMultiHeadAttention
from up.tasks.nas.bignas.utils.traverse import traverse_search_range


def count_dynamic_flops_and_params(model, input, depth=None):

    def dynamic_multiheadattention_hook(m, input):
        _, N, _ = input[0].size(0), input[0].size(1), input[0].size(2)
        # compute flops except linear
        flops = 0
        # qkv is linear, will compute in dynamiclinear
        # attn = (q @ k.transpose(-2, -1))
        flops += m.active_heads * N * 64 * N
        #  x = (attn @ v)
        flops += m.active_heads * N * N * 64
        # proj is linear, will compute in dynamiclinear
        m.total_flops += int(flops)
        m.total_params += 0

    def dynamic_conv2d_hook(m, input):
        n, c, h, w = input[0].size(0), input[0].size(1), input[0].size(
            2), input[0].size(3)
        flops = n * h * w * c * m.active_out_channel * m.active_kernel_size * m.active_kernel_size \
            / m.stride[0] / m.stride[1] / m.groups
        m.total_flops += int(flops)
        params = c * m.active_out_channel * m.active_kernel_size * m.active_kernel_size
        m.total_params += int(params)

    def dynamic_linear_hook(m, input):
        if input[0].dim() == 2:
            c = input[0].size(-1)
            flops = c * m.active_out_channel
        elif input[0].dim() == 3:
            c = input[0].size(1) * input[0].size(2)
            flops = c * m.active_out_channel
        else:
            raise NotImplementedError
        m.total_flops += int(flops)
        params = input[0].size(-1) * m.active_out_channel
        m.total_params += int(params)

    def linear_hook(m, input):
        if input[0].dim() == 2:
            c = input[0].size(-1)
            flops = c * m.out_features
        elif input[0].dim() == 3:
            c = input[0].size(1) * input[0].size(2)
            flops = c * m.out_features
        else:
            raise NotImplementedError
        m.total_flops += int(flops)
        params = c * m.active_out_channel
        m.total_params += int(params)

    def conv2d_hook(m, input):
        n, _, h, w = input[0].size(0), input[0].size(1), input[0].size(
            2), input[0].size(3)
        flops = n * h * w * m.in_channels * m.out_channels * m.kernel_size[0] * m.kernel_size[1] \
            / m.stride[0] / m.stride[1] / m.groups
        m.total_flops += int(flops)
        params = m.weight.size(0) * m.weight.size(1) * m.weight.size(
            2) * m.weight.size(3)
        m.total_params += int(params)

    _HOOK_FACTORY = {
        DynamicConv2d: dynamic_conv2d_hook,
        torch.nn.Conv2d: conv2d_hook,
        DynamicLinear: dynamic_linear_hook,
        torch.nn.Linear: linear_hook,
        # DynamicMultiHeadAttention: dynamic_multiheadattention_hook
    }

    def collect_flops(module):
        for m in module.children():
            module.total_flops += m.total_flops
            module.total_params += m.total_params

    hooks = []
    for m in model.modules():
        typ = type(m)
        if typ in _HOOK_FACTORY.keys():
            func = _HOOK_FACTORY[typ]
            h = m.register_forward_pre_hook(func)
            hooks.append(h)

        setattr(m, 'total_flops', torch.zeros(1))
        setattr(m, 'total_params', torch.zeros(1))

    training = model.training
    model.eval()
    with torch.no_grad():
        _ = model(input)

    model.apply(collect_flops)

    flops_dict, params_dict = {}, {}
    for name, module in model.named_modules():
        if name and (depth is None or name.count('.') <= depth):
            flops_dict[name] = float(module.total_flops)
            params_dict[name] = float(module.total_params)
    flops_dict['total'] = float(model.total_flops)
    params_dict['total'] = float(model.total_params)
    for h in hooks:
        h.remove()
    model.train(training)

    return flops_dict, params_dict


Quantities = {"T": 1e12, "G": 1e9, "M": 1e6, "K": 1e3, "B": 1}


def clever_format(nums, format="%.2f"):
    def _format_func(num):
        if isinstance(num, str):
            return num
        num = float(num)
        for k, v in Quantities.items():
            if num > v:
                return format % (num / v) + k
        return num

    if not isinstance(nums, Iterable):
        return _format_func(nums)
    elif isinstance(nums, dict):
        return {k: _format_func(v) for k, v in nums.items()}
    else:
        return [_format_func(_) for _ in nums]


def clever_dump_table(table, numerical_keys):
    def _format_keys(tb, k):
        if isinstance(tb, dict):
            if k in tb.keys():
                tb[k] = clever_format(tb[k])
            else:
                for _, __ in tb.items():
                    _format_keys(__, k)
        elif isinstance(tb, list):
            for _ in tb:
                _format_keys(_, k)

    table = table.copy()
    for k in numerical_keys:
        _format_keys(table, k)
    return json.dumps(table)


def parse_flops(flops):
    def _parse_item(it):
        if not isinstance(it, str):
            return it
        q = it[-1]
        if q in Quantities.keys():
            return float(it[:-1]) * Quantities[q]
        raise NotImplementedError

    if isinstance(flops, list):
        return [_parse_item(_) for _ in flops]
    elif isinstance(flops, dict):
        return {k: _parse_item(v) for k, v in flops.items()}
    elif isinstance(flops, str):
        return _parse_item(flops)
    elif flops is None:
        return 0
    else:
        raise NotImplementedError


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def reduce_update(self, tensor, num=1):
        allreduce(tensor)
        self.update(tensor.item(), num=num)

    def update(self, val, num=1):
        if self.length > 0:
            # currently assert num==1 to avoid bad usage, refine when there are some explict requirements
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val * num
            self.count += num
            self.avg = self.sum / self.count


def get_kwargs_itertools(kwargs_cfg):

    def parse_range_attribute(attribute):
        stage_wise_range = attribute['space']['dynamic_range']
        sample_method = attribute['sample_strategy']
        ranges = traverse_search_range(stage_wise_range, sample_method)
        return ranges

    parsed_cfg = {}

    for k, v in kwargs_cfg.items():
        if isinstance(v, dict) and v['search']:
            parsed_cfg[k] = parse_range_attribute(v)

    return parsed_cfg


def get_image_size_with_shape(image_size):
    """
    Args:
        image_size(int or list or tuple): image_size can be int or list or tuple
    Returns:
        input_size(tuple): process the input image size to 4 dimension input size
    """
    if isinstance(image_size, int):
        input_size = (1, 3, image_size, image_size)
    elif (isinstance(image_size, list)
            or isinstance(image_size, tuple)) and len(image_size) == 1:
        input_size = (1, 3, image_size[0], image_size[0])
    elif (isinstance(image_size, list)
            or isinstance(image_size, tuple)) and len(image_size) == 2:
        input_size = (1, 3, image_size[0], image_size[1])
    elif (isinstance(image_size, list)
            or isinstance(image_size, tuple)) and len(image_size) == 3:
        input_size = (1, image_size[0], image_size[1], image_size[2])
    elif (isinstance(image_size, list)
            or isinstance(image_size, tuple)) and len(image_size) == 4:
        input_size = image_size
    else:
        raise ValueError('image size should be lower than 5 dimension')
    return input_size


def reset_model_bn_forward(model, bn_mean, bn_var):
    """
    Reset the BatchNorm forward function of the model, and store mean and variance in bn_mean and bn_var
    Args:
        model (nn.Module): the model need to be reset
        data_loader (dict): data_loader with calibrate dataset
    """
    for name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, link.nn.SyncBatchNorm2d):
            bn_mean[name] = AverageMeter()
            bn_var[name] = AverageMeter()

            def new_forward(bn, mean_est, var_est):
                def lambda_forward(x):
                    batch_mean = x.mean(0, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)  # 1, C, 1, 1
                    batch_var = (x - batch_mean) * (x - batch_mean)
                    batch_var = batch_var.mean(0, keepdim=True).mean(
                        2, keepdim=True).mean(3, keepdim=True)

                    batch_mean = torch.squeeze(batch_mean).float()
                    batch_var = torch.squeeze(batch_var).float()
                    # 直接算正常的bn的mean 和 var

                    # 累计mean_est = batch_mean * batch
                    reduce_batch_mean = batch_mean.clone() / get_world_size()
                    reduce_batch_var = batch_var.clone() / get_world_size()
                    allreduce(reduce_batch_mean.data)
                    allreduce(reduce_batch_var.data)
                    mean_est.update(reduce_batch_mean.data, x.size(0))
                    var_est.update(reduce_batch_var.data, x.size(0))

                    # bn forward using calculated mean & var
                    _feature_dim = batch_mean.size(0)
                    return F.batch_norm(
                        x,
                        batch_mean,
                        batch_var,
                        bn.weight[:_feature_dim],
                        bn.bias[:_feature_dim],
                        False,
                        0.0,
                        bn.eps,
                    )

                return lambda_forward

            m.forward = new_forward(m, bn_mean[name], bn_var[name])


def copy_bn_statistics(model, bn_mean, bn_var):
    for name, m in model.named_modules():
        if name in bn_mean and bn_mean[name].count > 0:
            logger.info('copy bn statistics for layer {}'.format(name))
            feature_dim = bn_mean[name].avg.size(0)
            assert isinstance(m, nn.BatchNorm2d) or isinstance(m, link.nn.SyncBatchNorm2d)
            # 最后取得的均值
            m.running_mean.data[:feature_dim].copy_(bn_mean[name].avg)
            m.running_var.data[:feature_dim].copy_(bn_var[name].avg)


def rewrite_batch_norm_bias(model):
    """
    Args:
        model (bool): the model to be rewrite, in normal initialization the BN bias are set to zero,
                      in this way, nart will delete bias in convolution after mergeing BN,
                      while in most cases, convolution have bais and this will affect latency
    """
    for name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, link.nn.SyncBatchNorm2d):
            if m.bias.data[0].item() == 0:
                m.bias.data.fill_(1)
                logger.info(
                    'rewrite batch norm bias for layer {}'.format(name))


def adapt_settings(subnet_settings):
    new_settings = {}
    for k, v in subnet_settings.items():
        if '.' not in k:
            new_settings[k] = v
        else:
            a, b = k.split('.')
            if a not in new_settings.keys():
                new_settings[a] = {}
            new_settings[a][b] = v
    return new_settings
