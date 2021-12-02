# Standard Library
from collections.abc import Iterable, Mapping
from eod.utils.model.bn_helper import (
    FrozenBatchNorm2d,
    CaffeFrozenBatchNorm2d,
    PyTorchSyncBN,
    GroupNorm
)

import numpy as np
import torch
import torch.nn
from .global_flag import FP16_FLAG


def to_float32(func):
    """Whatever the input dtype is, this decorator convert the input dtype
    to fp32 to keep the computation precision, and the output dtype is fp32
    """

    def recursive_to(x):
        if not FP16_FLAG.fp16:
            return x
        if torch.is_tensor(x):
            if x.is_floating_point() and x.dtype != torch.float32:
                return x.to(dtype=torch.float32)
            else:
                return x
        elif isinstance(x, str) or isinstance(x, bytes):  # there is a dead loop when x is a str with len(x)=1n
            return x
        elif isinstance(x, np.ndarray):  # numpy recommends building array by calling np.array
            return np.array(list(map(recursive_to, x)))
        elif isinstance(x, Mapping):
            return type(x)({k : recursive_to(v) for k, v in x.items()})
        elif isinstance(x, Iterable):
            return type(x)(list(map(recursive_to, x)))
        else:
            return x

    def fp32_wrapper(*args, **kwargs):
        args_ = recursive_to(args)
        kwargs_ = recursive_to(kwargs)
        return func(*args_, **kwargs_)

    return fp32_wrapper


def register_float_module(keep_fp32):
    import spring.linklink as linklink
    if keep_fp32:
        linklink.fp16.register_float_module(torch.nn.BatchNorm2d, cast_args=True)
        # linklink.fp16.register_float_module(GroupSyncBatchNorm, cast_args=True)
        linklink.fp16.register_float_module(FrozenBatchNorm2d, cast_args=True)
        linklink.fp16.register_float_module(PyTorchSyncBN, cast_args=True)
        linklink.fp16.register_float_module(CaffeFrozenBatchNorm2d, cast_args=True)
        linklink.fp16.register_float_module(GroupNorm, cast_args=True)
        # linklink.fp16.register_float_module(TaskBatchNorm2d, cast_args=True)
        # linklink.fp16.register_float_module(SyncTaskBatchNorm2d, cast_args=True)
        linklink.fp16.init()
