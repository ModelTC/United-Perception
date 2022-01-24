# Standard Library
import math

# Import from third library
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn.modules.module import Module
from torch.nn.modules.utils import _pair

# Import from up
from up.utils.general.log_helper import default_logger as logger

# Import from local
from .._C import deform_conv_v1


class DeformableConvFunction(Function):

    @staticmethod
    def symbolic(g, input, offset, weight, out_channels,
                 kernel_size, stride, pad, dilations, groups, deformable_groups):

        assert input.type().sizes()[2] is not None, "Input Error: Only 4D input Tensors Supported"
        assert input.type().sizes()[3] is not None, "Input Error: Only 4D input Tensors Supported"
        return g.op("DeformConv", input, offset, weight, out_channels_i=out_channels,
                    kernel_size_i=kernel_size, stride_i=stride, pad_i=pad, dilations_i=dilations,
                    groups_i=groups, deformable_groups_i=deformable_groups)

    @staticmethod
    def forward(self, input, offset, weight, out_channels,
                kernel_size, stride, pad, dilations, groups, deformable_groups):
        if input is not None and input.dim() != 4:
            raise ValueError("Expected 4D tensor as input, got {}D tensor instead.".format(input.dim()))

        self.save_for_backward(input, offset, weight)

        self.stride = stride
        self.padding = pad
        self.dilation = dilations
        self.groups = groups
        self.deformable_groups = deformable_groups

        output = input.new(*DeformableConvFunction._output_size(
            input, weight, self.stride, self.padding, self.dilation))

        self.bufs_ = [
            input.new(input.size(1) * weight.size(3) * weight.size(2), output.size(2), output.size(3)).zero_(),
            input.new(output.size(2), output.size(3)).fill_(1)
        ]  # columns, ones

        forward_fn = deform_conv_v1.forward_cuda
        if not input.is_cuda:
            logger.warning(
                '---CPU version of DEFORMABLE CONV V1 is a dummpy function, which is used to support tocaffe')
            forward_fn = deform_conv_v1.forward_cpu

        forward_fn(
            input, weight, offset, output,
            self.bufs_[0], self.bufs_[1],
            weight.size(2), weight.size(3),
            self.stride[0], self.stride[1],
            self.padding[0], self.padding[1],
            self.dilation[0], self.dilation[1],
            self.groups, self.deformable_groups)
        return output

    @staticmethod
    def backward(self, grad_output):
        input, offset, weight = self.saved_tensors

        grad_input = None
        grad_offset = None
        grad_weight = None

        if not grad_output.is_cuda:
            raise NotImplementedError
        else:
            grad_output = grad_output.contiguous()
            if self.needs_input_grad[0] or self.needs_input_grad[1]:
                grad_input = input.new(*input.size()).zero_()
                grad_offset = offset.new(*offset.size()).zero_()

                deform_conv_v1.backward_input_cuda(
                    input, offset, grad_output, grad_input, grad_offset, weight, self.bufs_[0],
                    weight.size(2), weight.size(3), self.stride[0], self.stride[1],
                    self.padding[0], self.padding[1], self.dilation[0],
                    self.dilation[1], self.groups, self.deformable_groups)

            if self.needs_input_grad[2]:
                grad_weight = weight.new(*weight.size()).zero_()
                deform_conv_v1.backward_parameters_cuda(
                    input, offset, grad_output, grad_weight, self.bufs_[0], self.bufs_[1],
                    weight.size(2), weight.size(3), self.stride[0], self.stride[1],
                    self.padding[0], self.padding[1], self.dilation[0],
                    self.dilation[1], self.groups, self.deformable_groups, 1)
        return grad_input, grad_offset, grad_weight, None, None, None, None, None, None, None

    @staticmethod
    def _output_size(input, weight, stride, padding, dilation):
        channels = weight.size(0)

        output_size = (input.size(0), channels)
        for d in range(input.dim() - 2):
            _in_size = input.size(d + 2)
            _pad = padding[d]
            _kernel = dilation[d] * (weight.size(d + 2) - 1) + 1
            _stride = stride[d]
            output_size += ((_in_size + (2 * _pad) - _kernel) // _stride + 1, )

        if not all(map(lambda s: s > 0, output_size)):
            raise ValueError("convolution input is too small (output would be {})".format(
                'x'.join(map(str, output_size))))
        return output_size


class DeformableConv(Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 num_deformable_groups=1):
        super(DeformableConv, self).__init__()

        # logger.warning('warning! DeformableConv will be deprecated in the near future, '
        #                'plase use DeformConv2d instead, which is an unified module with '
        #                'the same interface with torch.nn.Conv2d')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.num_deformable_groups = num_deformable_groups

        assert in_channels % groups == 0
        assert out_channels % groups == 0
        assert (in_channels // groups % num_deformable_groups == 0)
        assert (out_channels // groups % num_deformable_groups == 0)
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, offset):
        # logger.info(self.stride)
        return DeformableConvFunction.apply(
            input, offset, self.weight, self.out_channels, self.kernel_size, self.stride, self.padding, self.dilation,
            self.groups, self.num_deformable_groups)

    def __repr__(self):
        s = ('{name} ({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}, num_deformable_groups={num_deformable_groups}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)
