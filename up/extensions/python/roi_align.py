# Import from third library
import torch
from torch.autograd import Function

# Import from pod
from up.utils.general.log_helper import default_logger as logger
from up.utils.general.global_flag import ALIGNED_FLAG

# Import from local
from ..ext import roi_align

# TODO use save_for_backward instead


class RoIAlignFunction(Function):

    @staticmethod
    def symbolic(g, features, rois, pooled_h, pooled_w, spatial_scale, sampling_ratio, pooled_mode):
        return g.op(
            "RoiAlign",
            features,
            rois,
            spatial_scale_f=spatial_scale,
            pooled_width_i=pooled_w,
            pooled_height_i=pooled_h,
            pooled_mode_s=pooled_mode,
            sample_num_i=sampling_ratio)

    @staticmethod
    def forward(self, features, rois, pooled_h, pooled_w, spatial_scale, sampling_ratio, pooled_mode):
        self.save_for_backward(features, rois)
        self.pooled_h = pooled_h
        self.pooled_w = pooled_w
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio
        self.pooled_mode = pooled_mode

        batch_size, num_channels, data_height, data_width = features.size()
        num_rois = rois.size(0)

        output = features.new(num_rois, num_channels, pooled_h, pooled_w).zero_()
        assert features.is_contiguous() and rois.is_contiguous()
        if output.numel() == 0:
            return output

        forward_fn = roi_align.forward_avg_cuda
        if pooled_mode == 'MAX':
            forward_fn = roi_align.forward_max_cuda
        if not features.is_cuda:
            logger.warning(
                '---CPU version of RoIAlignPooling is a dummpy function, which is used to support tocaffe')
            forward_fn = roi_align.forward_cpu

        forward_fn(ALIGNED_FLAG.aligned, pooled_h, pooled_w, spatial_scale, sampling_ratio, features, rois, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        grad_output = grad_output.data
        feature, rois = self.saved_tensors

        batch_size, num_channels, data_height, data_width = feature.shape
        grad_input = feature.new(batch_size, num_channels, data_height, data_width).zero_()
        if grad_output.data.numel() == 0:
            return grad_input, None, None, None, None, None, None
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        assert(grad_output.is_contiguous())
        assert(rois.is_contiguous())

        if self.pooled_mode == 'AVG':
            backward_fn = roi_align.backward_avg_cuda
            backward_fn(
                ALIGNED_FLAG.aligned, self.pooled_h, self.pooled_w, self.spatial_scale,
                self.sampling_ratio, grad_output, rois, grad_input)
        elif self.pooled_mode == 'MAX':
            backward_fn = roi_align.backward_max_cuda
            backward_fn(
                ALIGNED_FLAG.aligned, self.pooled_h, self.pooled_w, self.spatial_scale,
                self.sampling_ratio, grad_output, rois, grad_input, feature)
        else:
            raise KeyError

        return grad_input, None, None, None, None, None, None


class RoIAlignPool(torch.nn.Module):

    def __init__(self, pooled_h, pooled_w, sampling_ratio, pooled_mode='AVG', spatial_scale=None):
        super(RoIAlignPool, self).__init__()

        self.pooled_w = int(pooled_w)
        self.pooled_h = int(pooled_h)
        self.sampling_ratio = int(sampling_ratio)
        self.pooled_mode = pooled_mode

        if spatial_scale is not None:
            logger.warning('spatial_scale is deprecated when initializing RoIAlignPool'
                           'we move spatial_scale to forward arguments `stride` for flexiability')

    def forward(self, rois, feature, stride):
        """
        Arguments:
            rois: [N, >=5] (batch_idx, x1, y1, x2, y2)

        Notes:
            1. rois must be N*5 dim
            2. in fp16 mode, feature.dtype is fp16, but rois.dtype may not
            3. tensor must be contiguous before passing to the C code
        """
        rois = rois[:, :5].contiguous().to(dtype=feature.dtype)
        feature = feature.contiguous()
        assert rois.shape[1] == 5, rois.shape
        spatial_scale = 1.0 / stride
        return RoIAlignFunction.apply(
            feature, rois, self.pooled_h, self.pooled_w, spatial_scale, self.sampling_ratio, self.pooled_mode)

    def __repr__(self):
        s = '{name} ({pooled_h}, {pooled_w}, {sampling_ratio}, {pooled_mode})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    @classmethod
    def from_params(cls, params):
        # 7
        pooled_h = pooled_w = params['pool_size']
        # 2
        sampling_ratio = params['sampling_ratio']
        if 'pool_mode' not in params:
            pooled_mode = 'AVG'
        else:
            pooled_mode = params['pool_mode']
        return cls(pooled_h, pooled_w, sampling_ratio, pooled_mode)
