#include "psroi_align/psroi_align.h"

using at::Tensor;

int psroi_align_forward_cuda(int pooled_height,
                               int pooled_width,
                               int output_dim,
                               float spatial_scale,
                               int sampling_ratio,
                               Tensor features,
                               Tensor rois,
                               Tensor output,
                               Tensor mapping_channel)
{
    // Grab the input tensor
    CHECK_INPUT(features);
    CHECK_INPUT(rois);
    CHECK_INPUT(output);
    CHECK_INPUT(mapping_channel);
    // Number of ROIs
    int num_rois = rois.size(0);
    int size_rois = rois.size(1);

    assert(size_rois == 5);
    if (size_rois != 5)
    {
        exit(1);
        return 0;
    }

    // data height
    int data_height = features.size(2);
    // data width
    int data_width = features.size(3);
    // Number of channels
    int num_channels = features.size(1);

    PSROIAlignForwardLaucher(
        features, spatial_scale, num_rois, output_dim, size_rois, data_height,
        data_width, num_channels, pooled_height,
        pooled_width, sampling_ratio, rois,
        output, mapping_channel);

    return 1;
}

int psroi_align_backward_cuda(int pooled_height,
                                int pooled_width,
                                int output_dim,
                                float spatial_scale,
                                int sampling_ratio,
                                Tensor top_grad,
                                Tensor rois,
                                Tensor bottom_grad,
                                Tensor mapping_channel)
{
    // Grab the input tensor
    CHECK_INPUT(top_grad);
    CHECK_INPUT(rois);
    CHECK_INPUT(bottom_grad);
    CHECK_INPUT(mapping_channel);

    // Number of ROIs
    int num_rois = rois.size(0);
    int size_rois = rois.size(1);
    assert(size_rois == 5);
    if (size_rois != 5)
    {
        exit(1);
        return 0;
    }

    // batch size
    int batch_size = bottom_grad.size(0);
    // if (batch_size != 1)
    // {
    //     return 0;
    // }
    // data height
    int data_height = bottom_grad.size(2);
    // data width
    int data_width = bottom_grad.size(3);
    // Number of channels
    int num_channels = bottom_grad.size(1);

    PSROIAlignBackwardLaucher(
        top_grad, spatial_scale, batch_size, num_rois, output_dim, size_rois,
        data_height, data_width, num_channels, pooled_height,
        pooled_width, sampling_ratio, rois,
        bottom_grad, mapping_channel);

    return 1;
}

