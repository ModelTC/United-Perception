#include "roi_align/roi_align.h"

using at::Tensor;

int roi_align_avg_forward_cuda(bool aligned, int aligned_height, int aligned_width,
        float spatial_scale, int sampling_ratio,
        Tensor features, Tensor rois, Tensor output)
{
    // Grab the input tensor
    CHECK_INPUT(features);
    CHECK_INPUT(rois);
    CHECK_INPUT(output);
    // Number of ROIs
    int num_rois = rois.size(0);
    int size_rois = rois.size(1);
    if (size_rois != 5)
    {
        exit(1);
        return 1;
    }

    // data height
    int data_height = features.size(2);
    // data width
    int data_width = features.size(3);
    // Number of channels
    int num_channels = features.size(1);


    ROIAlignAvgForwardLaucher(
        aligned, features, spatial_scale, num_rois, data_height,
        data_width, num_channels, aligned_height,
        aligned_width, sampling_ratio, rois,
        output);

    return 0;
}

int roi_align_avg_backward_cuda(bool aligned, int aligned_height, int aligned_width,
        float spatial_scale, int sampling_ratio,
        Tensor top_grad, Tensor rois, Tensor bottom_grad)
{
    // Grab the input tensor
    CHECK_INPUT(top_grad);
    CHECK_INPUT(rois);
    CHECK_INPUT(bottom_grad);


    // Number of ROIs
    int num_rois = rois.size(0);
    int size_rois = rois.size(1);
    if (size_rois != 5)
    {
        exit(1);
        return 1;
    }

    // batch size
    int batch_size = bottom_grad.size(0);
    // data height
    int data_height = bottom_grad.size(2);
    // data width
    int data_width = bottom_grad.size(3);
    // Number of channels
    int num_channels = bottom_grad.size(1);

    ROIAlignAvgBackwardLaucher(
        aligned, top_grad, spatial_scale, batch_size, num_rois, data_height,
        data_width, num_channels, aligned_height,
        aligned_width, sampling_ratio, rois,
        bottom_grad);

    return 0;
}