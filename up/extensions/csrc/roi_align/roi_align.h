#ifndef ROIALIGN_H_
#define ROIALIGN_H_

#include <ATen/ATen.h>
#include <vector>
#include <cmath>

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// for tocaffe
int roi_align_forward(
    bool aligned, 
    int aligned_height, int aligned_width,
    float spatial_scale, int sampling_ratio,
    at::Tensor features, at::Tensor rois, at::Tensor output);

int roi_align_avg_forward_cuda(
    bool aligned, int aligned_height, int aligned_width,
    float spatial_scale, int sampling_ratio,
    at::Tensor features, at::Tensor rois, at::Tensor output);

int roi_align_avg_backward_cuda(
    bool aligned, int aligned_height, int aligned_width,
    float spatial_scale, int sampling_ratio,
    at::Tensor top_grad, at::Tensor rois, at::Tensor bottom_grad);

int ROIAlignAvgBackwardLaucher(
    const bool aligned, at::Tensor top_diff, const float spatial_scale,
    const int batch_size, const int num_rois, const int height, const int width,
    const int channels, const int aligned_height, const int aligned_width,
    const int sampling_ratio,
    at::Tensor bottom_rois, at::Tensor bottom_diff);

int ROIAlignAvgForwardLaucher(
    const bool aligned, at::Tensor bottom_data, const float spatial_scale,
    const int num_rois, const int height, const int width,
    const int channels, const int aligned_height, const int aligned_width,  
    const int sampling_ratio,
    at::Tensor bottom_rois, at::Tensor top_data);

/*
int roi_align_max_forward_cuda(
    bool aligned, int aligned_height, int aligned_width,
    float spatial_scale, int sampling_ratio,
    at::Tensor features, at::Tensor rois, at::Tensor output);

int roi_align_max_backward_cuda(
    bool aligned, int aligned_height, int aligned_width,
    float spatial_scale, int sampling_ratio,
    at::Tensor top_grad, at::Tensor rois, at::Tensor bottom_grad, at::Tensor bottom_data);
*/

#endif
