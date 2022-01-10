#ifndef PSROIALIGN_H_
#define PSROIALIGN_H_

#include <ATen/ATen.h>
#include <cmath>
#include <cassert>

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


// for tocaffe support
int psroi_align_forward(int pooled_height,
                          int pooled_width,
                          int output_dim,
                          float spatial_scale,
                          int sampling_ratio,
                          at::Tensor features,
                          at::Tensor rois,
                          at::Tensor output,
                          at::Tensor mapping_channel);

int PSROIAlignForwardLaucher(at::Tensor bottom_data, const float spatial_scale,
                             const int num_rois, const int output_dim,
                             const int size_rois, const int height,
                             const int width, const int channels,
                             const int pooled_height, const int pooled_width,
                             const float sampling_ratio, at::Tensor bottom_rois,
                             at::Tensor top_data, at::Tensor mapping_channel);

int PSROIAlignBackwardLaucher(at::Tensor top_diff, const float spatial_scale,
                              const int batch_size, const int num_rois,
                              const int output_dim, const int size_rois,
                              const int height, const int width,
                              const int channels, const int pooled_height,
                              const int pooled_width,
                              const float sampling_ratio, at::Tensor bottom_rois,
                              at::Tensor bottom_diff, at::Tensor mapping_channel);

int psroi_align_forward_cuda(int pooled_height,
                               int pooled_width,
                               int output_dim,
                               float spatial_scale,
                               int sampling_ratio,
                               at::Tensor features,
                               at::Tensor rois,
                               at::Tensor output,
                               at::Tensor mapping_channel);

int psroi_align_backward_cuda(int pooled_height,
                                int pooled_width,
                                int output_dim,
                                float spatial_scale,
                                int sampling_ratio,
                                at::Tensor top_grad,
                                at::Tensor rois,
                                at::Tensor bottom_grad,
                                at::Tensor mapping_channel);

#endif
