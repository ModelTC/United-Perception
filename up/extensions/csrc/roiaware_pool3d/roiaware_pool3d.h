#ifndef ROIAWAREPOOL3D_H_
#define ROIAWAREPOOL3D_H_

#include <ATen/ATen.h>
#include <cmath>
#include <cassert>
#include <cstdio>
using at::Tensor;


#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

int roiaware_pool3d_gpu(at::Tensor rois, at::Tensor pts, at::Tensor pts_feature, at::Tensor argmax,
    at::Tensor pts_idx_of_voxels, at::Tensor pooled_features, int pool_method);
int roiaware_pool3d_gpu_backward(at::Tensor pts_idx_of_voxels, at::Tensor argmax, at::Tensor grad_out, at::Tensor grad_in, int pool_method);
int points_in_boxes_gpu(at::Tensor boxes_tensor, at::Tensor pts_tensor, at::Tensor box_idx_of_points_tensor);
int points_in_boxes_cpu(at::Tensor boxes_tensor, at::Tensor pts_tensor, at::Tensor pts_indices_tensor);




#endif