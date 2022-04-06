#ifndef ROIPOINTPOOL3D_H_
#define ROIPOINTPOOL3D_H_

#include <ATen/ATen.h>
#include <cmath>
#include <cassert>
#include <cstdio>
using at::Tensor;


#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

int roipool3d_gpu(at::Tensor xyz, at::Tensor boxes3d, at::Tensor pts_feature, at::Tensor pooled_features, at::Tensor pooled_empty_flag);




#endif