#ifndef NMS_H_
#define NMS_H_

#include <ATen/ATen.h>
#include <cmath>
#include <cstdio>
#include <cfloat>

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

int gpu_nms(at::Tensor keep, at::Tensor num_out, at::Tensor boxes, float nms_overlap_thresh, int offset);
int cpu_nms(at::Tensor keep_out, at::Tensor num_out, at::Tensor boxes, at::Tensor order,
            at::Tensor areas, float nms_overlap_thresh, int offset);

#endif
