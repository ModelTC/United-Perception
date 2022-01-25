#ifndef IOUOVERLAP_H_
#define IOUOVERLAP_H_

#include <ATen/ATen.h>
#include <cmath>
#include <cstdio>

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

int IOUOverlap(at::Tensor bboxes1_data, 
               at::Tensor bboxes2_data, 
               const int size_bbox,
               const int num_bbox1,
               const int num_bbox2,
               at::Tensor top_data,
               const int mode);

void gpu_iou_overlaps(at::Tensor bboxes1, at::Tensor bboxes2, at::Tensor output, const int mode);

#endif
