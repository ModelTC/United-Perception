#ifndef IOU3D_NMS_H
#define IOU3D_NMS_H

#include <torch/serialize/tensor.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

int boxes_overlap_bev_gpu(at::Tensor boxes_a, at::Tensor boxes_b, at::Tensor ans_overlap);
int boxes_iou_bev_gpu(at::Tensor boxes_a, at::Tensor boxes_b, at::Tensor ans_iou);
int nms_gpu(at::Tensor boxes, at::Tensor keep, float nms_overlap_thresh);
int nms_normal_gpu(at::Tensor boxes, at::Tensor keep, float nms_overlap_thresh);
int boxes_iou_bev_cpu(at::Tensor boxes_a_tensor, at::Tensor boxes_b_tensor, at::Tensor ans_iou_tensor);

#endif
