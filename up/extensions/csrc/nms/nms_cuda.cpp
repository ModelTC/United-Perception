// ------------------------------------------------------------------
// Faster R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Shaoqing Ren
// ------------------------------------------------------------------
//
#include "nms/nms.h"

using at::Tensor;

#define DIVUP(x, y) (((x) + (y) - 1) / (y))
int const threadsPerBlock = sizeof(unsigned long long) * 8;

void _nms(int boxes_num, Tensor boxes_dev,
          Tensor mask_dev, float nms_overlap_thresh, int offset);


int gpu_nms(Tensor keep, Tensor num_out, Tensor boxes, float nms_overlap_thresh, int offset) {
  keep = keep.contiguous();
  boxes = boxes.contiguous();
  auto options_gpu =
  at::TensorOptions()
    .dtype(at::kLong)
    .layout(at::kStrided)
    .device({at::kCUDA})
    .requires_grad(false);
  auto options_cpu =
  at::TensorOptions()
    .dtype(at::kLong)
    .layout(at::kStrided)
    .device({at::kCPU})
    .requires_grad(false);
  // Number of ROIs
  int boxes_num = boxes.size(0);

  const int col_blocks = DIVUP(boxes_num, threadsPerBlock);
  Tensor mask = at::zeros({boxes_num, col_blocks}, options_gpu);

  _nms(boxes_num, boxes, mask, nms_overlap_thresh, offset);

  Tensor mask_cpu = mask.to(at::kCPU);
  unsigned long long * mask_flat = (unsigned long long *)mask_cpu.data_ptr<long>();
  Tensor remv_cpu = at::zeros({col_blocks}, options_cpu);
  unsigned long long *remv_cpu_flat = (unsigned long long *)remv_cpu.data_ptr<long>();

  long * keep_flat = keep.data_ptr<long>();
  long num_to_keep = 0;
  // printf("in cir\n");
  int i, j;
  for (i = 0; i < boxes_num; i++) {
    int nblock = i / threadsPerBlock;
    int inblock = i % threadsPerBlock;

    if (!(remv_cpu_flat[nblock] & (1ULL << inblock))) {
      keep_flat[num_to_keep++] = i;
      // unsigned long long *p = &mask_flat[0] + i * col_blocks;
      for (j = nblock; j < col_blocks; j++) {
        remv_cpu_flat[j] |= mask_flat[i * col_blocks + j];
      }
    }
  }
  // printf("out cir\n");
  long * num_out_flat = num_out.data_ptr<long>();
  // printf("***\n");
  num_out_flat[0] = num_to_keep;
  // printf("final\n");
  return 1;
}


int cpu_nms(Tensor keep_out, Tensor num_out, Tensor boxes, Tensor order, Tensor areas, float nms_overlap_thresh, int offset) {
    long boxes_num = boxes.size(0);
    long boxes_dim = boxes.size(1);
    float * boxes_flat = boxes.data_ptr<float>();
    long * order_flat = order.data_ptr<long>();
    float * areas_flat = areas.data_ptr<float>();

    Tensor suppressed = at::zeros({boxes_num}, at::dtype(at::kByte));
    unsigned char * suppressed_flat = suppressed.data_ptr< unsigned char >();
    // nominal indices
    int i, j;
    // sorted indices
    int _i, _j;
    // temp variables for box i's (the box currently under consideration)
    float ix1, iy1, ix2, iy2, iarea;
    // variables for computing overlap with box j (lower scoring box)
    float xx1, yy1, xx2, yy2;
    float w, h;
    float inter, ovr;

    long num_to_keep = 0;
    for (_i=0; _i < boxes_num; ++_i) {
        i = order_flat[_i];
        if (suppressed_flat[i] == 1) {
            continue;
        }
        keep_out[num_to_keep++] = i;
        ix1 = boxes_flat[i * boxes_dim];
        iy1 = boxes_flat[i * boxes_dim + 1];
        ix2 = boxes_flat[i * boxes_dim + 2];
        iy2 = boxes_flat[i * boxes_dim + 3];
        iarea = areas_flat[i];
        for (_j = _i + 1; _j < boxes_num; ++_j) {
            j = order_flat[_j];
            if (suppressed_flat[j] == 1) {
                continue;
            }
            xx1 = fmaxf(ix1, boxes_flat[j * boxes_dim]);
            yy1 = fmaxf(iy1, boxes_flat[j * boxes_dim + 1]);
            xx2 = fminf(ix2, boxes_flat[j * boxes_dim + 2]);
            yy2 = fminf(iy2, boxes_flat[j * boxes_dim + 3]);
            w = fmaxf(0.0, xx2 - xx1 + offset);
            h = fmaxf(0.0, yy2 - yy1 + offset);
            inter = w * h;
            ovr = inter / (iarea + areas_flat[j] - inter);
            if (ovr >= nms_overlap_thresh) {
                suppressed_flat[j] = 1;
            }
        }
    }
    long * num_out_flat = num_out.data_ptr<long>();
    num_out_flat[0] = num_to_keep;
    return 1;
}
