// ------------------------------------------------------------------
// Faster R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Shaoqing Ren
// ------------------------------------------------------------------
#include "nms/nms.h"

using at::Tensor;
using at::Half;

#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))
int const threadsPerBlock = sizeof(unsigned long long int) * 8;

template <typename scalar_t>
__device__ inline scalar_t devIoU(scalar_t const * const a, scalar_t const * const b, const int offset) {
  scalar_t left = fmaxf(a[0], b[0]), right = fminf(a[2], b[2]);
  scalar_t top = fmaxf(a[1], b[1]), bottom = fminf(a[3], b[3]);
  scalar_t width = fmaxf(right - left + offset, 0.f), height = fmaxf(bottom - top + offset, 0.f);
  scalar_t interS = width * height;
  scalar_t Sa = (a[2] - a[0] + offset) * (a[3] - a[1] + offset);
  scalar_t Sb = (b[2] - b[0] + offset) * (b[3] - b[1] + offset);
  return interS / (Sa + Sb - interS);
}


template <typename scalar_t>
__global__ void nms_kernel(const int n_boxes, const scalar_t nms_overlap_thresh,
                           const scalar_t *dev_boxes, unsigned long long *dev_mask, const int offset) {
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;

  // if (row_start > col_start) return;

  const int row_size =
        fminf(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
  const int col_size =
        fminf(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

  __shared__ scalar_t block_boxes[threadsPerBlock * 5];
  if (threadIdx.x < col_size) {
    block_boxes[threadIdx.x * 5 + 0] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 0];
    block_boxes[threadIdx.x * 5 + 1] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 1];
    block_boxes[threadIdx.x * 5 + 2] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 2];
    block_boxes[threadIdx.x * 5 + 3] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 3];
    block_boxes[threadIdx.x * 5 + 4] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 4];
  }
  __syncthreads();

  if (threadIdx.x < row_size) {
    const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
    const scalar_t *cur_box = dev_boxes + cur_box_idx * 5;
    int i = 0;
    unsigned long long int t = 0;
    int start = 0;
    if (row_start == col_start) {
      start = threadIdx.x + 1;
    }
    for (i = start; i < col_size; i++) {
      if (devIoU(cur_box, block_boxes + i * 5, offset) > nms_overlap_thresh) {
        t |= 1ULL << i;
      }
    }
    const int col_blocks = DIVUP(n_boxes, threadsPerBlock);
    dev_mask[cur_box_idx * col_blocks + col_start] = t;
  }
}


void _nms(int boxes_num, Tensor boxes_dev,
          Tensor mask_dev, float nms_overlap_thresh, int offset) {

  dim3 blocks(DIVUP(boxes_num, threadsPerBlock),
              DIVUP(boxes_num, threadsPerBlock));
  dim3 threads(threadsPerBlock);
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(boxes_dev.scalar_type(), "roi_pooling_forward_cuda", ([&] {
    nms_kernel<scalar_t><<<blocks, threads>>>(boxes_num,
                                  nms_overlap_thresh,
                                  boxes_dev.data_ptr<scalar_t>(),
                                  (unsigned long long *)mask_dev.data_ptr<long>(),
                                  offset);
  }));
}

