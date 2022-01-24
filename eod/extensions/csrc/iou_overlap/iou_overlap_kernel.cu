// ------------------------------------------------------------------
// Sensetime DSK
// Written by Liang Liu
// ------------------------------------------------------------------
#include "iou_overlap/iou_overlap.h"

using at::Tensor;
using at::Half;

#define DIVUP(m, n) ((m) / (m) + ((m) % (n) > 0))

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)
/*
template <typename scalar_t>
__device__ inline scalar_t devIoU(scalar_t const * const a, scalar_t const * const b) {
  scalar_t left = fmaxf(a[0], b[0]), right = fminf(a[2], b[2]);
  scalar_t top = fmaxf(a[1], b[1]), bottom = fminf(a[3], b[3]);
  scalar_t width = fmaxf(right - left + 1, 0.f), height = fmaxf(bottom - top + 1, 0.f);
  scalar_t interS = width * height;
  scalar_t Sa = (a[2] - a[0] + 1) * (a[3] - a[1] + 1);
  scalar_t Sb = (b[2] - b[0] + 1) * (b[3] - b[1] + 1);
  return interS / (Sa + Sb - interS);
}
*/
template <typename scalar_t>
__global__ void IOUOverlapKernel(
    const scalar_t* bbox1,
    const scalar_t* bbox2,
    const int size_bbox,
    const int num_bbox1,
    const int num_bbox2,
    scalar_t* top_data,
    const int mode){
    CUDA_KERNEL_LOOP(index, num_bbox1 * num_bbox2){
        int b1 = index / num_bbox2;
        int b2 = index % num_bbox2;

        int base1 = b1 * size_bbox;
        scalar_t b1_x1 = bbox1[base1];
        scalar_t b1_y1 = bbox1[base1 + 1];
        scalar_t b1_x2 = bbox1[base1 + 2];
        scalar_t b1_y2 = bbox1[base1 + 3];
        scalar_t b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1); 

        int base2 = b2 * size_bbox;
        scalar_t b2_x1 = bbox2[base2];
        scalar_t b2_y1 = bbox2[base2 + 1];
        scalar_t b2_x2 = bbox2[base2 + 2];
        scalar_t b2_y2 = bbox2[base2 + 3];
        scalar_t b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1); 

        scalar_t left = fmaxf(b1_x1, b2_x1), right  = fminf(b1_x2, b2_x2);
        scalar_t top  = fmaxf(b1_y1, b2_y1), bottom = fminf(b1_y2, b2_y2);
        scalar_t width = fmaxf(right - left + 1, 0.f), height = fmaxf(bottom - top
                                                                   + 1, 0.f);
        scalar_t interS = width * height;
        scalar_t baseS = 1.0;
        if (mode == 0) {
          baseS = fmaxf(b1_area + b2_area - interS, 1.0);
        } else if (mode == 1){
          baseS = fmaxf(b1_area, 1.0);
        } else {
          baseS = fmaxf(b2_area, 1.0);
        }
        top_data[b1 * num_bbox2 + b2] = interS / baseS;
    }
}

int IOUOverlap(
    Tensor bboxes1_data, 
    Tensor bboxes2_data, 
    const int size_bbox,
    const int num_bbox1,
    const int num_bbox2,
    Tensor top_data,
    const int mode)
{
    const int kThreadsPerBlock = 1024;
    int output_size = num_bbox1 * num_bbox2;
    //int output_size = num_bbox1;
    cudaError_t err;

    err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf( stderr, "%s#%d: cudaCheckError() failed : %s\n", __FILE__,
            __LINE__, cudaGetErrorString( err ) );
        exit( -1 );
    }

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(bboxes1_data.scalar_type(), "IOUOverlap_cuda", ([&] {
        IOUOverlapKernel<scalar_t><<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock>>>(
                     bboxes1_data.data_ptr<scalar_t>(), 
                     bboxes2_data.data_ptr<scalar_t>(), 
                     size_bbox, num_bbox1, num_bbox2, 
                     top_data.data_ptr<scalar_t>(),
                     mode);
    }));
    err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf( stderr, "%s#%d: cudaCheckError() failed : %s\n", __FILE__,
                __LINE__, cudaGetErrorString( err ) );
        exit( -1 );
    }

    return 1;
}

