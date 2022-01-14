#include "psroi_pooling/psroi_pooling.h"

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

#ifndef CAFFE_COMMON_CUH_
#define CAFFE_COMMON_CUH_

#include <cuda.h>

  #if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600

  #else//performence loss
      static __inline__ __device__ double atomicAdd(double *address, double val) {
        unsigned long long int* address_as_ull = (unsigned long long int*)address;
        unsigned long long int old = *address_as_ull, assumed;
        if (val==0.0)
          return __longlong_as_double(old);
        do {
          assumed = old;
          old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val +__longlong_as_double(assumed)));
        } while (assumed != old);
        return __longlong_as_double(old);
      }
  #endif
#endif

static __inline__ __device__ at::Half atomicAdd(at::Half* address, at::Half val) {
  unsigned int *aligned = (unsigned int*)((size_t)address - ((size_t)address & 2));
  unsigned int old = *aligned;
  unsigned int assumed;
  unsigned short old_as_us;
  do {
    assumed = old;
    old_as_us = (unsigned short)((size_t)address & 2 ? old >> 16 : old & 0xffff);
#if __CUDACC_VER_MAJOR__ >= 9
    half sum = __float2half_rn(__half2float(__ushort_as_half(old_as_us)) + float(val));
    unsigned short sum_as_us = __half_as_ushort(sum);
#else
    unsigned short sum_as_us = __float2half_rn(__half2float(old_as_us) + float(val));
#endif
    unsigned int sum_as_ui = (size_t)address & 2 ? (sum_as_us << 16) | (old & 0xffff)
                                                 : (old & 0xffff0000) | sum_as_us;
    old = atomicCAS(aligned, assumed, sum_as_ui);
  } while(assumed != old);
  //__half_raw raw = {old_as_us};
  //return at::Half(raw);
  return at::Half({__ushort_as_half(old_as_us)});
};

template <typename scalar_t>
__global__ void PSROIPoolingForward(
    const int nthreads,
    const scalar_t* bottom_data,
    const scalar_t spatial_scale,
    const int channels,
    const int height, const int width,
    const int pooled_height, const int pooled_width,
    const scalar_t* bottom_rois,
    const int output_dim,
    const int group_size,
    scalar_t* top_data,
    int* mapping_channel,
    const int shape) {
    CUDA_KERNEL_LOOP(index, nthreads) {
      // The output is in order (n, ctop, ph, pw)
      int pw = index % pooled_width;
      int ph = (index / pooled_width) % pooled_height;
      int ctop = (index / pooled_width / pooled_height) % output_dim;
      int n = index / pooled_width / pooled_height / output_dim;

      // [start, end) interval for spatial sampling
      // liqq 2016/09/25
      // bottom_rois += n * shape;

      int roi_batch_ind = (int)bottom_rois[n * shape + 0];
      scalar_t roi_start_w = static_cast<scalar_t>(round(bottom_rois[n * shape + 1])) * spatial_scale;
      scalar_t roi_start_h = static_cast<scalar_t>(round(bottom_rois[n * shape + 2])) * spatial_scale;
      scalar_t roi_end_w = static_cast<scalar_t>(round(bottom_rois[n * shape + 3]) + 1.) * spatial_scale;
      scalar_t roi_end_h = static_cast<scalar_t>(round(bottom_rois[n * shape + 4]) + 1.) * spatial_scale;

      // Force too small ROIs to be 1x1
      scalar_t roi_width = max(roi_end_w - roi_start_w, 0.1); //avoid 0
      scalar_t roi_height = max(roi_end_h - roi_start_h, 0.1);

      // Compute w and h at bottom
      scalar_t bin_size_h = roi_height / static_cast<scalar_t>(pooled_height);
      scalar_t bin_size_w = roi_width / static_cast<scalar_t>(pooled_width);

      int hstart = (int)floor(static_cast<scalar_t>(ph) * bin_size_h
                          + roi_start_h);
      int wstart = (int)floor(static_cast<scalar_t>(pw)* bin_size_w
                          + roi_start_w);
      int hend = (int)ceil(static_cast<scalar_t>(ph + 1) * bin_size_h
                        + roi_start_h);
      int wend = (int)ceil(static_cast<scalar_t>(pw + 1) * bin_size_w
                        + roi_start_w);
      // Add roi offsets and clip to input boundaries
      hstart = min(max(hstart, 0), height);
      hend = min(max(hend, 0), height);
      wstart = min(max(wstart, 0),width);
      wend = min(max(wend, 0), width);
      bool is_empty = (hend <= hstart) || (wend <= wstart);

      int gw = pw;
      int gh = ph;
      int c = (ctop*group_size + gh)*group_size + gw;

      bottom_data += (roi_batch_ind * channels + c) * height * width;
      scalar_t out_sum = 0;
      for (int h = hstart; h < hend; ++h){
        for (int w = wstart; w < wend; ++w){
          int bottom_index = h*width + w;
          out_sum += bottom_data[bottom_index];
        }
      }

      scalar_t bin_area = (hend - hstart)*(wend - wstart);
      top_data[index] = is_empty? scalar_t(0.) : out_sum/bin_area;
      mapping_channel[index] = c;
    }
  }


int PSROIPoolForwardLaucher(
    Tensor bottom_data, const float spatial_scale, const int num_rois, const int output_dim, const int size_rois,
    const int height, const int width, const int channels, const int pooled_height,
    const int pooled_width, Tensor bottom_rois,
    Tensor top_data, Tensor mapping_channel)
{
    const int kThreadsPerBlock = 1024;
    int output_size = num_rois * pooled_height * pooled_width * output_dim;
    cudaError_t err;

    err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf( stderr, "%s#%d: cudaCheckError() failed : %s\n", __FILE__,
                __LINE__, cudaGetErrorString( err ) );
        exit( -1 );
    }
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(bottom_data.scalar_type(), "psroi_pooling_forward_cuda", ([&] {
      PSROIPoolingForward<scalar_t><<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock>>>(
        output_size, bottom_data.data_ptr<scalar_t>(), spatial_scale, channels, height, width, pooled_height,
        pooled_width, bottom_rois.data_ptr<scalar_t>(), output_dim, pooled_height, 
        top_data.data_ptr<scalar_t>(), mapping_channel.data_ptr<int>(), size_rois);
    }));
    // pooled_height == pooled_width == group_size
    err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf( stderr, "%s#%d: cudaCheckError() failed : %s\n", __FILE__,
                __LINE__, cudaGetErrorString( err ) );
        exit( -1 );
    }

    return 1;
}
template <typename scalar_t>
__global__ void PSROIPoolingBackward(
    const int nthreads,
    const scalar_t* top_diff,
    const int* mapping_channel,
    const int num_rois,
    const scalar_t spatial_scale,
    const int channels,
    const int height, const int width,
    const int pooled_height, const int pooled_width,
    const int output_dim,
    scalar_t* bottom_diff,
    const scalar_t* bottom_rois,
    const int shape) {
    CUDA_KERNEL_LOOP(index, nthreads) {
      // The output is in order (n, ctop, ph, pw)
      int pw = index % pooled_width;
      int ph = (index / pooled_width) % pooled_height;
      int n = index / pooled_width / pooled_height / output_dim;

      // [start, end) interval for spatial sampling
      // liqq 2016/09/25
      // bottom_rois += n * shape;

      int roi_batch_ind = (int)bottom_rois[n * shape + 0];
      scalar_t roi_start_w = static_cast<scalar_t>(round(bottom_rois[n * shape + 1])) * spatial_scale;
      scalar_t roi_start_h = static_cast<scalar_t>(round(bottom_rois[n * shape + 2])) * spatial_scale;
      scalar_t roi_end_w = static_cast<scalar_t>(round(bottom_rois[n * shape + 3]) + 1.) * spatial_scale;
      scalar_t roi_end_h = static_cast<scalar_t>(round(bottom_rois[n * shape + 4]) + 1.) * spatial_scale;

      // Force too small ROIs to be 1x1
      scalar_t roi_width = max(roi_end_w - roi_start_w, 0.1); //avoid 0
      scalar_t roi_height = max(roi_end_h - roi_start_h, 0.1);

      // Compute w and h at bottom
      scalar_t bin_size_h = roi_height / static_cast<scalar_t>(pooled_height);
      scalar_t bin_size_w = roi_width / static_cast<scalar_t>(pooled_width);

      int hstart = (int)floor(static_cast<scalar_t>(ph)* bin_size_h
        + roi_start_h);
      int wstart = (int)floor(static_cast<scalar_t>(pw)* bin_size_w
        + roi_start_w);
      int hend = (int)ceil(static_cast<scalar_t>(ph + 1) * bin_size_h
        + roi_start_h);
      int wend = (int)ceil(static_cast<scalar_t>(pw + 1) * bin_size_w
        + roi_start_w);
      // Add roi offsets and clip to input boundaries
      hstart = min(max(hstart, 0), height);
      hend = min(max(hend, 0), height);
      wstart = min(max(wstart, 0), width);
      wend = min(max(wend, 0), width);
      bool is_empty = (hend <= hstart) || (wend <= wstart);

      // Compute c at bottom
      int c = mapping_channel[index];
      scalar_t* offset_bottom_diff = bottom_diff + (roi_batch_ind * channels + c) * height * width;
      scalar_t bin_area = (hend - hstart)*(wend - wstart);
      scalar_t diff_val = is_empty ? scalar_t(0.) : top_diff[index] / bin_area;
      for (int h = hstart; h < hend; ++h){
        for (int w = wstart; w < wend; ++w){
          int bottom_index = h*width + w;
          // caffe_gpu_atomic_add(diff_val, offset_bottom_diff + bottom_index);
          atomicAdd(offset_bottom_diff + bottom_index, diff_val);
        }
      }
    }
  }

int PSROIPoolBackwardLaucher(Tensor top_diff, const float spatial_scale, const int batch_size, const int num_rois,
    const int output_dim, const int size_rois, const int height, const int width, const int channels, const int pooled_height,
    const int pooled_width, Tensor bottom_rois,
    Tensor bottom_diff, Tensor mapping_channel)
{
    const int kThreadsPerBlock = 1024;
    //int output_size = batch_size * height * width * output_dim;
    int output_size = output_dim * pooled_height * pooled_width * num_rois;
    cudaError_t err;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(top_diff.scalar_type(), "psroi_pooling_backward_cuda", ([&] {
      PSROIPoolingBackward<scalar_t><<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock>>>(
        output_size, top_diff.data_ptr<scalar_t>(), mapping_channel.data_ptr<int>(), 
        num_rois, spatial_scale, channels, height, width, pooled_height,
        pooled_width, output_dim, bottom_diff.data_ptr<scalar_t>(), bottom_rois.data_ptr<scalar_t>(), size_rois);
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


