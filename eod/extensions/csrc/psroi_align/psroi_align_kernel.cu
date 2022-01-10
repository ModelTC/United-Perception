#include "psroi_align/psroi_align.h"

using at::Half;
using at::Tensor;

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

#else // performence loss
static __inline__ __device__ double atomicAdd(double *address, double val) {
  unsigned long long int *address_as_ull = (unsigned long long int *)address;
  unsigned long long int old = *address_as_ull, assumed;
  if (val == 0.0)
    return __longlong_as_double(old);
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}
#endif
#endif

static __inline__ __device__ at::Half atomicAdd(at::Half *address,
                                                at::Half val) {
  unsigned int *aligned =
      (unsigned int *)((size_t)address - ((size_t)address & 2));
  unsigned int old = *aligned;
  unsigned int assumed;
  unsigned short old_as_us;
  do {
    assumed = old;
    old_as_us =
        (unsigned short)((size_t)address & 2 ? old >> 16 : old & 0xffff);
#if __CUDACC_VER_MAJOR__ >= 9
    half sum =
        __float2half_rn(__half2float(__ushort_as_half(old_as_us)) + float(val));
    unsigned short sum_as_us = __half_as_ushort(sum);
#else
    unsigned short sum_as_us =
        __float2half_rn(__half2float(old_as_us) + float(val));
#endif
    unsigned int sum_as_ui = (size_t)address & 2
                                 ? (sum_as_us << 16) | (old & 0xffff)
                                 : (old & 0xffff0000) | sum_as_us;
    old = atomicCAS(aligned, assumed, sum_as_ui);
  } while (assumed != old);
  //__half_raw raw = {old_as_us};
  // return at::Half(raw);
  return at::Half({__ushort_as_half(old_as_us)});
};

#define CUDA_1D_KERNEL_LOOP(i, n)                                              \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;                   \
       i += blockDim.x * gridDim.x)

/*** Forward ***/
template <typename scalar_t>
__device__ scalar_t bilinear_interpolate(
    const scalar_t *bottom_data, const int height, const int width, scalar_t y,
    scalar_t x, const int index /* index for debug only*/) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    // empty
    return 0;
  }

  if (y <= 0) {
    y = 0;
  }
  if (x <= 0) {
    x = 0;
  }

  int y_low = (int)y;
  int x_low = (int)x;
  int y_high;
  int x_high;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (scalar_t)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (scalar_t)x_low;
  } else {
    x_high = x_low + 1;
  }

  scalar_t ly = y - y_low;
  scalar_t lx = x - x_low;
  scalar_t hy = 1. - ly, hx = 1. - lx;
  // do bilinear interpolation
  scalar_t v1 = bottom_data[y_low * width + x_low];
  scalar_t v2 = bottom_data[y_low * width + x_high];
  scalar_t v3 = bottom_data[y_high * width + x_low];
  scalar_t v4 = bottom_data[y_high * width + x_high];
  scalar_t w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  return val;
}

template <typename scalar_t>
__global__ void
PSROIAlignForward(const int nthreads, const scalar_t *bottom_data,
                  const scalar_t spatial_scale, const int channels,
                  const int height, const int width, const int pooled_height,
                  const int pooled_width, const scalar_t *bottom_rois,
                  const int output_dim, const int group_size,
                  const int sampling_ratio, scalar_t *top_data,
                  int *mapping_channel, const int shape) {
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
    scalar_t roi_start_w =
        static_cast<scalar_t>(bottom_rois[n * shape + 1]) * spatial_scale;
    scalar_t roi_start_h =
        static_cast<scalar_t>(bottom_rois[n * shape + 2]) * spatial_scale;
    scalar_t roi_end_w =
        static_cast<scalar_t>(bottom_rois[n * shape + 3] + 1.) * spatial_scale;
    scalar_t roi_end_h =
        static_cast<scalar_t>(bottom_rois[n * shape + 4] + 1.) * spatial_scale;

    // Force too small ROIs to be 1x1
    scalar_t roi_width = max(roi_end_w - roi_start_w, 0.1); // avoid 0
    scalar_t roi_height = max(roi_end_h - roi_start_h, 0.1);

    // Compute w and h at bottom
    scalar_t bin_size_h = roi_height / static_cast<scalar_t>(pooled_height);
    scalar_t bin_size_w = roi_width / static_cast<scalar_t>(pooled_width);
    int gw = pw;
    int gh = ph;
    int c = (ctop * group_size + gh) * group_size + gw;

    const scalar_t *offset_bottom_data =
        bottom_data + (roi_batch_ind * channels + c) * height * width;

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio > 0)
                             ? sampling_ratio
                             : ceil(roi_height / pooled_height); // e.g., = 2
    int roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

    // use max pooling
    scalar_t maxval = -1E+10;
    int maxidx = -1;

    for (int iy = 0; iy < roi_bin_grid_h; iy++) // e.g., iy = 0, 1
    {
      const scalar_t y =
          roi_start_h + ph * bin_size_h +
          (iy + .5f) * bin_size_h / roi_bin_grid_h; // e.g., 0.5, 1.5
      for (int ix = 0; ix < roi_bin_grid_w; ix++) {
        const scalar_t x = roi_start_w + pw * bin_size_w +
                           (ix + .5f) * bin_size_w / roi_bin_grid_w;

        scalar_t val = bilinear_interpolate(offset_bottom_data, height, width,
                                            y, x, index);
        int bottom_index = iy * roi_bin_grid_w + ix;
        if (val > maxval) {
          maxval = val;
          maxidx = bottom_index;
        }
      }
    }
    top_data[index] = maxval;
    mapping_channel[index] = maxidx;
  }
}

int PSROIAlignForwardLaucher(Tensor bottom_data, const float spatial_scale,
                             const int num_rois, const int output_dim,
                             const int size_rois, const int height,
                             const int width, const int channels,
                             const int pooled_height, const int pooled_width,
                             const float sampling_ratio, Tensor bottom_rois,
                             Tensor top_data, Tensor mapping_channel) {
  const int kThreadsPerBlock = 1024;
  int output_size = num_rois * pooled_height * pooled_width * output_dim;
  cudaError_t err;

  err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "%s#%d: cudaCheckError() failed : %s\n", __FILE__, __LINE__,
            cudaGetErrorString(err));
    exit(-1);
  }
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      bottom_data.scalar_type(), "psroi_align_forward_cuda", ([&] {
        PSROIAlignForward<scalar_t>
            <<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock,
               kThreadsPerBlock>>>(
                output_size, bottom_data.data_ptr<scalar_t>(), spatial_scale,
                channels, height, width, pooled_height, pooled_width,
                bottom_rois.data_ptr<scalar_t>(), output_dim, pooled_height,
                sampling_ratio, top_data.data_ptr<scalar_t>(),
                mapping_channel.data_ptr<int>(), size_rois);
      }));
  // pooled_height == pooled_width == group_size
  err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "%s#%d: cudaCheckError() failed : %s\n", __FILE__, __LINE__,
            cudaGetErrorString(err));
    exit(-1);
  }

  return 1;
}

/*** Backward ***/
template <typename scalar_t>
inline __device__ scalar_t gpu_atomic_add(scalar_t val, scalar_t *address);
template <typename scalar_t>
inline __device__ scalar_t gpu_atomic_add(scalar_t val, scalar_t *address) {
  return atomicAdd(address, val);
}
template <typename scalar_t>
__device__ void bilinear_interpolate_gradient(
    const int height, const int width, scalar_t y, scalar_t x, scalar_t &w1,
    scalar_t &w2, scalar_t &w3, scalar_t &w4, int &x_low, int &x_high,
    int &y_low, int &y_high, const int index /* index for debug only*/) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    // empty
    w1 = w2 = w3 = w4 = 0.;
    x_low = x_high = y_low = y_high = -1;
    return;
  }

  if (y <= 0) {
    y = 0;
  }
  if (x <= 0) {
    x = 0;
  }

  y_low = (int)y;
  x_low = (int)x;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (scalar_t)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (scalar_t)x_low;
  } else {
    x_high = x_low + 1;
  }

  scalar_t ly = y - y_low;
  scalar_t lx = x - x_low;
  scalar_t hy = 1. - ly, hx = 1. - lx;

  w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  return;
}

template <typename scalar_t>
__global__ void PSROIAlignBackward(
    const int nthreads, const scalar_t *top_diff, const int *mapping_channel,
    const scalar_t spatial_scale, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int output_dim, const int group_size, const int sampling_ratio,
    scalar_t *bottom_diff, const scalar_t *bottom_rois, const int shape) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // The output is in order (n, ctop, ph, pw)
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int ctop = (index / pooled_width / pooled_height) % output_dim;
    int n = index / pooled_width / pooled_height / output_dim;

    // [start, end) interval for spatial sampling
    // liqq 2016/09/25
    // bottom_rois += n * shape;
    // Do not using rounding; this implementation detail is critical
    int roi_batch_ind = (int)bottom_rois[n * shape + 0];
    scalar_t roi_start_w =
        static_cast<scalar_t>(bottom_rois[n * shape + 1]) * spatial_scale;
    scalar_t roi_start_h =
        static_cast<scalar_t>(bottom_rois[n * shape + 2]) * spatial_scale;
    scalar_t roi_end_w =
        static_cast<scalar_t>(bottom_rois[n * shape + 3] + 1.) * spatial_scale;
    scalar_t roi_end_h =
        static_cast<scalar_t>(bottom_rois[n * shape + 4] + 1.) * spatial_scale;

    // Force too small ROIs to be 1x1
    scalar_t roi_width = max(roi_end_w - roi_start_w, 0.1); // avoid 0
    scalar_t roi_height = max(roi_end_h - roi_start_h, 0.1);

    // Compute w and h at bottom
    scalar_t bin_size_h = roi_height / static_cast<scalar_t>(pooled_height);
    scalar_t bin_size_w = roi_width / static_cast<scalar_t>(pooled_width);

    int gw = pw;
    int gh = ph;
    int c = (ctop * group_size + gh) * group_size + gw;

    scalar_t *offset_bottom_diff =
        bottom_diff + (roi_batch_ind * channels + c) * height * width;

    int top_offset = (n * output_dim + ctop) * pooled_height * pooled_width;
    scalar_t top_diff_this_bin =
        top_diff[top_offset + ph * pooled_width + pw];

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio > 0)
                             ? sampling_ratio
                             : ceil(roi_height / pooled_height); // e.g. = 2
    int roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

    int maxidx = mapping_channel[top_offset + ph * pooled_width + pw];
    int iy = maxidx / roi_bin_grid_w;
    int ix = maxidx % roi_bin_grid_w;

    scalar_t y = roi_start_h + ph * bin_size_h +
                 static_cast<float>(iy + .5f) * bin_size_h /
                     static_cast<float>(roi_bin_grid_h); // e.g. 0.5, 1.5
    scalar_t x = roi_start_w + pw * bin_size_w +
                 static_cast<float>(ix + .5f) * bin_size_w /
                     static_cast<float>(roi_bin_grid_w);

    scalar_t w1, w2, w3, w4;
    int x_low, x_high, y_low, y_high;

    // bilinear_interpolation_gradient
    bilinear_interpolate_gradient(height, width, y, x, w1, w2, w3, w4, x_low,
                                  x_high, y_low, y_high, index);

    scalar_t g1 = top_diff_this_bin * w1;
    scalar_t g2 = top_diff_this_bin * w2;
    scalar_t g3 = top_diff_this_bin * w3;
    scalar_t g4 = top_diff_this_bin * w4;

    if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
      gpu_atomic_add<scalar_t>(g1, offset_bottom_diff + y_low * width + x_low);
      gpu_atomic_add<scalar_t>(g2, offset_bottom_diff + y_low * width + x_high);
      gpu_atomic_add<scalar_t>(g3, offset_bottom_diff + y_high * width + x_low);
      gpu_atomic_add<scalar_t>(g4,
                               offset_bottom_diff + y_high * width + x_high);
    }
  }
}

int PSROIAlignBackwardLaucher(Tensor top_diff, const float spatial_scale,
                              const int batch_size, const int num_rois,
                              const int output_dim, const int size_rois,
                              const int height, const int width,
                              const int channels, const int pooled_height,
                              const int pooled_width,
                              const float sampling_ratio, Tensor bottom_rois,
                              Tensor bottom_diff, Tensor mapping_channel) {
  const int kThreadsPerBlock = 1024;
  // int output_size = batch_size * height * width * output_dim;
  int output_size = output_dim * pooled_height * pooled_width * num_rois;
  cudaError_t err;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      top_diff.scalar_type(), "psroi_align_backward_cuda", ([&] {
        PSROIAlignBackward<scalar_t>
            <<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock,
               kThreadsPerBlock>>>(output_size, top_diff.data_ptr<scalar_t>(),
                                   mapping_channel.data_ptr<int>(), spatial_scale,
                                   channels, height, width, pooled_height,
                                   pooled_width, output_dim, pooled_height,
                                   sampling_ratio, bottom_diff.data_ptr<scalar_t>(),
                                   bottom_rois.data_ptr<scalar_t>(), size_rois);
      }));
  err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "%s#%d: cudaCheckError() failed : %s\n", __FILE__, __LINE__,
            cudaGetErrorString(err));
    exit(-1);
  }

  return 1;
}
