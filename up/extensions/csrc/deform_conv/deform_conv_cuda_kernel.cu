#include "deform_conv/deformable_conv.h"

using at::Tensor;
using at::Half;

#define CUDA_KERNEL_LOOP(i, n)                                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);                 \
       i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 1024;

inline int GET_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

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
__device__ scalar_t deformable_im2col_bilinear(const scalar_t *bottom_data,
                                               const int data_width,
                                               const int height, const int width,
                                               scalar_t h, scalar_t w) {

  int h_low = floor(h);
  int w_low = floor(w);
  int h_high;
  int w_high;
  if (h_low >= height - 1) {
    h_high = h_low = height - 1;
    h = (scalar_t)h_low;
  } else {
    h_high = h_low + 1;
  }

  if (w_low >= width - 1) {
    w_high = w_low = width - 1;
    w = (scalar_t)w_low;
  } else {
    w_high = w_low + 1;
  }

  scalar_t lh = h - h_low;
  scalar_t lw = w - w_low;
  scalar_t hh = 1 - lh, hw = 1 - lw;

  scalar_t v1 = bottom_data[h_low * data_width + w_low];
  scalar_t v2 = bottom_data[h_low * data_width + w_high];
  scalar_t v3 = bottom_data[h_high * data_width + w_low];
  scalar_t v4 = bottom_data[h_high * data_width + w_high];
  scalar_t w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

template <typename scalar_t>
__device__ scalar_t get_gradient_weight(scalar_t argmax_h, scalar_t argmax_w,
                                     const int h, const int w, const int height,
                                     const int width) {

  if (argmax_h < 0 || argmax_h > height || argmax_w < 0 || argmax_w > width) {
    // empty
    return 0;
  }

  argmax_h = max(argmax_h, (scalar_t)0.0f);
  argmax_w = max(argmax_w, (scalar_t)0.0f);

  int argmax_h_low = (int)argmax_h;
  int argmax_w_low = (int)argmax_w;
  int argmax_h_high;
  int argmax_w_high;
  if (argmax_h_low >= height - 1) {
    argmax_h_high = argmax_h_low = height - 1;
    argmax_h = (scalar_t)argmax_h_low;
  } else {
    argmax_h_high = argmax_h_low + 1;
  }
  if (argmax_w_low >= width - 1) {
    argmax_w_high = argmax_w_low = width - 1;
    argmax_w = (scalar_t)argmax_w_low;
  } else {
    argmax_w_high = argmax_w_low + 1;
  }
  scalar_t weight = 0;
  if (h == argmax_h_low) {
    if (w == argmax_w_low) {
      weight = (h + 1 - argmax_h) * (w + 1 - argmax_w);
    } else if (w == argmax_w_high) {
      weight = (h + 1 - argmax_h) * (argmax_w + 1 - w);
    }
  } else if (h == argmax_h_high) {
    if (w == argmax_w_low) {
      weight = (argmax_h + 1 - h) * (w + 1 - argmax_w);
    } else if (w == argmax_w_high) {
      weight = (argmax_h + 1 - h) * (argmax_w + 1 - w);
    }
  }
  return weight;
}

template <typename scalar_t>
__device__ scalar_t get_coordinate_weight(scalar_t argmax_h, scalar_t argmax_w,
                                          const int height, const int width,
                                          const scalar_t *im_data,
                                          const int data_width, const int bp_dir) {

  if (argmax_h < 0 || argmax_h > height || argmax_w < 0 || argmax_w > width) {
    // empty
    return 0;
  }

  if (argmax_h < 0)
    argmax_h = 0;
  if (argmax_w < 0)
    argmax_w = 0;

  int argmax_h_low = (int)argmax_h;
  int argmax_w_low = (int)argmax_w;
  int argmax_h_high;
  int argmax_w_high;
  if (argmax_h_low >= height - 1) {
    argmax_h_high = argmax_h_low = height - 1;
    argmax_h = (scalar_t)argmax_h_low;
  } else {
    argmax_h_high = argmax_h_low + 1;
  }
  if (argmax_w_low >= width - 1) {
    argmax_w_high = argmax_w_low = width - 1;
    argmax_w = (scalar_t)argmax_w_low;
  } else {
    argmax_w_high = argmax_w_low + 1;
  }
  scalar_t weight = 0;

  if (bp_dir == 0) {
    weight += -1 * (argmax_w_low + 1 - argmax_w) *
              im_data[argmax_h_low * data_width + argmax_w_low];
    weight += -1 * (argmax_w - argmax_w_low) *
              im_data[argmax_h_low * data_width + argmax_w_high];
    weight += (argmax_w_low + 1 - argmax_w) *
              im_data[argmax_h_high * data_width + argmax_w_low];
    weight += (argmax_w - argmax_w_low) *
              im_data[argmax_h_high * data_width + argmax_w_high];
  } else if (bp_dir == 1) {
    weight += -1 * (argmax_h_low + 1 - argmax_h) *
              im_data[argmax_h_low * data_width + argmax_w_low];
    weight += (argmax_h_low + 1 - argmax_h) *
              im_data[argmax_h_low * data_width + argmax_w_high];
    weight += -1 * (argmax_h - argmax_h_low) *
              im_data[argmax_h_high * data_width + argmax_w_low];
    weight += (argmax_h - argmax_h_low) *
              im_data[argmax_h_high * data_width + argmax_w_high];
  }

  return weight;
}

template <typename scalar_t>
__global__ void deformable_im2col_gpu_kernel(
    const int n, const scalar_t *data_im, const scalar_t *data_offset,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int channel_per_deformable_group, const int height_col,
    const int width_col, scalar_t *data_col) {
  CUDA_KERNEL_LOOP(index, n) {
    // index index of output matrix
    const int w_col = index % width_col;
    const int h_col = (index / width_col) % height_col;
    const int c_im = (index / width_col) / height_col;
    const int c_col = c_im * kernel_h * kernel_w;

    // compute deformable group index
    const int deformable_group_index = c_im / channel_per_deformable_group;

    const int h_in = h_col * stride_h - pad_h;
    const int w_in = w_col * stride_w - pad_w;
    scalar_t *data_col_ptr =
        data_col + (c_col * height_col + h_col) * width_col + w_col;
    const scalar_t *data_im_ptr = data_im + (c_im * height + h_in) * width + w_in;
    const scalar_t *data_offset_ptr = data_offset + deformable_group_index * 2 *
                                                     kernel_h * kernel_w *
                                                     height_col * width_col;

    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
        const int data_offset_h_ptr =
            ((2 * (i * kernel_w + j)) * height_col + h_col) * width_col + w_col;
        const int data_offset_w_ptr =
            ((2 * (i * kernel_w + j) + 1) * height_col + h_col) * width_col +
            w_col;
        const scalar_t offset_h = data_offset_ptr[data_offset_h_ptr];
        const scalar_t offset_w = data_offset_ptr[data_offset_w_ptr];
        scalar_t val = static_cast<scalar_t>(0);
        const scalar_t h_im = h_in + i * dilation_h + offset_h;
        const scalar_t w_im = w_in + j * dilation_w + offset_w;
        if (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) {
          const scalar_t map_h = i * dilation_h + offset_h;
          const scalar_t map_w = j * dilation_w + offset_w;
          const int cur_height = height - h_in;
          const int cur_width = width - w_in;
          val = deformable_im2col_bilinear<scalar_t>(data_im_ptr, width, cur_height,
                                           cur_width, map_h, map_w);
        }
        *data_col_ptr = val;
        data_col_ptr += height_col * width_col;
      }
    }
  }
}

void deformable_im2col(Tensor data_im,
                       Tensor data_offset, const int channels,
                       const int height, const int width, const int ksize_h, const int ksize_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w, const int dilation_h, const int dilation_w,
                       const int deformable_group, Tensor data_col) {
  // We are going to launch channels * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
  int height_col =
      (height + 2 * pad_h - (dilation_h * (ksize_h - 1) + 1)) / stride_h + 1;
  int width_col =
      (width + 2 * pad_w - (dilation_w * (ksize_w - 1) + 1)) / stride_w + 1;
  int num_kernels = channels * height_col * width_col;
  int channel_per_deformable_group = channels / deformable_group;
  // Launch
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(data_im.scalar_type(), "deformable_im2col_cuda", ([&] {
    deformable_im2col_gpu_kernel<scalar_t><<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
      num_kernels, data_im.data_ptr<scalar_t>(), data_offset.data_ptr<scalar_t>(), height, width, ksize_h, ksize_w, pad_h,
      pad_w, stride_h, stride_w, dilation_h, dilation_w,
      channel_per_deformable_group, height_col, width_col, data_col.data_ptr<scalar_t>());
  }));
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in deformable_im2col: %s\n", cudaGetErrorString(err));
    // TODO(BZ) panic
  }
}

template <typename scalar_t>
__global__ void deformable_col2im_gpu_kernel(
    const int n, const scalar_t *data_col, const scalar_t *data_offset,
    const int channels, const int height, const int width, const int kernel_h,
    const int kernel_w, const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    const int channel_per_deformable_group, const int height_col,
    const int width_col, scalar_t *grad_im) {
  CUDA_KERNEL_LOOP(index, n) {
    const int j = (index / width_col / height_col) % kernel_w;
    const int i = (index / width_col / height_col / kernel_w) % kernel_h;
    const int c = index / width_col / height_col / kernel_w / kernel_h;
    // compute the start and end of the output

    const int deformable_group_index = c / channel_per_deformable_group;

    int w_out = index % width_col;
    int h_out = (index / width_col) % height_col;
    int w_in = w_out * stride_w - pad_w;
    int h_in = h_out * stride_h - pad_h;

    const scalar_t *data_offset_ptr = data_offset + deformable_group_index * 2 *
                                                     kernel_h * kernel_w *
                                                     height_col * width_col;
    const int data_offset_h_ptr =
        ((2 * (i * kernel_w + j)) * height_col + h_out) * width_col + w_out;
    const int data_offset_w_ptr =
        ((2 * (i * kernel_w + j) + 1) * height_col + h_out) * width_col + w_out;
    const scalar_t offset_h = data_offset_ptr[data_offset_h_ptr];
    const scalar_t offset_w = data_offset_ptr[data_offset_w_ptr];
    const scalar_t cur_inv_h_data = h_in + i * dilation_h + offset_h;
    const scalar_t cur_inv_w_data = w_in + j * dilation_w + offset_w;

    const scalar_t cur_top_grad = data_col[index];
    const int cur_h = (int)cur_inv_h_data;
    const int cur_w = (int)cur_inv_w_data;
    for (int dy = -2; dy <= 2; dy++) {
      for (int dx = -2; dx <= 2; dx++) {
        if (cur_h + dy >= 0 && cur_h + dy < height && cur_w + dx >= 0 &&
            cur_w + dx < width && abs(cur_inv_h_data - (cur_h + dy)) < 1 &&
            abs(cur_inv_w_data - (cur_w + dx)) < 1) {
          int cur_bottom_grad_pos =
              (c * height + cur_h + dy) * width + cur_w + dx;
          scalar_t weight =
              get_gradient_weight<scalar_t>(cur_inv_h_data, cur_inv_w_data, cur_h + dy,
                                  cur_w + dx, height, width);
          atomicAdd(grad_im + cur_bottom_grad_pos, weight * cur_top_grad);
        }
      }
    }
  }
}

void deformable_col2im(Tensor data_col,
                       Tensor data_offset, const int channels,
                       const int height, const int width, const int ksize_h,
                       const int ksize_w, const int pad_h, const int pad_w,
                       const int stride_h, const int stride_w,
                       const int dilation_h, const int dilation_w,
                       const int deformable_group, Tensor grad_im) {

  int height_col =
      (height + 2 * pad_h - (dilation_h * (ksize_h - 1) + 1)) / stride_h + 1;
  int width_col =
      (width + 2 * pad_w - (dilation_w * (ksize_w - 1) + 1)) / stride_w + 1;
  int num_kernels = channels * ksize_h * ksize_w * height_col * width_col;
  int channel_per_deformable_group = channels / deformable_group;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(data_col.scalar_type(), "deformable_col2im_cuda", ([&] {
    deformable_col2im_gpu_kernel<scalar_t><<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
      num_kernels, data_col.data_ptr<scalar_t>(), data_offset.data_ptr<scalar_t>(), channels, height, width, ksize_h,
      ksize_w, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
      channel_per_deformable_group, height_col, width_col, grad_im.data_ptr<scalar_t>());
  }));
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in deformable_col2im: %s\n", cudaGetErrorString(err));
    // TODO(BZ) panic
  }
}

template <typename scalar_t>
__global__ void deformable_col2im_coord_gpu_kernel(
    const int n, const scalar_t *data_col, const scalar_t *data_im,
    const scalar_t *data_offset, const int channels, const int height,
    const int width, const int kernel_h, const int kernel_w, const int pad_h,
    const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int channel_per_deformable_group, const int height_col,
    const int width_col, scalar_t *grad_offset) {
  CUDA_KERNEL_LOOP(index, n) {
    scalar_t val = 0;
    int w = index % width_col;
    int h = (index / width_col) % height_col;
    int c = index / width_col / height_col;
    // compute the start and end of the output

    const int deformable_group_index = c / (2 * kernel_h * kernel_w);
    const int col_step = kernel_h * kernel_w;
    int cnt = 0;
    const scalar_t *data_col_ptr = data_col + deformable_group_index *
                                               channel_per_deformable_group *
                                               width_col * height_col;
    const scalar_t *data_im_ptr =
        data_im + deformable_group_index * channel_per_deformable_group /
                      kernel_h / kernel_w * height * width;
    const scalar_t *data_offset_ptr = data_offset + deformable_group_index * 2 *
                                                     kernel_h * kernel_w *
                                                     height_col * width_col;

    const int offset_c = c - deformable_group_index * 2 * kernel_h * kernel_w;

    for (int col_c = (offset_c / 2); col_c < channel_per_deformable_group;
         col_c += col_step) {
      const int col_pos = ((col_c * height_col) + h) * width_col + w;
      const int bp_dir = offset_c % 2;

      int j = (col_pos / width_col / height_col) % kernel_w;
      int i = (col_pos / width_col / height_col / kernel_w) % kernel_h;
      int w_out = col_pos % width_col;
      int h_out = (col_pos / width_col) % height_col;
      int w_in = w_out * stride_w - pad_w;
      int h_in = h_out * stride_h - pad_h;
      const int data_offset_h_ptr =
          (((2 * (i * kernel_w + j)) * height_col + h_out) * width_col + w_out);
      const int data_offset_w_ptr =
          (((2 * (i * kernel_w + j) + 1) * height_col + h_out) * width_col +
           w_out);
      const scalar_t offset_h = data_offset_ptr[data_offset_h_ptr];
      const scalar_t offset_w = data_offset_ptr[data_offset_w_ptr];
      scalar_t inv_h = h_in + i * dilation_h + offset_h;
      scalar_t inv_w = w_in + j * dilation_w + offset_w;
      if (inv_h < 0 || inv_w < 0 || inv_h >= height || inv_w >= width) {
        inv_h = inv_w = -1;
      }
      const scalar_t weight = get_coordinate_weight<scalar_t>(
          inv_h, inv_w, height, width, data_im_ptr + cnt * height * width,
          width, bp_dir);
      val += weight * data_col_ptr[col_pos];
      cnt += 1;
    }

    grad_offset[index] = val;
  }
}

void deformable_col2im_coord(Tensor data_col,
                             Tensor data_im, Tensor data_offset,
                             const int channels, const int height,
                             const int width, const int ksize_h,
                             const int ksize_w, const int pad_h,
                             const int pad_w, const int stride_h,
                             const int stride_w, const int dilation_h,
                             const int dilation_w, const int deformable_group,
                             Tensor grad_offset) {

  int height_col =
      (height + 2 * pad_h - (dilation_h * (ksize_h - 1) + 1)) / stride_h + 1;
  int width_col =
      (width + 2 * pad_w - (dilation_w * (ksize_w - 1) + 1)) / stride_w + 1;
  int num_kernels =
      height_col * width_col * 2 * ksize_h * ksize_w * deformable_group;
  int channel_per_deformable_group =
      channels * ksize_h * ksize_w / deformable_group;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(data_col.scalar_type(), "deformable_col2im_cuda", ([&] {
    deformable_col2im_coord_gpu_kernel<scalar_t><<<GET_BLOCKS(num_kernels),CUDA_NUM_THREADS>>>(
      num_kernels, data_col.data_ptr<scalar_t>(), data_im.data_ptr<scalar_t>(), data_offset.data_ptr<scalar_t>(), channels, height, width,
      ksize_h, ksize_w, pad_h, pad_w, stride_h, stride_w, dilation_h,
      dilation_w, channel_per_deformable_group, height_col, width_col,
      grad_offset.data_ptr<scalar_t>());
  }));
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in deformable_col2im: %s\n", cudaGetErrorString(err));
    // TODO(BZ) panic
  }
}

