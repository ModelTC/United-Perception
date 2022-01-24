#include "focal_loss/focal_loss.h"

#include <cfloat>

using at::Tensor;
using at::Half;

#define DIVUP(m, n) ((m) / (m) + ((m) % (n) > 0))

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)
template <typename scalar_t>
__global__ void SigmoidFocalLossKernel(
    const int N, const scalar_t* logits,
    const int* targets, const scalar_t weight_pos,
    const scalar_t gamma, const scalar_t alpha,
    const int num_classes, scalar_t* losses) {
  CUDA_1D_KERNEL_LOOP(i, N) {
      int d = i % num_classes;   //current class
      int tmp = i / num_classes; //targets index
      int t = targets[tmp];

    // check whether the class is true class or not.
    // The target classes are in range 1 - 81 and the d is in range 0-80
    // because we predict A*80 dim, so for comparison purpose, compare t and (d+1)
    scalar_t c1 = (t == (d + 1));
    scalar_t c2 = (t != -1 & t != (d + 1));

    scalar_t Np = max(weight_pos, 1.0);
    scalar_t zn = (1.0 - alpha) / Np;
    scalar_t zp = alpha / Np;

    // p = 1. / 1. + expf(-x)
    scalar_t p = 1. / (1. + expf(-logits[i]));

    // (1 - p)**gamma * log(p) where
    scalar_t term1 = powf((1. - p), gamma) * logf(max(p, FLT_MIN));
    // p**gamma * log(1 - p)
    scalar_t term2 =
        powf(p, gamma) *
        (-1. * logits[i] * (logits[i] >= 0) -
         logf(1. + expf(logits[i] - 2. * logits[i] * (logits[i] >= 0))));

    losses[i] = 0.0;
    losses[i] += -c1 * term1 * zp;
    losses[i] += -c2 * term2 * zn;
  }
}

template <typename scalar_t>
__global__ void SigmoidFocalLossGradientKernel(
    const int N, const scalar_t* logits,
    const int* targets, scalar_t* dX_data, const scalar_t weight_pos,
    const scalar_t gamma, const scalar_t alpha, const int num_classes) {
  CUDA_1D_KERNEL_LOOP(i, N) {
      int d = i % num_classes;   //current class
      int tmp = i / num_classes; //targets index
      int t = targets[tmp];

      scalar_t Np = max(weight_pos, 1.0);
      scalar_t zn = (1.0 - alpha) / Np;
      scalar_t zp = alpha / Np;
      //int t = targets[n * (H * W * A) + a * (H * W) + y * W + x];

      scalar_t c1 = (t == (d + 1));
      scalar_t c2 = (t != -1 & t != (d + 1));
      scalar_t p = 1. / (1. + expf(-logits[i]));

      // (1-p)**g * (1 - p - g*p*log(p))
      scalar_t term1 =
          powf((1. - p), gamma) *
          (1. - p - (p * gamma * logf(max(p, FLT_MIN))));
      // (p**g) * (g*(1-p)*log(1-p) - p)
      scalar_t term2 =
          powf(p, gamma) *
          ((-1. * logits[i] * (logits[i] >= 0) -
           logf(1. + expf(logits[i] - 2. * logits[i] * (logits[i] >= 0)))) *
           (1. - p) * gamma - p);
      dX_data[i] = 0.0;
      dX_data[i] += -c1 * zp * term1;
      dX_data[i] += -c2 * zn * term2;
  }
}

int SigmoidFocalLossForwardLaucher(
    const int N, Tensor logits,
    Tensor targets, const float weight_pos,
    const float gamma, const float alpha,
    const int num_classes, Tensor losses){

    const int kThreadsPerBlock = 1024;
    int output_size = N;
    cudaError_t err;

    err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf( stderr, "%s#%d: cudaCheckError() failed : %s\n", __FILE__,
                __LINE__, cudaGetErrorString( err ) );
        exit( -1 );
    }
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(logits.scalar_type(), "sigmoid_focal_loss_forward", ([&] {
      SigmoidFocalLossKernel<scalar_t><<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock>>>(
        N, logits.data_ptr<scalar_t>(), 
        targets.data_ptr<int>(), 
        weight_pos, gamma, alpha, num_classes, 
        losses.data_ptr<scalar_t>());
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


int SigmoidFocalLossBackwardLaucher(
    const int N, Tensor logits, Tensor targets,
    Tensor dX_data, const float weight_pos,
    const float gamma, const float alpha, const int num_classes){

    const int kThreadsPerBlock = 1024;
    int output_size = N;
    cudaError_t err;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(logits.scalar_type(), "sigmoid_focal_loss_backward", ([&] {
      SigmoidFocalLossGradientKernel<scalar_t><<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock>>>(
          N, logits.data_ptr<scalar_t>(), 
          targets.data_ptr<int>(), 
          dX_data.data_ptr<scalar_t>(), 
          weight_pos, gamma, alpha, num_classes);
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


