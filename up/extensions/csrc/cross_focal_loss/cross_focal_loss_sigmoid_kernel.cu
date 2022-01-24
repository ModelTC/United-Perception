#include "cross_focal_loss/cross_focal_loss.h"

#include <cfloat>

using at::Tensor;
using at::Half;

#define DIVUP(m, n) ((m) / (m) + ((m) % (n) > 0))

#define CUDA_1D_KERNEL_LOOP(i, n)                                              \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;                   \
       i += blockDim.x * gridDim.x)

template <typename scalar_t>
__global__ void CrossSigmoidFocalLossKernel(
    const int N, const scalar_t *logits, const int *targets,
    const scalar_t weight_pos, const scalar_t gamma, const scalar_t alpha,
    const int num_classes, scalar_t *losses, const int *neg_map) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    int d = i % num_classes;   // current class
    int tmp = i / num_classes; // targets index
    int t = targets[tmp];

    float c1 = (t == (d + 1));
    bool c2 = (t != -1 & t != (d + 1));
    int curr_pos = 0;
    int record_num = neg_map[curr_pos++];

    // cross training
    // base the pos_to_neg label, we get c2
    // int flag = 0;
    // for (int r = 0; r < record_num; ++r) {
    //   int record_length = neg_map[curr_pos++];
    //   if (d == neg_map[curr_pos++] - 1)
    //     for (int _r = 0; _r < record_length - 1; ++_r) {
    //       if (t == neg_map[curr_pos++])
    //         flag++;
    //     }
    //   else
    //     curr_pos += record_length - 1;
    // }
    // c2 = c2 & (flag > 0);

    // base on class we need to avoid
    int flag = 1;
    for (int r = 0; r < record_num; ++r) {
      int record_length = neg_map[curr_pos++];
      if (d == neg_map[curr_pos++] - 1)
        for (int _r = 0; _r < record_length - 1; ++_r) {
          if (t == neg_map[curr_pos++])
            flag = 0;
        }
      else
        curr_pos += record_length - 1;
    }
    c2 = c2 & (flag > 0);


    float Np = max(weight_pos, 1.0);
    float zn = (1.0 - alpha) / Np;
    float zp = alpha / Np;

    // p = 1. / 1. + expf(-x)
    float p = 1. / (1. + expf(-logits[i]));

    // (1 - p)**gamma * log(p) where
    float term1 = powf((1. - p), gamma) * logf(max(p, FLT_MIN));
    // p**gamma * log(1 - p)
    float term2 =
        powf(p, gamma) *
        (-1. * logits[i] * (logits[i] >= 0) -
         logf(1. + expf(logits[i] - 2. * logits[i] * (logits[i] >= 0))));

    losses[i] = 0.0;
    losses[i] += -c1 * term1 * zp;
    losses[i] += -c2 * term2 * zn;
  }
}

template <typename scalar_t>
__global__ void CrossSigmoidFocalLossGradientKernel(
    const int N, const scalar_t *logits, const int *targets, scalar_t *dX_data,
    const scalar_t weight_pos, const scalar_t gamma, const scalar_t alpha,
    const int num_classes, const int *neg_map) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    int d = i % num_classes;   // current class
    int tmp = i / num_classes; // targets index
    int t = targets[tmp];

    float Np = max(weight_pos, 1.0);
    float zn = (1.0 - alpha) / Np;
    float zp = alpha / Np;
    // int t = targets[n * (H * W * A) + a * (H * W) + y * W + x];

    float c1 = (t == (d + 1));
    bool c2 = (t != -1 & t != (d + 1));
    int curr_pos = 0;
    int record_num = neg_map[curr_pos++];

    // cross training
    // base the pos_to_neg label, we get c2
    // int flag = 0;
    // for (int r = 0; r < record_num; ++r) {
    //   int record_length = neg_map[curr_pos++];
    //   if (d == neg_map[curr_pos++] - 1)
    //     for (int _r = 0; _r < record_length - 1; ++_r) {
    //       if (t == neg_map[curr_pos++])
    //         flag++;
    //     }
    //   else
    //     curr_pos += record_length - 1;
    // }
    // c2 = c2 & (flag > 0);

    // base on class we need to avoid
    int flag = 1;
    for (int r = 0; r < record_num; ++r) {
      int record_length = neg_map[curr_pos++];
      if (d == neg_map[curr_pos++] - 1)
        for (int _r = 0; _r < record_length - 1; ++_r) {
          if (t == neg_map[curr_pos++])
            flag = 0;
        }
      else
        curr_pos += record_length - 1;
    }
    c2 = c2 & (flag > 0);

    float p = 1. / (1. + expf(-logits[i]));

    // (1-p)**g * (1 - p - g*p*log(p))
    float term1 =
        powf((1. - p), gamma) * (1. - p - (p * gamma * logf(max(p, FLT_MIN))));
    // (p**g) * (g*(1-p)*log(1-p) - p)
    float term2 =
        powf(p, gamma) *
        ((-1. * logits[i] * (logits[i] >= 0) -
          logf(1. + expf(logits[i] - 2. * logits[i] * (logits[i] >= 0)))) *
             (1. - p) * gamma -
         p);

    dX_data[i] = 0.0;
    dX_data[i] += -c1 * zp * term1;
    dX_data[i] += -c2 * zn * term2;
  }
}

int CrossSigmoidFocalLossForwardLauncher(const int N, Tensor logits,
                                         Tensor targets, const float weight_pos,
                                         const float gamma, const float alpha,
                                         const int num_classes, Tensor losses,
                                         Tensor neg_map) {

  const int kThreadsPerBlock = 1024;
  int output_size = N;
  cudaError_t err;

  err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "%s#%d: cudaCheckError() failed : %s\n", __FILE__, __LINE__,
            cudaGetErrorString(err));
    exit(-1);
  }
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      logits.scalar_type(), "cross_sigmoid_focal_loss_forward", ([&] {
        CrossSigmoidFocalLossKernel<scalar_t>
            <<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock,
               kThreadsPerBlock>>>(N, logits.data_ptr<scalar_t>(),
                                   targets.data_ptr<int>(), weight_pos, gamma,
                                   alpha, num_classes, losses.data_ptr<scalar_t>(),
                                   neg_map.data_ptr<int>());
      }));
  err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "%s#%d: cudaCheckError() failed : %s\n", __FILE__, __LINE__,
            cudaGetErrorString(err));
    exit(-1);
  }

  return 1;
}

int CrossSigmoidFocalLossBackwardLauncher(const int N, Tensor logits,
                                          Tensor targets, Tensor dX_data,
                                          const float weight_pos,
                                          const float gamma, const float alpha,
                                          const int num_classes,
                                          Tensor neg_map) {

  const int kThreadsPerBlock = 1024;
  int output_size = N;
  cudaError_t err;
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      logits.scalar_type(), "cross_sigmoid_focal_loss_backward", ([&] {
        CrossSigmoidFocalLossGradientKernel<scalar_t>
            <<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock,
               kThreadsPerBlock>>>(N, logits.data_ptr<scalar_t>(),
                                   targets.data_ptr<int>(),
                                   dX_data.data_ptr<scalar_t>(), weight_pos, gamma,
                                   alpha, num_classes, neg_map.data_ptr<int>());
      }));
  err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "%s#%d: cudaCheckError() failed : %s\n", __FILE__, __LINE__,
            cudaGetErrorString(err));
    exit(-1);
  }

  return 1;
}
