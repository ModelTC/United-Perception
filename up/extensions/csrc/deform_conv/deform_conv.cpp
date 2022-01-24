#include "deform_conv/deformable_conv.h"

#include <cstdio>
using at::Tensor;

int deform_conv_forward(Tensor input, Tensor weight,
                        Tensor offset, Tensor output,
                        Tensor columns, Tensor ones, int kH,
                        int kW, int dH, int dW, int padH, int padW,
                        int dilationH, int dilationW, int groups,
                        int deformable_group) {
    // ONNX requires operations support cpu forward
    return 0;
}
