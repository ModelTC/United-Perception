#include "psroi_pooling/psroi_pooling.h"

using at::Tensor;

int psroi_pooling_forward(int pooled_height,
                          int pooled_width,
                          int output_dim,
                          float spatial_scale,
                          Tensor features,
                          Tensor rois,
                          Tensor output,
                          Tensor mapping_channel) {
    // ONNX requires cpu forward support
    return 0;
}
