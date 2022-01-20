#include "roi_align/roi_align.h"

using at::Tensor;

int roi_align_forward(bool aligned, int aligned_height, int aligned_width,
        float spatial_scale, int sampling_ratio,
        Tensor features, Tensor rois, Tensor output) {
    // ONNX requires cpu forward support
    return 0;
}
