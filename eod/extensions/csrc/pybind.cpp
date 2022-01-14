#include "roi_align/roi_align.h"
#include "psroi_align/psroi_align.h"
#include "psroi_pooling/psroi_pooling.h"
#include "nms/nms.h"
#include <torch/torch.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // pybind ROIAlignPooling
    pybind11::module ra = m.def_submodule("roi_align",
                                          "roi alignment pooling");
    ra.def("forward_avg_cuda", &roi_align_avg_forward_cuda, "roi_align avg forward (CUDA)");
    ra.def("backward_avg_cuda", &roi_align_avg_backward_cuda, "roi_align avg backward (CUDA)");
    // ra.def("forward_max_cuda", &roi_align_max_forward_cuda, "roi_align max forward (CUDA)");
    // ra.def("backward_max_cuda", &roi_align_max_backward_cuda, "roi_align max backward (CUDA)");
    ra.def("forward_cpu", &roi_align_forward, "roi_align forward (CPU)");

    // pybind PSROIAlign
    pybind11::module pa = m.def_submodule("psroi_align",
                                          "position sensetive roi align");
    pa.def("forward_cuda", &psroi_align_forward_cuda, "psroi_align forward (CUDA)");
    pa.def("backward_cuda", &psroi_align_backward_cuda, "psroi_align backward (CUDA)");
    // pa.def("forward_cpu", &psroi_align_forward, "psroi_align forward (CPU)");
    
    // pybind PSROIPooling
    pybind11::module pp = m.def_submodule("psroi_pooling",
                                        "position sensetive roi pooling");
    pp.def("forward_cuda", &psroi_pooling_forward_cuda, "psroi_pooling forward (CUDA)");
    pp.def("backward_cuda", &psroi_pooling_backward_cuda, "psroi_pooling backward (CUDA)");
    pp.def("forward_cpu", &psroi_pooling_forward, "psroi_pooling forward (CPU)");

    // pybind vanilla nms
    pybind11::module naive_nms = m.def_submodule("naive_nms",
                                                 "vanilla nms method");
    naive_nms.def("gpu_nms", &gpu_nms, "gpu_nms (CUDA)");
    naive_nms.def("cpu_nms", &cpu_nms, "cpu_nms (CPU)");
}
