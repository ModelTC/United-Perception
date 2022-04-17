#include "deform_conv/deformable_conv.h"
#include "roi_align/roi_align.h"
#include "psroi_align/psroi_align.h"
#include "psroi_pooling/psroi_pooling.h"
#include "nms/nms.h"
#include "softer_nms/softer_nms.h"
#include "focal_loss/focal_loss.h"
#include "cross_focal_loss/cross_focal_loss.h"
#include "iou_overlap/iou_overlap.h"
#include "roiaware_pool3d/roiaware_pool3d.h"
#include "roipoint_pool3d/roipoint_pool3d.h"
#include "iou3d_nms/iou3d_cpu.h"
#include "iou3d_nms/iou3d_nms.h"
#include <torch/torch.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // pybind deform conv v1
    pybind11::module dc_v1 = m.def_submodule("deform_conv_v1",
                                           "deformable convolution v1");
    dc_v1.def("backward_parameters_cuda",
              &deform_conv_backward_parameters_cuda,
              "deform_conv_backward_parameters_cuda (CUDA)");
    dc_v1.def("backward_input_cuda",
              &deform_conv_backward_input_cuda,
              "deform_conv_backward_input_cuda (CUDA)");
    dc_v1.def("forward_cuda",
              &deform_conv_forward_cuda,
              "deform_conv_forward_cuda (CUDA)");
    dc_v1.def("forward_cpu",
              &deform_conv_forward,
              "deform_conv_forward_cpu (CPU)");
    
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

    // pybind focal loss
    pybind11::module fl = m.def_submodule("focal_loss",
                                          "focal loss for RetinaNet");
    fl.def("sigmoid_forward_cuda",
           &focal_loss_sigmoid_forward_cuda,
           "sigmoid_forward_cuda forward (CUDA)");
    fl.def("sigmoid_backward_cuda",
           &focal_loss_sigmoid_backward_cuda,
           "sigmoid_backward_cuda backward (CUDA)");
    fl.def("softmax_forward_cuda",
           &focal_loss_softmax_forward_cuda,
           "softmax_forward_cuda forward (CUDA)");
    fl.def("softmax_backward_cuda",
           &focal_loss_softmax_backward_cuda,
           "softmax_backward_cuda backward (CUDA)");

    // pybind cross focal loss
    pybind11::module cross_fl = m.def_submodule("cross_focal_loss", "cross focal loss for RetinaNet");
    cross_fl.def("cross_sigmoid_forward_cuda",
                 &cross_focal_loss_sigmoid_forward_cuda,
                 "sigmoid_forward_cuda forward (CUDA)");
    cross_fl.def("cross_sigmoid_backward_cuda",
                 &cross_focal_loss_sigmoid_backward_cuda,
                 "sigmoid_backward_cuda backward (CUDA)");

    // pybind IOUOverlap
    pybind11::module iou = m.def_submodule("overlaps",
                                           "calculate iou between bboxes & gts");
    iou.def("iou", &gpu_iou_overlaps, "bbox iou overlaps with gt (CUDA)");
 
    // pybind softer_nms (variance voting)
    pybind11::module softer_nms =
        m.def_submodule("softer_nms", "softer nms, variance voting");
    py::enum_<IOUMethod>(softer_nms, "IOUMethod")
        .value("LINEAR", IOUMethod::LINEAR)
        .value("GAUSSIAN", IOUMethod::GAUSSIAN)
        .value("HARD", IOUMethod::HARD)
        .export_values();
    py::enum_<Method>(softer_nms, "NMSMethod")
        .value("VAR_VOTING", Method::VAR_VOTING)
        .value("SOFTER", Method::SOFTER)
        .export_values();
    softer_nms.def("cpu_softer_nms", &cpu_softer_nms, "softer_nms(CPU)");

    // pybind roiaware_pool3d
    pybind11::module rw = m.def_submodule("roiaware_pool3d",
                                          "roiaware pool3d");
    rw.def("forward", &roiaware_pool3d_gpu, "roiaware pool3d forward (CUDA)");
    rw.def("backward", &roiaware_pool3d_gpu_backward, "roiaware pool3d backward (CUDA)");
    rw.def("points_in_boxes_gpu", &points_in_boxes_gpu, "points_in_boxes_gpu forward (CUDA)");
    rw.def("points_in_boxes_cpu", &points_in_boxes_cpu, "points_in_boxes_cpu forward (CUDA)");

    // pybind roipoint_pool3d
    pybind11::module rp = m.def_submodule("roipoint_pool3d",
                                          "roipool3d");
    rp.def("forward", &roipool3d_gpu, "roipool3d forward (CUDA)");

    // pybind  iou3d_nms
    pybind11::module iou3d_nms = m.def_submodule("iou3d_nms",
                                                 "nms of 3d boxes");
	iou3d_nms.def("boxes_overlap_bev_gpu", &boxes_overlap_bev_gpu, "oriented boxes overlap");
	iou3d_nms.def("boxes_iou_bev_gpu", &boxes_iou_bev_gpu, "oriented boxes iou");
	iou3d_nms.def("nms_gpu", &nms_gpu, "oriented nms gpu");
	iou3d_nms.def("nms_normal_gpu", &nms_normal_gpu, "nms gpu");
	iou3d_nms.def("boxes_iou_bev_cpu", &boxes_iou_bev_cpu, "oriented boxes iou");

}
