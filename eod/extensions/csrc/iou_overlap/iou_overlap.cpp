#include "iou_overlap/iou_overlap.h"

using at::Tensor;

void gpu_iou_overlaps(Tensor bboxes1, Tensor bboxes2, Tensor output, const int mode){
    // Grad the input tensor
    CHECK_INPUT(bboxes1);
    CHECK_INPUT(bboxes2);
    CHECK_INPUT(output);

    // Number of boxes
    int num_bbox1 = bboxes1.size(0);
    int num_bbox2 = bboxes2.size(0);
    int size_bbox1 = bboxes1.size(1);
    int size_bbox2 = bboxes2.size(1);

    AT_ASSERTM(output.is_cuda(), "output must be cuda tensor");

    AT_ASSERTM(size_bbox1 == size_bbox2, "bbox1 dim must match bbox2");
    
    IOUOverlap(bboxes1,
               bboxes2,
               size_bbox1,
               num_bbox1,
               num_bbox2,
               output,
               mode);
}

