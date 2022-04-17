#include "softer_nms/softer_nms.h"

using at::Tensor;
/*
 *  * gpu kernel not implemented
 *  void _softer_nms(int boxes_num, Tensor boxes_dev, IOUMethod iou_method, Method method, float sigma, float iou_thresh,
 *   float iou_sigma);
 *   */

inline static bool score_cmp(const Proposal& a, const Proposal& b) {
  return a.score < b.score;
}

static float calculate_iou(const Proposal& a, const Proposal& b) {
  float iou = 0.;
  float iw = fminf(a.x2, b.x2) - fmaxf(a.x1, b.x1) + 1;
  if (iw > 0.) {
    float ih = fminf(a.y2, b.y2) - fmaxf(a.y1, b.y1) + 1;
    if (ih > 0) {
      float area_sum = (b.x2 - b.x1 + 1) * (b.y2 - b.y1 + 1) +
                       (a.x2 - a.x1 + 1) * (a.y2 - a.y1 + 1);
      float area_in = iw * ih;
      iou = area_in / (area_sum - area_in);
    }
  }
  return iou;
}

int cpu_softer_nms(Tensor boxes, Tensor inds, float sigma, float iou_thresh,
                   IOUMethod iou_method, float iou_sigma, Method method) {
  // boxes should be [N, 9] tensor, 4 coor + 4 log variance + 1 score
  long boxes_num = boxes.size(0);
  long i, j;
  float *boxes_flat = boxes.data_ptr<float>();
  long *inds_flat = inds.data_ptr<long>();
  auto boxes_p = reinterpret_cast<Proposal*>(boxes_flat);
  for (i = 0; i < boxes_num; ++i) {
    auto mx = std::max_element(boxes_p + i, boxes_p + boxes_num, score_cmp);
    std::swap(boxes_p[i], *mx);
    std::swap(inds_flat[i], inds_flat[mx - boxes_p]);
    double nx1 = 1. / exp(boxes_p[i].vx1);
    double ny1 = 1. / exp(boxes_p[i].vy1);
    double nx2 = 1. / exp(boxes_p[i].vx2);
    double ny2 = 1. / exp(boxes_p[i].vy2);
    double dx1 = boxes_p[i].x1 * nx1;
    double dy1 = boxes_p[i].y1 * ny1;
    double dx2 = boxes_p[i].x2 * nx2;
    double dy2 = boxes_p[i].y2 * ny2;
    for (j = i + 1; j < boxes_num; ++j) {
      float iou = calculate_iou(boxes_p[i], boxes_p[j]);
      float ws = 1.;
      switch (iou_method) {
        case IOUMethod::LINEAR:
          ws = iou > iou_thresh ? 1. - iou : 1.0f;
          break;
        case IOUMethod::GAUSSIAN:
          ws = exp(-iou * iou / sigma);
          break;
        case IOUMethod::HARD:
          ws = iou > iou_thresh ? 0. : 1.;
          break;
      }
      // Update score (soft-nms)
      boxes_p[j].score *= ws;
      if (iou <= 0) { 
	    continue;
      }
      float p = 1.;
      if (method == Method::VAR_VOTING) {
        p = exp(-(1 - iou) * (1 - iou) / iou_sigma);
      }
      else if (iou < iou_thresh) {
        continue;
      }
      float wx1, wx2, wy1, wy2;
      // weights of coors
      wx1 = wx2 = wy1 = wy2 = p;

      wx1 /= exp(boxes_p[j].vx1);
      wx2 /= exp(boxes_p[j].vx2);
      wy1 /= exp(boxes_p[j].vy1);
      wy2 /= exp(boxes_p[j].vy2);

      dx1 += boxes_p[j].x1 * wx1;
      dx2 += boxes_p[j].x2 * wx2;
      dy1 += boxes_p[j].y1 * wy1;
      dy2 += boxes_p[j].y2 * wy2;

      nx1 += wx1;
      nx2 += wx2;
      ny1 += wy1;
      ny2 += wy2;
    }

    dx1 /= nx1;
    dx2 /= nx2;
    dy1 /= ny1;
    dy2 /= ny2;

    boxes_p[i].x1 = (float)dx1;
    boxes_p[i].x2 = (float)dx2;
    boxes_p[i].y1 = (float)dy1;
    boxes_p[i].y2 = (float)dy2;
  }
  return 1;
}

