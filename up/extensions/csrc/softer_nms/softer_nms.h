#ifndef SOFTER_NMS_H_
#define SOFTER_NMS_H_

#include <ATen/ATen.h>
#include <cmath>
#include <cstdio>
#include <cfloat>

struct Proposal {
    /*
 *      * Corresponding to 9-channel tensors
 *           * (x1, y1), (x2, y2): Coordinates of the top-left corner and the bottom-right corner
 *                * (vx1, vy1, vx2, vy2): Predicted log std variance of each coordinate
 *                     * score: Classification score of the proposal
 *                          */
    float x1, y1, x2, y2, vx1, vy1, vx2, vy2, score;
};

struct Boxes {
    /* Coordinates of the top-left corner and the bottom-right corner */
    float x1, y1, x2, y2;
};

enum class IOUMethod : uint32_t
{
    /* The methods used in soft-nms*/
    LINEAR = 0,
    GAUSSIAN,
    HARD
};

enum class Method : uint32_t
{
    /* The methods used in softer-nms*/
    VAR_VOTING = 0, //newest version: variance voting
    SOFTER          //deprecated version
};

int cpu_softer_nms(at::Tensor boxes, at::Tensor inds, float sigma, float iou_thresh,
                   IOUMethod iou_method, float iou_sigma, Method method);

#endif

