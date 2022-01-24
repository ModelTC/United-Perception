#include "cross_focal_loss/cross_focal_loss.h"

using at::Tensor;


int cross_focal_loss_sigmoid_forward_cuda(
    int N,
    Tensor logits,
    Tensor targets,
    float weight_pos,
    float gamma,
    float alpha,
    int num_classes,
    Tensor losses,
    Tensor neg_map)
{
    // Grab the input tensor
    CHECK_INPUT(logits);
    CHECK_INPUT(targets);
    CHECK_INPUT(losses);
    CHECK_INPUT(neg_map);

    CrossSigmoidFocalLossForwardLauncher(
        N, logits, targets, weight_pos,
        gamma, alpha, num_classes, losses, neg_map);

    return 1;
}

int cross_focal_loss_sigmoid_backward_cuda(
    int N,
    Tensor logits,
    Tensor targets,
    Tensor dX_data,
    float weight_pos,
    float gamma,
    float alpha,
    int num_classes,
    Tensor neg_map)
{
    // Grab the input tensor
    CHECK_INPUT(logits);
    CHECK_INPUT(targets);
    CHECK_INPUT(dX_data);
    CHECK_INPUT(neg_map);

    CrossSigmoidFocalLossBackwardLauncher(
        N, logits, targets, dX_data,
        weight_pos, gamma, alpha, num_classes, neg_map);

    return 1;
}
