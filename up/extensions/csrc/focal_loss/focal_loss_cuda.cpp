#include "focal_loss/focal_loss.h"

using at::Tensor;

int focal_loss_sigmoid_forward_cuda(
                           int N,
                           Tensor logits,
                           Tensor targets,
                           float weight_pos,
                           float gamma, 
                           float alpha,
                           int num_classes,
                           Tensor losses){
    // Grab the input tensor
    CHECK_INPUT(logits);
    CHECK_INPUT(targets);
    CHECK_INPUT(losses);

    SigmoidFocalLossForwardLaucher(
        N, logits, targets, weight_pos, 
        gamma, alpha, num_classes, losses);

    return 1;
}

int focal_loss_sigmoid_backward_cuda(
                           int N,
                           Tensor  logits,
                           Tensor targets,
                           Tensor  dX_data,
                           float weight_pos,
                           float gamma,
                           float alpha,
                           int num_classes){
    // Grab the input tensor
    CHECK_INPUT(logits);
    CHECK_INPUT(targets);
    CHECK_INPUT(dX_data);

    SigmoidFocalLossBackwardLaucher(
        N, logits, targets, dX_data,
        weight_pos, gamma, alpha, num_classes);

    return 1;
}

int focal_loss_softmax_forward_cuda(
                           int N,
                           Tensor logits,
                           Tensor targets,
                           float weight_pos,
                           float gamma, 
                           float alpha,
                           int num_classes,
                           Tensor losses,
                           Tensor priors){
    // Grab the input tensor
    CHECK_INPUT(logits);
    CHECK_INPUT(targets);
    CHECK_INPUT(losses);
    CHECK_INPUT(priors);

    SoftmaxFocalLossForwardLaucher(
        N, logits, targets, weight_pos, 
        gamma, alpha, num_classes, losses, priors);

    return 1;
}

int focal_loss_softmax_backward_cuda(
                           int N,
                           Tensor logits,
                           Tensor targets,
                           Tensor dX_data,
                           float weight_pos,
                           float gamma,
                           float alpha,
                           int num_classes,
                           Tensor priors,
                           Tensor buff){
    // Grab the input tensor
    CHECK_INPUT(logits);
    CHECK_INPUT(targets);
    CHECK_INPUT(dX_data);
    CHECK_INPUT(priors);
    CHECK_INPUT(buff);
    
    SoftmaxFocalLossBackwardLaucher(
        N, logits, targets, dX_data,
        weight_pos, gamma, alpha, num_classes, priors, buff);

    return 1;
}

