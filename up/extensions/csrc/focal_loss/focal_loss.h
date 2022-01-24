#ifndef FOCAL_LOSS_H_
#define FOCAL_LOSS_H_

#include <ATen/ATen.h>
#include <cmath>
#include <cassert>
#include <cstdio>
using at::Tensor;

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

int focal_loss_sigmoid_forward_cuda(
                           int N,
                           Tensor logits,
                           Tensor targets,
                           float weight_pos,
                           float gamma, 
                           float alpha,
                           int num_classes,
                           Tensor losses);

int focal_loss_sigmoid_backward_cuda(
                           int N,
                           Tensor  logits,
                           Tensor targets,
                           Tensor  dX_data,
                           float weight_pos,
                           float gamma,
                           float alpha,
                           int num_classes);

int focal_loss_softmax_forward_cuda(
                           int N,
                           Tensor logits,
                           Tensor targets,
                           float weight_pos,
                           float gamma, 
                           float alpha,
                           int num_classes,
                           Tensor losses,
                           Tensor priors);

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
                           Tensor buff);

int SoftmaxFocalLossForwardLaucher(
    const int N, Tensor logits,
    Tensor targets, const float weight_pos,
    const float gamma, const float alpha,
    const int num_classes, Tensor losses,
    Tensor priors);

int SoftmaxFocalLossBackwardLaucher(
    const int N, Tensor logits, Tensor targets,
    Tensor dX_data, const float weight_pos,
    const float gamma, const float alpha, const int num_classes, 
    Tensor priors, Tensor buff);

int SigmoidFocalLossForwardLaucher(
    const int N, Tensor logits,
    Tensor targets, const float weight_pos,
    const float gamma, const float alpha,
    const int num_classes, Tensor losses);

int SigmoidFocalLossBackwardLaucher(
    const int N, Tensor logits, Tensor targets,
    Tensor dX_data, const float weight_pos,
    const float gamma, const float alpha, const int num_classes);

#endif
