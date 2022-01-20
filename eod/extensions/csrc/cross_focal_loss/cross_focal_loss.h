#ifndef CROSS_FOCAL_LOSS_H_
#define CROSS_FOCAL_LOSS_H_

#include <ATen/ATen.h>
#include <cmath>
#include <cassert>
#include <cstdio>
using at::Tensor;

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

int cross_focal_loss_sigmoid_forward_cuda(
    int N,
    Tensor logits,
    Tensor targets,
    float weight_pos,
    float gamma,
    float alpha,
    int num_classes,
    Tensor losses,
    Tensor neg_map);

int cross_focal_loss_sigmoid_backward_cuda(
    int N,
    Tensor logits,
    Tensor targets,
    Tensor dX_data,
    float weight_pos,
    float gamma,
    float alpha,
    int num_classes,
    Tensor neg_map);

int CrossSigmoidFocalLossForwardLauncher(
    const int N, Tensor logits,
    Tensor targets, const float weight_pos,
    const float gamma, const float alpha,
    const int num_classes, Tensor losses, Tensor neg_map);

int CrossSigmoidFocalLossBackwardLauncher(
    const int N, Tensor logits,
    Tensor targets, Tensor dX_data, const float weight_pos,
    const float gamma, const float alpha, const int num_classes,
    Tensor neg_map);

#endif
