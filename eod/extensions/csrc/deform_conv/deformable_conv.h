#ifndef DEFORMABLE_CONV_H_
#define DEFORMABLE_CONV_H_

#include <ATen/ATen.h>
#include <cstdio>

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


// for tocaffe
int deform_conv_forward(at::Tensor input, at::Tensor weight,
                        at::Tensor offset, at::Tensor output,
                        at::Tensor columns, at::Tensor ones, int kH,
                        int kW, int dH, int dW, int padH, int padW,
                        int dilationH, int dilationW, int groups,
                        int deformable_group);

int deform_conv_forward_cuda(at::Tensor input, at::Tensor weight,
                             at::Tensor offset, at::Tensor output,
                             at::Tensor columns, at::Tensor ones, int kH,
                             int kW, int dH, int dW, int padH, int padW,
                             int dilationH, int dilationW, int groups,
                             int deformable_group);

int deform_conv_backward_input_cuda(
    at::Tensor input, at::Tensor offset, at::Tensor gradOutput,
    at::Tensor gradInput, at::Tensor gradOffset, at::Tensor weight,
    at::Tensor columns, int kH, int kW, int dH, int dW, int padH, int padW,
    int dilationH, int dilationW, int groups, int deformable_group) ;

int deform_conv_backward_parameters_cuda(
    at::Tensor input, at::Tensor offset, at::Tensor gradOutput,
    at::Tensor gradWeight, /*Tensor gradBias, */
    at::Tensor columns, at::Tensor ones, int kH, int kW, int dH, int dW,
    int padH, int padW, int dilationH, int dilationW, int groups, int deformable_group,
    float scale);

void deformable_col2im(at::Tensor data_col,
                       at::Tensor data_offset, const int channels,
                       const int height, const int width, const int ksize_h,
                       const int ksize_w, const int pad_h, const int pad_w,
                       const int stride_h, const int stride_w,
                       const int dilation_h, const int dilation_w,
                       const int deformable_group, at::Tensor grad_im);

void deformable_col2im_coord(at::Tensor data_col,
                             at::Tensor data_im, at::Tensor data_offset,
                             const int channels, const int height,
                             const int width, const int ksize_h,
                             const int ksize_w, const int pad_h,
                             const int pad_w, const int stride_h,
                             const int stride_w, const int dilation_h,
                             const int dilation_w, const int deformable_group,
                             at::Tensor grad_offset);

void deformable_im2col(at::Tensor data_im,
                       at::Tensor data_offset, const int channels,
                       const int height, const int width, const int ksize_h, const int ksize_w, 
                       const int pad_h, const int pad_w, const int stride_h, const int stride_w, 
                       const int dilation_h, const int dilation_w,
                       const int deformable_group, at::Tensor data_col);
#endif
