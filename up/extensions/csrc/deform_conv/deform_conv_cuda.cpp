#include "deform_conv/deformable_conv.h"

#include <cstdio>
using at::Tensor;


int deform_conv_forward_cuda(Tensor input, Tensor weight,
                             Tensor offset, Tensor output,
                             Tensor columns, Tensor ones, int kH,
                             int kW, int dH, int dW, int padH, int padW,
                             int dilationH, int dilationW, int groups,
                             int deformable_group) {

  input = input.contiguous();
  offset = offset.contiguous();
  weight = weight.contiguous();
  output = output.contiguous();
  columns = columns.contiguous();
  ones = ones.contiguous();

  int batch = 1;

  if (input.ndimension() == 3) {
    // Force batch
    batch = 0;
    input = input.view({1, input.size(0), input.size(1), input.size(2)});
    offset = offset.view({1, offset.size(0), offset.size(1), offset.size(2)});
  }

  long batchSize = input.size(0);
  long nInputPlane = input.size(1);
  long inputHeight = input.size(2);
  long inputWidth = input.size(3);

  long nOutputPlane = weight.size(0);

  long outputWidth =
      (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  long outputHeight =
      (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;
  output = output.view({batchSize, groups, (int)(nOutputPlane / groups), outputHeight * outputWidth});

  columns = columns.view({groups, (int)(nInputPlane / groups )* kW * kH, outputHeight * outputWidth}).zero_();

  weight = weight.view({groups, (int)(nOutputPlane / groups), (int)(nInputPlane / groups) * kH * kW});
  if (ones.ndimension() != 2 ||
      ones.size(0) * ones.size(1) < outputHeight * outputWidth) {
    ones = ones.view({outputHeight, outputWidth});
    ones.fill_(1);
  }

  
  int elt;
  for (elt = 0; elt < batchSize; elt++) {

    Tensor input_n = input[elt];
    Tensor offset_n = offset[elt];
    Tensor output_n = output[elt];


    output_n.zero_();

    deformable_im2col(input_n, offset_n, nInputPlane, inputHeight,
        inputWidth, kH, kW, padH, padW, dH, dW, dilationH, dilationW,
        deformable_group, columns);


    int g = 0;
    for (g = 0; g < groups; g++) {
        Tensor columns_g = columns[g];
        Tensor weight_g =weight[g];
        Tensor output_g =output_n[g];

        output_g.copy_(mm(weight_g, columns_g) + output_g);
    }

    }

  // printf("Here *****\n");
  if (batch == 0) {
    output = output.view({nOutputPlane, outputHeight, outputWidth});
    input = input.view({nInputPlane, inputHeight, inputWidth});
    offset = offset.view({offset.size(1), offset.size(2), offset.size(3)});
  }

  output = output.view({batchSize, nOutputPlane, outputHeight, outputWidth});

  weight = weight.view({nOutputPlane, (int)(nInputPlane / groups), kH,kW});

  return 1;
}

int deform_conv_backward_input_cuda(
    Tensor input, Tensor offset, Tensor gradOutput,
    Tensor gradInput, Tensor gradOffset, Tensor weight,
    Tensor columns, int kH, int kW, int dH, int dW, int padH, int padW,
    int dilationH, int dilationW, int groups, int deformable_group) {

  input = input.contiguous();
  offset = offset.contiguous();
  gradInput = gradInput.contiguous();
  gradOutput = gradOutput.contiguous();
  gradOffset = gradOffset.contiguous();
  weight = weight.contiguous();
  columns = columns.contiguous();

  int batch = 1;
  if (input.ndimension() == 3) {
    // Force batch
    batch = 0;
    input = input.view({1, input.size(0), input.size(1), input.size(2)});
    offset = offset.view({1, offset.size(0), offset.size(1), offset.size(2)});
    gradOutput = gradOutput.view({1, gradOutput.size(0), gradOutput.size(1), gradOutput.size(2)});
  }

  long batchSize = input.size(0);
  long nInputPlane = input.size(1);
  long inputHeight = input.size(2);
  long inputWidth = input.size(3);

  long nOutputPlane = weight.size(0);

  long outputWidth =
      (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  long outputHeight =
      (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;

  // THArgCheck((offset.size(0) == batchSize), 3, "invalid batch size of offset");

  long m = (int)(nInputPlane / groups) * kW * kH;
  long n = outputHeight * outputWidth;
  long k = (int)(nOutputPlane / groups);

  gradOutput = gradOutput.view({batchSize, groups, k, n});

  int elt;
  for (elt = 0; elt < batchSize; elt++) {
    Tensor gradInput_n = gradInput[elt];
    Tensor gradOffset_n = gradOffset[elt];
    Tensor input_n = input[elt];
    Tensor offset_n = offset[elt];
    Tensor gradOutput_n = gradOutput[elt];

    int g;

    columns = columns.view({groups, m, n});
    weight = weight.view({groups, k, m});
    for (g = 0; g < groups; g++) {
        Tensor gradOutput_g = gradOutput_n[g];
        Tensor weight_g = weight[g];
        Tensor columns_g = columns[g];

        columns_g.copy_(mm(weight_g.t(), gradOutput_g));
    }

    deformable_col2im_coord(columns, input_n, offset_n,
        nInputPlane, inputHeight, inputWidth, kH, kW, padH, padW, dH, dW,
        dilationH, dilationW, deformable_group, gradOffset_n);

    deformable_col2im(columns, offset_n, nInputPlane, inputHeight,
        inputWidth, kH, kW, padH, padW, dH, dW, dilationH, dilationW,
        deformable_group, gradInput_n);
  }

  weight = weight.view({nOutputPlane,  (int)(nInputPlane / groups), kH, kW});
  gradOutput = gradOutput.view({batchSize, nOutputPlane, outputHeight, outputWidth});

  if (batch == 0) {
    gradOutput = gradOutput.view({nOutputPlane, outputHeight, outputWidth});
    input = input.view({nInputPlane, inputHeight, inputWidth});
    gradInput = gradInput.view({nInputPlane, inputHeight, inputWidth});
    offset = offset.view({offset.size(1), offset.size(2), offset.size(3)});
    gradOffset = gradOffset.view({offset.size(1), offset.size(2), offset.size(3)});
  }

  return 1;
}

int deform_conv_backward_parameters_cuda(
    Tensor input, Tensor offset, Tensor gradOutput,
    Tensor gradWeight, /*Tensor gradBias, */
    Tensor columns, Tensor ones, int kH, int kW, int dH, int dW,
    int padH, int padW, int dilationH, int dilationW, int groups, int deformable_group,
    float scale) {

  input = input.contiguous();
  offset = offset.contiguous();
  gradOutput = gradOutput.contiguous();
  gradWeight = gradWeight.contiguous();
  columns = columns.contiguous();
  ones = ones.contiguous();  

  int batch = 1;
  // printf("%d %d %d",input.size(0), input.size(1), input.size(2));
  if (input.ndimension() == 3) {
    // Force batch
    batch = 0;
    input = input.view({1, input.size(0), input.size(1), input.size(2)});
    gradOutput = gradOutput.view({1, gradOutput.size(0), gradOutput.size(1), gradOutput.size(2)});
  }

  //printf("%d %d %d %d\n", gradOutput.size(0), gradOutput.size(1), gradOutput.size(2), gradOutput.size(3));
  long batchSize = input.size(0);
  long nInputPlane = input.size(1);
  long inputHeight = input.size(2);
  long inputWidth = input.size(3);

  long nOutputPlane = gradWeight.size(0);

  long outputWidth =
      (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  long outputHeight =
      (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;

  long m = (int)(nInputPlane / groups) * kW * kH;
  long n = outputHeight * outputWidth;
  long k = (int)(nOutputPlane / groups);

  columns = columns.view({groups, m, n});
  gradOutput = gradOutput.view({batchSize, groups, k, n});
  gradWeight = gradWeight.view({groups, k, m});

  gradWeight.zero_();
  int elt;
  for (elt = 0; elt < batchSize; elt++) {
    Tensor input_n = input[elt];
    Tensor offset_n = offset[elt];
    Tensor gradOutput_n = gradOutput[elt];

    deformable_im2col(input_n, offset_n, nInputPlane, inputHeight,
        inputWidth, kH, kW, padH, padW, dH, dW, dilationH, dilationW,
        deformable_group, columns);

    int g;
    for (g = 0; g < groups; g++) {
        Tensor columns_g = columns[g];
        Tensor gradOutput_g = gradOutput_n[g];
        Tensor gradWeight_g = gradWeight[g];

        gradWeight_g.copy_(scale * mm( gradOutput_g, columns_g.t()) + gradWeight_g);
    }

  }
  gradWeight = gradWeight.view({nOutputPlane,  (int)nInputPlane / groups, kH, kW});
  gradOutput = gradOutput.view({batchSize, nOutputPlane, outputHeight, outputWidth});


  if (batch == 0) {
    gradOutput = gradOutput.view({nOutputPlane, outputHeight, outputWidth});
    input = input.view({nInputPlane, inputHeight, inputWidth});
  }

  return 1;
}
