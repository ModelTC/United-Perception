import torch.nn as nn
from up.utils.model.normalize import build_norm_layer


class ConvBnRelu(nn.Module):
    def __init__(self,
                 in_planes,
                 out_planes,
                 ksize,
                 stride=1,
                 pad=0,
                 dilation=1,
                 groups=1,
                 has_bn=True,
                 normalize={'type': 'solo_bn'},
                 has_relu=True,
                 inplace=True,
                 has_bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize,
                              stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = build_norm_layer(out_planes, normalize)[1]
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)
        return x
