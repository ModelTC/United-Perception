# Standard Library

# Import from third library
import torch
import torch.nn as nn

from eod.utils.model.normalize import build_norm_layer
from eod.utils.model.act_fn import build_act_fn


class ConvBnAct(nn.Module):
    # CBH/CBL
    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size=1,
                 stride=1,
                 padding=None,
                 groups=1,
                 normalize={'type': 'solo_bn'},
                 act_fn={'type': 'Hardswish'}):
        super(ConvBnAct, self).__init__()
        if padding is None:
            padding = kernel_size // 2 if isinstance(kernel_size, int) else [x // 2 for x in kernel_size]
        self.norm1_name, norm1 = build_norm_layer(out_planes, normalize, 1)
        self.act1_name, act1 = build_act_fn(act_fn)

        self.conv = nn.Conv2d(in_planes,
                              out_planes,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              groups=groups,
                              bias=False)
        self.add_module(self.norm1_name, norm1)
        self.add_module(self.act1_name, act1)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def act1(self):
        return getattr(self, self.act1_name)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm1(x)
        x = self.act1(x)
        return x


class Space2Depth(nn.Module):
    def __init__(self, block_size):
        super(Space2Depth, self).__init__()
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, C, H // self.bs, self.bs, W // self.bs, self.bs)  # (N, C, H//bs, bs, W//bs, bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # (N, bs, bs, C, H//bs, W//bs)
        x = x.view(N, C * (self.bs ** 2), H // self.bs, W // self.bs)  # (N, C*bs^2, H//bs, W//bs)
        return x


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self,
                 in_planes,
                 out_planes,
                 block_size=2,
                 kernel_size=1,
                 stride=1,
                 padding=None,
                 groups=1,
                 normalize={'type': 'solo_bn'},
                 act_fn={'type': 'Hardswish'}):
        super(Focus, self).__init__()
        self.space2depth = Space2Depth(block_size)
        self.conv_block = ConvBnAct(in_planes * (block_size ** 2), out_planes, kernel_size, stride,
                                    padding, groups, normalize=normalize, act_fn=act_fn)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        x = self.space2depth(x)
        x = self.conv_block(x)
        return x


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self,
                 in_planes,
                 out_planes,
                 groups=1,
                 shortcut=True,
                 normalize={'type': 'solo_bn'},
                 act_fn={'type': 'Hardswish'},
                 expansion=0.5):
        super(Bottleneck, self).__init__()
        hidden_planes = int(out_planes * expansion)  # hidden channels
        self.shortcut = shortcut and (in_planes == out_planes)

        self.conv_block1 = ConvBnAct(in_planes, hidden_planes, 1, 1, normalize=normalize, act_fn=act_fn)
        self.conv_block2 = ConvBnAct(hidden_planes, out_planes, 3, 1, normalize=normalize, act_fn=act_fn, groups=groups)

    def forward(self, x):
        if self.shortcut:
            residual = x
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        # residual
        if self.shortcut:
            x = x + residual
        return x


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self,
                 in_planes,
                 out_planes,
                 groups=1,
                 nums=1,
                 shortcut=True,
                 normalize={'type': 'solo_bn'},
                 act_fn={'type': 'Hardswish'},
                 expansion=0.5):
        super(BottleneckCSP, self).__init__()
        hidden_planes = int(out_planes * expansion)  # hidden channels
        self.norm1_name, norm1 = build_norm_layer(2 * hidden_planes, normalize, 1)

        self.conv_block1 = ConvBnAct(in_planes, hidden_planes, 1, 1, normalize=normalize, act_fn=act_fn)
        self.conv2 = nn.Conv2d(in_planes, hidden_planes, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(hidden_planes, hidden_planes, 1, 1, bias=False)
        self.conv_block4 = ConvBnAct(2 * hidden_planes, out_planes, 1, 1, normalize=normalize, act_fn=act_fn)
        self.add_module(self.norm1_name, norm1)   # applied to cat(conv2, conv3)
        self.act1_name, act1 = build_act_fn(act_fn)
        self.add_module(self.act1_name, act1)
        # self.act = nn.LeakyReLU(0.1, inplace=True)
        self.cbh_blocks = nn.Sequential(*[Bottleneck(hidden_planes, hidden_planes,
                                                     groups, shortcut, normalize,
                                                     act_fn=act_fn, expansion=1.0) for _ in range(nums)])

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def act1(self):
        return getattr(self, self.act1_name)

    def forward(self, x):
        csp1 = self.conv3(self.cbh_blocks(self.conv_block1(x)))
        csp2 = self.conv2(x)
        x = torch.cat((csp1, csp2), dim=1)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv_block4(x)
        return x


class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self,
                 in_planes,
                 out_planes,
                 kernels=(5, 9, 13),
                 normalize={'type': 'solo_bn'},
                 act_fn={'type': 'Hardswish'}):
        super(SPP, self).__init__()
        hidden_planes = in_planes // 2  # hidden channels
        self.conv_block1 = ConvBnAct(in_planes, hidden_planes, 1, 1, normalize=normalize, act_fn=act_fn)
        self.conv_block2 = ConvBnAct(hidden_planes * (len(kernels) + 1), out_planes, 1, 1,
                                     normalize=normalize, act_fn=act_fn)
        self.pooling_blocks = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in kernels])

    def forward(self, x):
        x = self.conv_block1(x)
        x = torch.cat([x] + [m(x) for m in self.pooling_blocks], 1)
        x = self.conv_block2(x)
        return x


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)
