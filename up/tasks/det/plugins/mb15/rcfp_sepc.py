import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.utils.checkpoint as checkpoint
from up.extensions import DeformableConv
from up.utils.model.normalize import build_norm_layer
from up.utils.general.registry_factory import MODULE_ZOO_REGISTRY
from .sepc import SEPC

__all__ = ['RCSEPC']


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def xavier_init(module, gain=1, bias=0, distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class conv3x3(nn.Module):
    def __init__(self,
                 in_planes=256,
                 out_planes=256,
                 stride=1,
                 normalize=None,
                 activation='relu'):
        super(conv3x3, self).__init__()
        """3x3 convolution with padding"""
        self.normalize = normalize
        self.activation = activation
        self.norm = nn.Identity()
        self.act = nn.Identity()
        if normalize is not None:
            norm_name, self.norm = build_norm_layer(out_planes, normalize)
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'gelu':
            self.act = nn.GELU()
        self.deform_offset = nn.Conv2d(
            in_planes, 18, kernel_size=3, stride=stride, padding=1)
        self.deform_conv = DeformableConv(in_planes, out_planes, kernel_size=3, stride=stride, padding=1)
        self.deform_offset.weight.data.zero_()
        if self.deform_offset.bias is not None:
            self.deform_offset.bias.data.zero_()
        xavier_init(self.deform_conv, distribution='uniform')

    def half(self):
        super().half()
        self.deform_offset.float()

    def forward(self, x):
        x = self.act(x)
        if x.dtype == torch.half:
            offset = self.deform_offset(x.float()).half()
        else:
            offset = self.deform_offset(x)

        input_pad = (x.size(2) < 3) or (x.size(3) < 3)
        if input_pad:
            pad_h = max(3 - x.size(2), 0)
            pad_w = max(3 - x.size(3), 0)
            x = F.pad(x, (0, pad_w, 0, pad_h), 'constant', 0).contiguous()
            offset = F.pad(offset, (0, pad_w, 0, pad_h), 'constant', 0)
            offset = offset.contiguous()
        x = self.deform_conv(x, offset)
        # x = checkpoint.checkpoint(self.deform_conv, *(x, offset))
        if input_pad:
            x = x[:, :, :x.size(2) - pad_h, :x.size(3)
                  - pad_w].contiguous()
        x = self.norm(x)
        return x


def conv1x1(in_planes, out_planes, normalize=None, activation='relu'):
    """1x1 convolution"""
    norm = nn.Identity()
    act = nn.Identity()
    if normalize is not None:
        norm_name, norm = build_norm_layer(out_planes, normalize)
    if activation == 'relu':
        act = nn.ReLU()
    elif activation == 'gelu':
        act = nn.GELU()
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
        norm,
        act
    )


class FusionNode(nn.Module):
    def __init__(self,
                 in_channels=256,
                 out_channels=256,
                 with_out_conv=True,
                 out_norm_cfg=None,
                 upsample_mode='bilinear',
                 op_num=2,
                 fusion_type='add',
                 activation='relu',
                 upsample_attn=True):
        super(FusionNode, self).__init__()
        self.with_out_conv = with_out_conv
        self.upsample_mode = upsample_mode
        self.op_num = op_num
        self.fusion_type = fusion_type
        self.upsample_attn = upsample_attn

        self.weight = nn.ModuleList()
        self.gap = nn.AdaptiveAvgPool2d(1)
        for i in range(op_num - 1):
            self.weight.append(nn.Conv2d(in_channels * 2, 1, kernel_size=1, bias=True))
            constant_init(self.weight[-1], 0)

        if self.fusion_type == 'concat':
            self.reduction = nn.ModuleList()
            for i in range(op_num - 1):
                norm_name, norm = build_norm_layer(in_channels, out_norm_cfg)
                self.reduction.append(nn.Sequential(
                    nn.ReLU(),
                    nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
                    norm))
                xavier_init(self.reduction[-1][1], distribution='uniform')

        if self.upsample_attn:
            self.spatial_weight = nn.Conv2d(in_channels * 2, 1, kernel_size=3, padding=1, bias=True)
            self.temp = nn.Parameter(torch.ones(1, dtype=torch.float32), requires_grad=True)
            for m in self.spatial_weight.modules():
                if isinstance(m, nn.Conv2d):
                    constant_init(m, 0)

        if self.with_out_conv:
            self.post_fusion = conv3x3(out_channels, out_channels, 1, out_norm_cfg, activation)
        if op_num > 2:
            self.pre_fusion = conv3x3(out_channels, out_channels, 1, out_norm_cfg, activation)

    def dynamicFusion(self, x):
        x1, x2 = x[0], x[1]
        batch, channel, height, width = x1.size()
        weight1 = self.gap(x1)
        weight2 = self.gap(x2)
        if self.upsample_attn:
            upsample_weight = self.spatial_weight(torch.cat((x1, x2), dim=1))
            if x1.dtype == torch.half:
                upsample_weight = upsample_weight.float()
            upsample_weight = upsample_weight * self.temp * (channel)**(-0.5)
            upsample_weight = F.softmax(upsample_weight.reshape(batch, 1, -1), dim=-1).reshape(batch, 1, height, width) * height # noqa
            upsample_weight = upsample_weight * width
            if x1.dtype == torch.half:
                upsample_weight = upsample_weight.half()
            x2 = upsample_weight * x2
        weight = torch.cat((weight1, weight2), dim=1)
        weight = self.weight[0](weight)
        weight = torch.sigmoid(weight)
        if self.fusion_type == 'concat':
            result = torch.cat((weight * x1, (1 - weight) * x2), dim=1)
            result = self.reduction[0](result)
        else:
            result = weight * x1 + (1 - weight) * x2
        if self.op_num == 3:
            x3 = x[2]
            # x1 = self.pre_fusion(result)
            x1 = checkpoint.checkpoint(self.pre_fusion, result)
            weight1 = self.gap(x1)
            weight3 = self.gap(x3)
            weight = torch.cat((weight1, weight3), dim=1)
            weight = self.weight[1](weight)
            weight = torch.sigmoid(weight)
            if self.fusion_type == 'concat':
                result = torch.cat((weight * x1, (1 - weight) * x3), dim=1)
                result = self.reduction[1](result)
            else:
                result = weight * x1 + (1 - weight) * x3
        if self.with_out_conv:
            # result = self.post_fusion(result)
            result = checkpoint.checkpoint(self.post_fusion, result)
        return result

    def _resize(self, x, size):
        if x.shape[-2:] == size:
            return x
        elif x.shape[-2:] < size:
            return F.interpolate(x, size=size, mode=self.upsample_mode)
        else:
            x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
            if not x.shape[-2:] == size:
                x = F.interpolate(x, size=size, mode=self.upsample_mode)
            return x

    def forward(self, x, out_size=None):
        inputs = []
        for feat in x:
            inputs.append(self._resize(feat, out_size))
        return self.dynamicFusion(inputs)


class ScaleBranch(nn.Module):
    def __init__(self,
                 in_channels=256,
                 out_channels=256):
        super(ScaleBranch, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatialPooling = nn.AdaptiveAvgPool3d((3, 1, 1))
        self.scalePooling = nn.AdaptiveAvgPool2d(1)
        self.channel_agg = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.trans = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)

    def forward(self, x):
        x = self.spatialPooling(x.permute(0, 2, 1, 3, 4)).reshape(x.size(0), x.size(2), 3, 1)
        batch, channel, height, width = x.size()
        channel_context = self.channel_agg(x)
        channel_context = channel_context.view(batch, 1, height * width)
        channel_context = F.softmax(channel_context, dim=-1)
        channel_context = channel_context * height * width
        channel_context = channel_context.view(batch, 1, height, width)
        context = self.scalePooling(x * channel_context)
        context = self.trans(context).unsqueeze(0)
        return context


class SpatialBranch(nn.Module):
    def __init__(self,
                 in_channels=256,
                 out_channels=256):
        super(SpatialBranch, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scalePooling = nn.AvgPool3d((3, 1, 1))
        self.spatialPooling = nn.AdaptiveAvgPool2d(1)
        self.channel_agg = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.trans = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)

    def half(self):
        pass

    def forward(self, x):
        is_half = False
        if x.dtype == torch.half:
            x = x.float()
            is_half = True
        x = self.scalePooling(x.permute(0, 2, 1, 3, 4)).squeeze(2)
        batch, channel, height, width = x.size()
        channel_context = self.channel_agg(x)
        channel_context = channel_context.view(batch, 1, height * width)
        channel_context = F.softmax(channel_context, dim=-1)
        channel_context = channel_context * height * width
        channel_context = channel_context.view(batch, 1, height, width)
        context = self.spatialPooling(x * channel_context)
        context = self.trans(context).unsqueeze(0)
        if is_half:
            context = context.half()
        return context


class ScaleShift(nn.Module):
    def __init__(self,
                 in_channels=256,
                 out_channels=256,
                 out_norm_cfg=None,
                 activation='relu'):
        super(ScaleShift, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fold_div = 8
        self.aggregation = nn.ModuleList()
        for i in range(3):
            self.aggregation.append(nn.Sequential(
                conv1x1(self.in_channels + self.in_channels * 2 // self.fold_div,
                        self.in_channels, out_norm_cfg, activation),
                nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)))

    def forward(self, x):
        res = x
        x = self.shift(x, self.fold_div)  # 5*B*C*H*W
        feats = []
        for i in range(3):
            feat = self.aggregation[i](x[i])
            feats.append(feat.unsqueeze(1))  # B*1*C*H*W
        feats = torch.cat(feats, dim=1)  # B*5*C*H*W
        return (res + feats)

    def shift(self, x, fold_div=8):
        B, L, C, H, W = x.size()
        fold = C // fold_div
        out = torch.zeros_like(x[:, :, : 2 * fold])
        out[:, :-1, :fold] = x[:, 1:, :fold]
        out[:, -1:, :fold] = x[:, :1, :fold]
        out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]
        out[:, :1, fold: 2 * fold] = x[:, -1:, fold: 2 * fold]
        out = torch.cat((out, x), dim=2)
        return out.permute(1, 0, 2, 3, 4)


class CrossShiftNet(nn.Module):
    def __init__(self,
                 in_channels=256,
                 out_channels=256,
                 out_norm_cfg=None,
                 activation='relu'):
        super(CrossShiftNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatialContext = SpatialBranch(self.out_channels, self.out_channels)
        self.scaleContext = ScaleBranch(self.out_channels, self.out_channels)
        self.msm = ScaleShift(self.in_channels, self.in_channels, out_norm_cfg, activation)

    def forward(self, inputs):
        feats = []
        size = inputs[1].size()[2:]
        feats.append((F.adaptive_max_pool2d(inputs[0], output_size=size)).unsqueeze(1))
        for i in range(1, 3):
            feat = F.interpolate(inputs[i], size=size, mode='bilinear')
            feats.append(feat.unsqueeze(1))  # B*1*C*H*W
        feats = torch.cat(feats, dim=1)  # B*5*C*H*W
        feats = self.msm(feats)
        out = feats.permute(1, 0, 2, 3, 4) + self.spatialContext(feats) + self.scaleContext(feats)
        inputs[0] = inputs[0] + F.interpolate(out[0], size=inputs[0].size()[2:], mode='bilinear')
        for i in range(1, 3):
            size = inputs[i].size()[2:]
            inputs[i] = inputs[i] + F.adaptive_max_pool2d(out[i], output_size=size)
        return inputs


@MODULE_ZOO_REGISTRY.register("RCSEPC")
class RCSEPC(nn.Module):
    def __init__(self,
                 inplanes,
                 outplanes,
                 start_level,
                 num_level,
                 out_strides,
                 downsample,
                 upsample,
                 activation='relu',
                 normalize={'type': 'sync_bn'},
                 tocaffe_friendly=False,
                 initializer=None,
                 num_revfp=1,
                 fusion_type='add',
                 pconv_num_sepc=4,
                 extra_conv_sepc=1,
                 double_head_sepc=False,
                 t_head_sepc=False,
                 light_sepc=False,
                 residual_sepc=False,
                 upsample_attention=True,
                 use_checkpoint=True,
                 align_corners=True,
                 to_onnx=False,
                 use_p5=False,
                 skip=False):
        super(RCSEPC, self).__init__()
        assert isinstance(inplanes, list)
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.outstrides = out_strides
        self.start_level = start_level
        self.num_level = num_level
        self.downsample = downsample
        self.upsample = upsample
        self.num_ins = len(inplanes)  # num of input feature levels
        self.num_outs = 5  # num of output feature levels
        self.normalize = normalize
        self.num_revfp = num_revfp
        self.use_checkpoint = use_checkpoint

        # add lateral connections
        self.lateral_convs = nn.ModuleList()
        for i in range(self.num_level - self.start_level + 1):
            l_conv = conv1x1(
                inplanes[i],
                outplanes,
                normalize,
                False
            )
            self.lateral_convs.append(l_conv)

        # add extra downsample layers (stride-2 pooling or conv)
        # extra_levels = num_outs - self.num_level + self.start_level - 1
        self.p5top6 = conv3x3(outplanes, outplanes, 2, normalize, None)
        self.p6top7 = conv3x3(outplanes, outplanes, 2, normalize)
        self.RevFP_list = nn.ModuleList()
        for i in range(self.num_revfp):
            RevFP = nn.ModuleDict()
            '''
            self.RevFP['p7'] = FusionNode(
                in_channels=outplanes,
                out_channels=outplanes,
                out_norm_cfg=normalize,
                op_num=2,
                activation=activation,
                upsample_attn=False)

            self.RevFP['p6'] = FusionNode(
                in_channels=outplanes,
                out_channels=outplanes,
                out_norm_cfg=normalize,
                op_num=3,
                activation=activation)
            '''

            RevFP['p5'] = FusionNode(
                in_channels=outplanes,
                out_channels=outplanes,
                out_norm_cfg=normalize,
                op_num=2,
                fusion_type=fusion_type,
                activation=activation,
                upsample_attn=False)

            RevFP['p4'] = FusionNode(
                in_channels=outplanes,
                out_channels=outplanes,
                out_norm_cfg=normalize,
                op_num=3,
                fusion_type=fusion_type,
                activation=activation,
                upsample_attn=upsample_attention)

            RevFP['p3'] = FusionNode(
                in_channels=outplanes,
                out_channels=outplanes,
                out_norm_cfg=normalize,
                op_num=2,
                fusion_type=fusion_type,
                activation=activation,
                upsample_attn=upsample_attention)
            self.RevFP_list.append(RevFP)

        self.CSN = CrossShiftNet(
            in_channels=outplanes,
            out_channels=outplanes,
            out_norm_cfg=normalize,
            activation=activation)

        self.init_weights()

        self.sepc = SEPC(in_channels=[outplanes] * 5, out_channels=outplanes, normalize=normalize, Pconv_num=pconv_num_sepc, num_conv=extra_conv_sepc, double_head_sepc=double_head_sepc, # noqa
                         t_head_sepc=t_head_sepc, light_sepc=light_sepc, with_residual=residual_sepc)

    def init_weights(self):
        """Initialize the weights of module."""
        for m in self.lateral_convs.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
        for m in self.CSN.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def half(self):
        pass

    def forward(self, input):
        """Forward function."""
        # build P3-P5
        inputs = input['features']
        is_half = (inputs[0].dtype == torch.half)
        if is_half:
            feats = [feat.float() for feat in inputs]
            inputs = feats
        feats = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        p3, p4, p5 = feats
        for RevFP in self.RevFP_list:
            p3 = RevFP['p3']([p3, p4], out_size=p3.shape[-2:])
            p4 = RevFP['p4']([p4, p5, p3], out_size=p4.shape[-2:])
            p5 = RevFP['p5']([p5, p4], out_size=p5.shape[-2:])

        p3, p4, p5 = self.CSN([p3, p4, p5])
        p6 = self.p5top6(p5)
        p7 = self.p6top7(p6)

        if self.use_checkpoint:
            p3, p4, p5, p6, p7 = checkpoint.checkpoint(self.sepc, *(p3, p4, p5, p6, p7))
        else:
            p3, p4, p5, p6, p7 = self.sepc(p3, p4, p5, p6, p7)
        out = [p3, p4, p5, p6, p7]
        if is_half:
            feats = [feat.half() for feat in out]
            out = feats
        return {'features': out, 'strides': self.outstrides}

    def get_outplanes(self):
        return self.outplanes
