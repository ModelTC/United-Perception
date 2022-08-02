import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init as init
from up.extensions import DeformableConv
from up.utils.model.normalize import build_norm_layer
from up.utils.general.registry_factory import MODULE_ZOO_REGISTRY


__all__ = ['SEPC']


class conv3x3(nn.Module):
    def __init__(self,
                 in_planes=256,
                 out_planes=256,
                 stride=1,
                 normalize=None,
                 activation=None):
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
        self.deform_offset.bias.data.zero_()
        init.normal_(self.deform_conv.weight.data, 0, 0.01)

    def half(self):
        super().half()
        self.deform_offset.float()

    def forward(self, x):
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
        if input_pad:
            x = x[:, :, :x.size(2) - pad_h, :x.size(3)
                  - pad_w].contiguous()
        x = self.norm(x)
        x = self.act(x)
        return x


class TaskDecomposition(nn.Module):
    def __init__(self, feat_channels, stacked_convs, la_down_rate=8, norm_cfg={'type': 'sync_bn'}):
        super(TaskDecomposition, self).__init__()
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.in_channels = self.feat_channels * self.stacked_convs
        self.norm_cfg = norm_cfg
        self.la_conv1 = nn.Conv2d(self.in_channels, self.in_channels // la_down_rate, 1)
        self.relu = nn.ReLU(inplace=True)
        self.la_conv2 = nn.Conv2d(self.in_channels // la_down_rate, self.stacked_convs, 1, padding=0)
        self.sigmoid = nn.Sigmoid()

        self.reduction_conv = nn.Conv2d(self.in_channels, self.feat_channels, 1)
        _, self.reduction_norm = build_norm_layer(feat_channels, norm_cfg)
        self.init_weights()

    def init_weights(self):
        init.normal_(self.la_conv1.weight.data, std=0.001)
        init.normal_(self.la_conv2.weight.data, std=0.001)
        self.la_conv2.bias.data.zero_()
        init.normal_(self.reduction_conv.weight.data, std=0.01)

    def forward(self, feat, avg_feat=None):
        b, c, h, w = feat.shape
        if avg_feat is None:
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
        weight = self.relu(self.la_conv1(avg_feat))
        weight = self.sigmoid(self.la_conv2(weight))

        # here we first compute the product between layer attention weight and conv weight,
        # and then compute the convolution between new conv weight and feature map,
        # in order to save memory and FLOPs.
        conv_weight = weight.reshape(b, 1, self.stacked_convs, 1) * \
            self.reduction_conv.weight.reshape(1, self.feat_channels, self.stacked_convs, self.feat_channels)
        conv_weight = conv_weight.reshape(b, self.feat_channels, self.in_channels)
        feat = feat.reshape(b, self.in_channels, h * w)
        feat = torch.bmm(conv_weight, feat).reshape(b, self.feat_channels, h, w)
        feat = self.reduction_norm(feat)
        return self.relu(feat)


@MODULE_ZOO_REGISTRY.register('SEPC')
class SEPC(nn.Module):
    def __init__(
        self,
        in_channels=[256] * 5,
        out_channels=256,
        num_outs=5,
        pconv_deform=True,
        lcconv_deform=True,
        normalize={'type': 'sync_bn'},
        Pconv_num=4,
        num_conv=2,
        double_head_sepc=False,
        t_head_sepc=False,
        light_sepc=False,
        part_fusion=False,
        with_residual=False
    ):
        super(SEPC, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.Pconv_num = Pconv_num
        self.num_conv = num_conv
        assert num_outs == 5
        self.fp16_enabled = False
        self.double_head_sepc = double_head_sepc
        self.t_head_sepc = t_head_sepc
        self.light_sepc = light_sepc
        self.part_fusion = part_fusion
        self.with_residual = with_residual

        self.Pconvs = nn.ModuleList()
        self.relu = nn.ReLU(inplace=True)

        for i in range(Pconv_num):
            self.Pconvs.append(
                PConvModule(in_channels[0],
                            out_channels,
                            normalize=normalize,
                            part_deform=pconv_deform))

        if self.double_head_sepc:
            self.lconv = nn.ModuleList()
            self.cconv = nn.ModuleList()
            self.lbn = nn.ModuleList()
            self.cbn = nn.ModuleList()
            for i in range(num_conv):
                self.lconv.append(
                    conv3x3(out_channels, out_channels))
                self.cconv.append(
                    conv3x3(out_channels, out_channels))
                _, lbn = build_norm_layer(out_channels, normalize)
                _, cbn = build_norm_layer(out_channels, normalize)
                self.lbn.append(lbn)
                self.cbn.append(cbn)

        if self.t_head_sepc:
            self.cls_decomp = TaskDecomposition(out_channels, self.Pconv_num, self.Pconv_num * 8, normalize)
            self.reg_decomp = TaskDecomposition(out_channels, self.Pconv_num, self.Pconv_num * 8, normalize)

    def forward(self, p2, p3, p4, p5, p6):
        if self.part_fusion:
            inputs = [p3, p4, p5, p6]
        else:
            inputs = [p2, p3, p4, p5, p6]
        x = inputs
        inter_feats = [[] for i in range(len(x))]
        for i, pconv in enumerate(self.Pconvs):
            if self.light_sepc:
                p2, p3, p4, p5, p6 = x
                if self.Pconv_num % 2 == 0:
                    if i % 2 == 0:
                        x = [p3, p4, p5, p6]
                        x = pconv(x)
                        p3, p4, p5, p6 = x
                        x = [p2, p3, p4, p5, p6]
                    else:
                        x = pconv(x)
                else:
                    if i % 2 == 1:
                        x = [p3, p4, p5, p6]
                        x = pconv(x)
                        p3, p4, p5, p6 = x
                        x = [p2, p3, p4, p5, p6]
                    else:
                        x = pconv(x)
            else:
                if self.with_residual:
                    y = pconv(x)
                    x = [x[i] + y[i] for i in range(len(x))]
                else:
                    x = pconv(x)
            for i in range(len(x)):
                inter_feats[i].append(x[i])
        if (not self.t_head_sepc) and (not self.double_head_sepc):
            if self.part_fusion:
                return x[0], x[1], x[2], x[3]
            return x[0], x[1], x[2], x[3], x[4]
        cls = x
        loc = x
        if self.t_head_sepc:
            cls = []
            loc = []
            for i in range(len(x)):
                level_feats = inter_feats[i]
                feat = torch.cat(level_feats, 1)
                avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
                cls_feat = self.cls_decomp(feat, avg_feat)
                reg_feat = self.reg_decomp(feat, avg_feat)
                cls.append(cls_feat)
                loc.append(reg_feat)
        if self.double_head_sepc:
            for i in range(self.num_conv):
                cconv, lconv = self.cconv[i], self.lconv[i]
                cbn, lbn = self.cbn[i], self.lbn[i]
                cls = [cconv(item) for level, item in enumerate(cls)]
                loc = [lconv(item) for level, item in enumerate(loc)]
                cls = iBN(cls, cbn)
                loc = iBN(loc, lbn)
                cls = [self.relu(item) for item in cls]
                loc = [self.relu(item) for item in loc]
        outs = [torch.cat((s.unsqueeze(0), l.unsqueeze(0)), dim=0) for s, l in zip(cls, loc)]
        return outs[0], outs[1], outs[2], outs[3], outs[4]


class PConvModule(nn.Module):
    def __init__(
        self,
        in_channels=256,
        out_channels=256,
        kernel_size=[3, 3, 3],
        dilation=[1, 1, 1],
        groups=[1, 1, 1],
        normalize=None,
        part_deform=False,
    ):
        super(PConvModule, self).__init__()
        #     assert not (bias and iBN)
        self.iBN = True
        self.Pconv = nn.ModuleList()
        self.Pconv.append(
            conv3x3(out_channels, out_channels))

        self.Pconv.append(
            conv3x3(out_channels, out_channels))

        self.Pconv.append(
            conv3x3(out_channels, out_channels, stride=2))

        if normalize is not None:
            _, self.bn = build_norm_layer(out_channels, normalize)
        self.relu = nn.ReLU()

    def forward(self, x):
        next_x = []
        for level, feature in enumerate(x):
            temp_fea = self.Pconv[1](feature)
            if level > 0:
                temp_fea += self.Pconv[2](x[level - 1])
            if level < len(x) - 1:
                temp_fea += F.upsample_bilinear(
                    self.Pconv[0](x[level + 1]),
                    size=[temp_fea.size(2), temp_fea.size(3)])
            next_x.append(temp_fea)
        next_x = iBN(next_x, self.bn)
        next_x = [self.relu(item) for item in next_x]
        return next_x


def iBN(fms, bn):
    sizes = [p.shape[2:] for p in fms]
    n, c = fms[0].shape[0], fms[0].shape[1]
    fm = torch.cat([p.view(n, c, 1, -1) for p in fms], dim=-1)
    fm = bn(fm)
    fm = torch.split(fm, [s[0] * s[1] for s in sizes], dim=-1)
    return [p.view(n, c, s[0], s[1]) for p, s in zip(fm, sizes)]
