try:
    from spring import linklink
except:  # noqa
    from up.utils.general.fake_linklink import linklink

import numpy as np
import torch.nn as nn
import math
from up.utils.model.initializer import initialize_from_cfg
from up.utils.model.normalize import build_norm_layer


regnetX_200M_config = {'WA': 36.44, 'W0': 24, 'WM': 2.49, 'DEPTH': 13, 'GROUP_W': 8, 'SE_ON': False}
regnetX_400M_config = {'WA': 24.48, 'W0': 24, 'WM': 2.54, 'DEPTH': 22, 'GROUP_W': 16, 'SE_ON': False}
regnetX_600M_config = {'WA': 36.97, 'W0': 48, 'WM': 2.24, 'DEPTH': 16, 'GROUP_W': 24, 'SE_ON': False}
regnetX_800M_config = {'WA': 35.73, 'W0': 56, 'WM': 2.28, 'DEPTH': 16, 'GROUP_W': 16, 'SE_ON': False}
regnetX_1600M_config = {'WA': 34.01, 'W0': 80, 'WM': 2.25, 'DEPTH': 18, 'GROUP_W': 24, 'SE_ON': False}
regnetX_3200M_config = {'WA': 26.31, 'W0': 88, 'WM': 2.25, 'DEPTH': 25, 'GROUP_W': 48, 'SE_ON': False}
regnetX_4000M_config = {'WA': 38.65, 'W0': 96, 'WM': 2.43, 'DEPTH': 23, 'GROUP_W': 40, 'SE_ON': False}
regnetX_6400M_config = {'WA': 60.83, 'W0': 184, 'WM': 2.07, 'DEPTH': 17, 'GROUP_W': 56, 'SE_ON': False}
regnetY_200M_config = {'WA': 36.44, 'W0': 24, 'WM': 2.49, 'DEPTH': 13, 'GROUP_W': 8, 'SE_ON': True}
regnetY_400M_config = {'WA': 27.89, 'W0': 48, 'WM': 2.09, 'DEPTH': 16, 'GROUP_W': 8, 'SE_ON': True}
regnetY_600M_config = {'WA': 32.54, 'W0': 48, 'WM': 2.32, 'DEPTH': 15, 'GROUP_W': 16, 'SE_ON': True}
regnetY_800M_config = {'WA': 38.84, 'W0': 56, 'WM': 2.4, 'DEPTH': 14, 'GROUP_W': 16, 'SE_ON': True}
regnetY_1600M_config = {'WA': 20.71, 'W0': 48, 'WM': 2.65, 'DEPTH': 27, 'GROUP_W': 24, 'SE_ON': True}
regnetY_3200M_config = {'WA': 42.63, 'W0': 80, 'WM': 2.66, 'DEPTH': 21, 'GROUP_W': 24, 'SE_ON': True}
regnetY_4000M_config = {'WA': 31.41, 'W0': 96, 'WM': 2.24, 'DEPTH': 22, 'GROUP_W': 64, 'SE_ON': True}
regnetY_6400M_config = {'WA': 33.22, 'W0': 112, 'WM': 2.27, 'DEPTH': 25, 'GROUP_W': 72, 'SE_ON': True}


__all__ = ['regnetx_200m', 'regnetx_400m', 'regnetx_600m', 'regnetx_800m',
           'regnetx_1600m', 'regnetx_3200m', 'regnetx_4000m', 'regnetx_6400m',
           'regnety_200m', 'regnety_400m', 'regnety_600m', 'regnety_800m',
           'regnety_1600m', 'regnety_3200m', 'regnety_4000m', 'regnety_6400m']


class SimpleStemIN(nn.Module):
    """Simple stem for ImageNet."""

    def __init__(self, in_w, out_w, normalize):
        super(SimpleStemIN, self).__init__()
        self._construct(in_w, out_w, normalize)

    def _construct(self, in_w, out_w, normalize):
        # 3x3, BN, ReLU
        self.conv = nn.Conv2d(
            in_w, out_w, kernel_size=3, stride=2, padding=1, bias=False
        )
        _, self.bn = build_norm_layer(out_w, normalize)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class SE(nn.Module):
    """Squeeze-and-Excitation (SE) block"""

    def __init__(self, w_in, w_se):
        super(SE, self).__init__()
        self._construct(w_in, w_se)

    def _construct(self, w_in, w_se):
        # AvgPool
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # FC, Activation, FC, Sigmoid
        self.f_ex = nn.Sequential(
            nn.Conv2d(w_in, w_se, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(w_se, w_in, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.f_ex(self.avg_pool(x))


class BottleneckTransform(nn.Module):
    """Bottlenect transformation: 1x1, 3x3, 1x1"""

    def __init__(self, w_in, w_out, stride, bm, gw, se_r, normalize):
        super(BottleneckTransform, self).__init__()
        self._construct(w_in, w_out, stride, bm, gw, se_r, normalize)

    def _construct(self, w_in, w_out, stride, bm, gw, se_r, normalize):
        # Compute the bottleneck width
        w_b = int(round(w_out * bm))
        # Compute the number of groups
        num_gs = w_b // gw
        # 1x1, BN, ReLU
        self.a = nn.Conv2d(w_in, w_b, kernel_size=1, stride=1, padding=0, bias=False)
        _, self.a_bn = build_norm_layer(w_b, normalize)
        self.a_relu = nn.ReLU(True)
        # 3x3, BN, ReLU
        self.b = nn.Conv2d(
            w_b, w_b, kernel_size=3, stride=stride, padding=1, groups=num_gs, bias=False
        )
        _, self.b_bn = build_norm_layer(w_b, normalize)
        self.b_relu = nn.ReLU(True)
        # Squeeze-and-Excitation (SE)
        if se_r:
            w_se = int(round(w_in * se_r))
            self.se = SE(w_b, w_se)
        # 1x1, BN
        self.c = nn.Conv2d(w_b, w_out, kernel_size=1, stride=1, padding=0, bias=False)
        _, self.c_bn = build_norm_layer(w_out, normalize)
        self.c_bn.final_bn = True

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class ResBottleneckBlock(nn.Module):
    """Residual bottleneck block: x + F(x), F = bottleneck transform"""

    def __init__(self, w_in, w_out, stride, bm=1.0, gw=1, se_r=None, normalize=None):
        super(ResBottleneckBlock, self).__init__()
        self._construct(w_in, w_out, stride, bm, gw, se_r, normalize)

    def _add_skip_proj(self, w_in, w_out, stride, normalize):
        self.proj = nn.Conv2d(
            w_in, w_out, kernel_size=1, stride=stride, padding=0, bias=False
        )
        _, self.bn = build_norm_layer(w_out, normalize)

    def _construct(self, w_in, w_out, stride, bm, gw, se_r, normalize):
        # Use skip connection with projection if shape changes
        self.proj_block = (w_in != w_out) or (stride != 1)
        if self.proj_block:
            self._add_skip_proj(w_in, w_out, stride, normalize)
        self.f = BottleneckTransform(w_in, w_out, stride, bm, gw, se_r, normalize)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        if self.proj_block:
            x = self.bn(self.proj(x)) + self.f(x)
        else:
            x = x + self.f(x)
        x = self.relu(x)
        return x


class AnyHead(nn.Module):
    """AnyNet head."""

    def __init__(self, w_in, nc):
        super(AnyHead, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(w_in, nc, bias=True)

    def forward(self, x):
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class AnyStage(nn.Module):
    """AnyNet stage (sequence of blocks w/ the same output shape)."""

    def __init__(self, w_in, w_out, stride, d, block_fun, bm, gw, se_r, normalize):
        super(AnyStage, self).__init__()
        self._construct(w_in, w_out, stride, d, block_fun, bm, gw, se_r, normalize)

    def _construct(self, w_in, w_out, stride, d, block_fun, bm, gw, se_r, normalize):
        # Construct the blocks
        for i in range(d):
            # Stride and w_in apply to the first block of the stage
            b_stride = stride if i == 0 else 1
            b_w_in = w_in if i == 0 else w_out
            # Construct the block
            self.add_module(
                "b{}".format(i + 1), block_fun(b_w_in, w_out, b_stride, bm, gw, se_r, normalize)
            )

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x


class AnyNet(nn.Module):
    """AnyNet model."""

    def __init__(self, **kwargs):
        super(AnyNet, self).__init__()
        self.layer0 = None
        self.layer1 = None
        self.layer2 = None
        self.layer3 = None
        self.layer4 = None
        if kwargs:
            self._construct(
                stem_w=kwargs["stem_w"],
                ds=kwargs["ds"],
                ws=kwargs["ws"],
                ss=kwargs["ss"],
                bms=kwargs["bms"],
                gws=kwargs["gws"],
                se_r=kwargs["se_r"],
                nc=kwargs["nc"],
                normalize=kwargs["normalize"],
            )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Note that there is no bias due to BN
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=math.sqrt(2.0 / fan_out))
            elif (isinstance(m, linklink.nn.SyncBatchNorm2d) or isinstance(m, nn.BatchNorm2d)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 1.0 / float(n))
                m.bias.data.zero_()

    def _construct(self, stem_w, ds, ws, ss, bms, gws, se_r, nc, normalize):
        # Generate dummy bot muls and gs for models that do not use them
        bms = bms if bms else [1.0 for _d in ds]
        gws = gws if gws else [1 for _d in ds]
        # Group params by stage
        stage_params = list(zip(ds, ws, ss, bms, gws))
        # Construct the stem
        self.stem = SimpleStemIN(3, stem_w, normalize)
        # Construct the stages
        block_fun = ResBottleneckBlock
        prev_w = stem_w
        helper = [self.layer0, self.layer1, self.layer2, self.layer3, self.layer4]
        for i, (d, w, s, bm, gw) in enumerate(stage_params):
            helper[i] = AnyStage(prev_w, w, s, d, block_fun, bm, gw, se_r, normalize)
            self.add_module("s{}".format(i + 1), helper[i])
            prev_w = w

    def freeze_layer(self):
        layers = [
            self.layer0,
            self.layer1, self.layer2, self.layer3, self.layer4
        ]
        for layer_idx in self.frozen_layers:
            layer = layers[layer_idx]
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, input):
        x = input['image']
        outs = []
        for module in self.children():
            x = module(x)
            outs.append(x)
        ret = []
        for idx in self.out_layers:
            ret.append(outs[idx])
        return {'features': ret, 'strides': self.get_outstrides()}

    def get_outplanes(self):
        return self.out_planes

    def get_outstrides(self):
        return self.out_strides


def quantize_float(f, q):
    """Converts a float to closest non-zero int divisible by q."""
    return int(round(f / q) * q)


def adjust_ws_gs_comp(ws, bms, gs):
    """Adjusts the compatibility of widths and groups."""
    ws_bot = [int(w * b) for w, b in zip(ws, bms)]
    gs = [min(g, w_bot) for g, w_bot in zip(gs, ws_bot)]
    ws_bot = [quantize_float(w_bot, g) for w_bot, g in zip(ws_bot, gs)]
    ws = [int(w_bot / b) for w_bot, b in zip(ws_bot, bms)]
    return ws, gs


def get_stages_from_blocks(ws, rs):
    """Gets ws/ds of network at each stage from per block values."""
    ts_temp = zip(ws + [0], [0] + ws, rs + [0], [0] + rs)
    ts = [w != wp or r != rp for w, wp, r, rp in ts_temp]
    s_ws = [w for w, t in zip(ws, ts[:-1]) if t]
    s_ds = np.diff([d for d, t in zip(range(len(ts)), ts) if t]).tolist()
    return s_ws, s_ds


def generate_regnet(w_a, w_0, w_m, d, q=8):
    """Generates per block ws from RegNet parameters.

    args:
        w_a(float): slope
        w_0(int): initial width
        w_m(float): an additional parameter that controls quantization
        d(int): number of depth
        q(int): the coefficient of division

    procedure:
        1. generate a linear parameterization for block widths. Eql(2)
        2. compute corresponding stage for each block $log_{w_m}^{w_j/w_0}$. Eql(3)
        3. compute per-block width via $w_0*w_m^(s_j)$ and qunatize them that can be divided by q. Eql(4)

    return:
        ws(list of quantized float): quantized width list for blocks in different stages
        num_stages(int): total number of stages
        max_stage(float): the maximal index of stage
        ws_cont(list of float): original width list for blocks in different stages
    """
    assert w_a >= 0 and w_0 > 0 and w_m > 1 and w_0 % q == 0
    ws_cont = np.arange(d) * w_a + w_0
    ks = np.round(np.log(ws_cont / w_0) / np.log(w_m))
    ws = w_0 * np.power(w_m, ks)
    ws = np.round(np.divide(ws, q)) * q
    num_stages, max_stage = len(np.unique(ws)), ks.max() + 1
    ws, ws_cont = ws.astype(int).tolist(), ws_cont.tolist()
    return ws, num_stages, max_stage, ws_cont


class RegNet(AnyNet):
    """RegNet model."""

    def __init__(self, cfg, frozen_layers=None, out_layers=None, out_strides=None, normalize=None, initializer=None):
        self.frozen_layers = frozen_layers
        self.out_layers = out_layers
        self.out_strides = out_strides
        self.normalize = normalize
        self.initializer = initializer
        # Generate RegNet ws per block
        b_ws, num_s, _, _ = generate_regnet(
            cfg['WA'], cfg['W0'], cfg['WM'], cfg['DEPTH']
        )
        # Convert to per stage format
        ws, ds = get_stages_from_blocks(b_ws, b_ws)
        # Generate group widths and bot muls
        gws = [cfg['GROUP_W'] for _ in range(num_s)]
        bms = [1 for _ in range(num_s)]
        # Adjust the compatibility of ws and gws
        ws, gws = adjust_ws_gs_comp(ws, bms, gws)
        self.out_planes = ws
        # Use the same stride for each stage, stride set to 2
        ss = [2 for _ in range(num_s)]
        # Use SE for RegNetY
        se_r = 0.25 if cfg['SE_ON'] else None
        # Construct the model
        STEM_W = 32

        kwargs = {
            "stem_w": STEM_W,
            "ss": ss,
            "ds": ds,
            "ws": ws,
            "bms": bms,
            "gws": gws,
            "se_r": se_r,
            "nc": 1000,
            "normalize": normalize,
        }
        super(RegNet, self).__init__(**kwargs)
        print(self.initializer, "None!!!!!!!!!")
        if self.initializer is not None:
            initialize_from_cfg(self, self.initializer)
        if self.frozen_layers is not None:
            self.freeze_layer()


def regnetx_200m(**kwargs):
    model = RegNet(regnetX_200M_config, **kwargs)
    return model


def regnetx_400m(**kwargs):
    model = RegNet(regnetX_400M_config, **kwargs)
    return model


def regnetx_600m(**kwargs):
    model = RegNet(regnetX_600M_config, **kwargs)
    return model


def regnetx_800m(**kwargs):
    model = RegNet(regnetX_800M_config, **kwargs)
    return model


def regnetx_1600m(**kwargs):
    model = RegNet(regnetX_1600M_config, **kwargs)
    return model


def regnetx_3200m(**kwargs):
    model = RegNet(regnetX_3200M_config, **kwargs)
    return model


def regnetx_4000m(**kwargs):
    model = RegNet(regnetX_4000M_config, **kwargs)
    return model


def regnetx_6400m(**kwargs):
    model = RegNet(regnetX_6400M_config, **kwargs)
    return model


def regnety_200m(**kwargs):
    model = RegNet(regnetY_200M_config, **kwargs)
    return model


def regnety_400m(**kwargs):
    model = RegNet(regnetY_400M_config, **kwargs)
    return model


def regnety_600m(**kwargs):
    model = RegNet(regnetY_600M_config, **kwargs)
    return model


def regnety_800m(**kwargs):
    model = RegNet(regnetY_800M_config, **kwargs)
    return model


def regnety_1600m(**kwargs):
    model = RegNet(regnetY_1600M_config, **kwargs)
    return model


def regnety_3200m(**kwargs):
    model = RegNet(regnetY_3200M_config, **kwargs)
    return model


def regnety_4000m(**kwargs):
    model = RegNet(regnetY_4000M_config, **kwargs)
    return model


def regnety_6400m(**kwargs):
    model = RegNet(regnetY_6400M_config, **kwargs)
    return model
