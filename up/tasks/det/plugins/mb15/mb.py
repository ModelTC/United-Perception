import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.autograd import Variable
from torch.nn.init import zeros_

from up.utils.model.normalize import build_norm_layer
from up.utils.model.utils import DropPath, Mlp
from up.utils.model.initializer import trunc_normal_, lecun_normal_
from up.utils.general.registry_factory import MODULE_ZOO_REGISTRY
from up.utils.general.log_helper import default_logger as logger

try:
    from lightseq.training import LSTransformerEncoderLayer
except: # noqa
    LSTransformerEncoderLayer = None

__all__ = ['MB4', 'MB4_gvm', 'MB7', 'MB15']


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def to4d(x, h, w):
    if len(x.shape) == 4:
        return x
    B, N, C = x.shape
    return x.transpose(1, 2).reshape(B, C, h, w)


def to3d(x):
    if len(x.shape) == 3:
        return x
    B, C, h, w = x.shape
    N = h * w
    return x.reshape(B, C, N).transpose(1, 2)


class SqueezeExcitation(nn.Module):

    def __init__(self, input_channels: int, squeeze_factor: int = 4):
        super().__init__()
        squeeze_channels = _make_divisible(input_channels // squeeze_factor, 8)
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)

    def _scale(self, input, inplace):
        scale = F.adaptive_avg_pool2d(input, 1)
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        scale = F.relu(scale / 6 + 0.5)
        scale = 1 - F.relu(1 - scale)
        return scale
        # return F.hardsigmoid(scale)
        # return hard_sigmoid(scale, inplace=inplace)

    def forward(self, input):
        scale = self._scale(input, True)
        return scale * input


class MixerBlock(nn.Module):
    def __init__(
        self, in_features, out_features=None, stride=1,
            mlp_ratio=4, use_se=True, drop=0., drop_path=0.,
            seq_l=196, head_dim=32, init_values=1e-6, **kwargs):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = int(in_features * mlp_ratio)
        tokens_dim = int(seq_l * stride ** 2 * 2)
        # normalize = {'type': 'sync_bn'}
        if in_features != out_features or stride != 1:
            self.r1 = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=stride, padding=1),
                nn.Conv2d(in_features, in_features, kernel_size=1)
            )
            self.r2 = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                nn.Conv2d(in_features, out_features, kernel_size=1)
            )
            self.norm1 = nn.LayerNorm(in_features)
            self.mlp_tokens = Mlp(seq_l * stride ** 2, tokens_dim, out_features=seq_l, act_layer=nn.GELU, drop=drop)
            self.norm2 = nn.LayerNorm(in_features)
            self.mlp_channels = Mlp(in_features, hidden_features,
                                    out_features=out_features, act_layer=nn.GELU, drop=drop)
            self.drop_path = nn.Identity()
        else:
            self.r1 = nn.Identity()
            self.r2 = nn.Identity()
            self.norm1 = nn.LayerNorm(in_features)
            self.mlp_tokens = Mlp(seq_l, tokens_dim, act_layer=nn.GELU, drop=drop)
            self.norm2 = nn.LayerNorm(in_features)
            self.mlp_channels = Mlp(in_features, hidden_features, act_layer=nn.GELU, drop=drop)
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
            self.gamma_1 = nn.Parameter(init_values * torch.ones((out_features)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((out_features)), requires_grad=True)

    def forward(self, x):
        residual = self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2)
        if isinstance(self.r1, nn.Identity):
            x = x + self.gamma_1 * self.drop_path(residual)
        else:
            x = to3d(self.r1(to4d(x))) + self.drop_path(residual)

        residual = self.mlp_channels(self.norm2(x))
        if isinstance(self.r2, nn.Identity):
            x = x + self.gamma_2 * self.drop_path(residual)
        else:
            x = to3d(self.r2(to4d(x))) + self.drop_path(residual)
        return x


class FusedMBConv3x3(nn.Module):
    def __init__(self, in_features, out_features=None, stride=1,
                 mlp_ratio=4, use_se=True, drop=0., drop_path=0.,
                 seq_l=196, head_dim=32, normalize={'type': 'sync_bn'}, **kwargs):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = int(in_features * mlp_ratio)
        if in_features != out_features or stride != 1:
            self.residual = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=stride, padding=1),
                nn.Conv2d(in_features, out_features, kernel_size=1)
            )
        else:
            self.residual = nn.Identity()

        self.b1 = None
        if in_features != hidden_features:
            layers_b1 = []
            _, bn = build_norm_layer(in_features, normalize)
            layers_b1.append(bn)
            layers_b1.append(nn.Conv2d(in_features, hidden_features, kernel_size=(3, 3),
                                       stride=stride, padding=(1, 1), bias=False))
            _, bn = build_norm_layer(hidden_features, normalize)
            layers_b1.append(bn)
            layers_b1.append(nn.GELU())
            self.b1 = nn.Sequential(*layers_b1)

        layers = []
        if use_se:
            layers.append(SqueezeExcitation(hidden_features))

        if in_features != hidden_features:
            kernel_size = 1
            padding = 0
            stride = 1
        else:
            kernel_size = 3
            padding = 1
            stride = stride
        layers.append(nn.Conv2d(hidden_features, out_features, kernel_size=kernel_size, padding=padding,
                                stride=stride))
        _, bn = build_norm_layer(out_features, normalize)
        layers.append(bn)
        self.b2 = nn.Sequential(*layers)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        zeros_(self.b2[-1].weight)

    def forward(self, x, h=14, w=14):
        # pdb.set_trace()
        h = int(h)
        w = int(w)
        residual = self.residual(x)
        if self.b1 is not None:
            x = self.b1(x)
        x = self.b2(x)

        return residual + self.drop_path(x)


class MBConv3x3(nn.Module):
    def __init__(self, in_features, out_features=None, stride=1,
                 mlp_ratio=4, use_se=True, drop=0., drop_path=0.,
                 seq_l=196, head_dim=32, normalize={'type': 'sync_bn'}, **kwargs):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = int(in_features * mlp_ratio)
        if in_features != out_features or stride != 1:
            self.residual = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=stride, padding=1),
                nn.Conv2d(in_features, out_features, kernel_size=1)
            )
        else:
            self.residual = nn.Identity()

        self.b1 = None
        if in_features != hidden_features or stride != 1:
            layers_b1 = []
            _, bn = build_norm_layer(in_features, normalize)
            layers_b1.append(bn)
            layers_b1.append(nn.Conv2d(in_features, hidden_features, kernel_size=(1, 1),
                                       stride=1, padding=(0, 0), bias=False))
            _, bn = build_norm_layer(hidden_features, normalize)
            layers_b1.append(bn)
            layers_b1.append(nn.GELU())
            self.b1 = nn.Sequential(*layers_b1)

        layers = []
        layers.append(nn.Conv2d(hidden_features, hidden_features, kernel_size=(3, 3), padding=(1, 1),
                                groups=hidden_features, stride=stride, bias=False))
        _, bn = build_norm_layer(hidden_features, normalize)
        layers.append(bn)
        layers.append(nn.GELU())
        if use_se:
            layers.append(SqueezeExcitation(hidden_features))

        layers.append(nn.Conv2d(hidden_features, out_features, kernel_size=(1, 1), padding=(0, 0)))
        _, bn = build_norm_layer(out_features, normalize)
        layers.append(bn)
        self.b2 = nn.Sequential(*layers)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        zeros_(self.b2[-1].weight)

    def forward(self, x, h=14, w=14):
        h = int(h)
        w = int(w)
        residual = self.residual(x)
        if self.b1 is not None:
            x = self.b1(x)
        x = self.b2(x)

        return residual + self.drop_path(x)


class Attention(nn.Module):
    def __init__(self, dim, out_dim=None, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 static=False, seq_l=196, window=False):
        super().__init__()
        out_dim = out_dim or dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.static = static
        if self.static:
            self.static_a = nn.Parameter(torch.Tensor(1, num_heads, seq_l, seq_l))
            trunc_normal_(self.static_a)
        self.custom_flops = 2 * seq_l * seq_l * dim
        self.window = window

    def forward(self, x, head=0, mask_type=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        if mask_type:
            mask = torch.ones_like(qkv)
            mask[:, :, head] = 0
            if mask_type == 'layer':
                mask = 1 - mask
            qkv = qkv * mask
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        q, k, v = q.float(), k.float(), v.float()

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        if self.static:
            attn = attn + self.static_a
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = x.type_as(qkv)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, in_features, out_features, stride=1, mlp_ratio=4, head_dim=32,
                 qkv_bias=True, qk_scale=None, drop=0., drop_path=0., attn_drop=0., seq_l=196,
                 conv_embedding=1, init_values=1e-6, window_attention=False, window_size=14, encoder_turbo=None, normalize=None): # noqa
        super().__init__()

        self.encoder_turbo = encoder_turbo
        self.stride = stride
        self.in_features = in_features
        self.out_features = out_features
        mlp_hidden_dim = int(in_features * mlp_ratio)
        num_heads = in_features // head_dim
        self.init_values = init_values
        self.conv_embedding = conv_embedding
        self.window_attention = window_attention
        self.window_size = window_size
        if conv_embedding == 1:
            self.pos_embed = nn.Conv2d(in_features, in_features, 3, padding=1, groups=in_features)
        elif conv_embedding == 2:
            self.pos_embed = nn.Linear(seq_l * stride ** 2, seq_l * stride ** 2)
        elif conv_embedding == 3:
            self.pos_embed = nn.MaxPool2d(3, 1, 1)
        else:
            self.pos_embed = None
        self.window_attention = window_attention

        if not self.use_encoder():
            self.attn = Attention(in_features, out_features, num_heads=num_heads, qkv_bias=qkv_bias,
                                  qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, seq_l=seq_l)
            self.norm1 = nn.LayerNorm(in_features)
            self.ds = nn.MaxPool2d(kernel_size=3, stride=stride, padding=1)
            self.residual = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=stride, padding=1),
                nn.Conv2d(in_features, out_features, kernel_size=1)
            )
            if init_values != -1:
                self.gamma_1 = nn.Parameter(init_values * torch.ones((out_features)), requires_grad=True)
        else:
            if self.encoder_turbo and self.encoder_turbo["enabled"]:
                self.init_encoder(in_features, num_heads, mlp_hidden_dim, qkv_bias, qk_scale, attn_drop, drop,
                                  drop_path, init_values, window_size if window_attention else 0)
            else:
                self.attn = Attention(in_features, out_features, num_heads=num_heads, qkv_bias=qkv_bias,
                                      qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, seq_l=seq_l)
                self.norm1 = nn.LayerNorm(in_features)
                self.norm2 = nn.LayerNorm(in_features)
                self.mlp = Mlp(in_features=in_features, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=drop)
                if init_values != -1:
                    self.gamma_1 = nn.Parameter(init_values * torch.ones((out_features)), requires_grad=True)
                    self.gamma_2 = nn.Parameter(init_values * torch.ones((out_features)), requires_grad=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def use_encoder(self):
        return self.stride == 1 and self.in_features == self.out_features

    def init_encoder(self, in_features, num_heads, mlp_hidden_dim, qkv_bias=True, qk_scale=None, attn_drop=0., ffn_drop=0., drop_path_ratio=0., gamma_init_values=-1, window_size=0): # noqa
        mb_config = {
            "atten_droppath_ratio": drop_path_ratio,
            "fnn_droppath_ratio": drop_path_ratio,
            "init_gamma_values": gamma_init_values,
            "window_size": window_size,
        }

        max_seq_len = self.encoder_turbo["max_seq_len"]
        max_batch_tokens = max_seq_len * self.encoder_turbo["batch_size"]
        config = LSTransformerEncoderLayer.get_config(
            max_batch_tokens=max_batch_tokens,
            max_seq_len=max_seq_len,
            hidden_size=in_features,
            intermediate_size=mlp_hidden_dim,
            nhead=num_heads,
            attn_prob_dropout_ratio=attn_drop,
            activation_dropout_ratio=attn_drop,
            hidden_dropout_ratio=ffn_drop,
            pre_layer_norm=True,
            fp16=self.encoder_turbo["fp16"],
            local_rank=int(os.environ['SLURM_PROCID']) % torch.cuda.device_count(),
            activation_fn="gelu",
            mb_config=mb_config,
        )

        self.encoder = LSTransformerEncoderLayer(config)
        self.dummy_mask = torch.Tensor()

    def forward(self, x, h=14, w=14):
        h = int(h)
        w = int(w)
        if self.conv_embedding == 1 or self.conv_embedding == 3:
            x = x + to3d(self.pos_embed(to4d(x, h, w)))
        elif self.conv_embedding == 2:
            x = x + self.pos_embed(x.transpose(1, 2)).transpose(1, 2)

        if self.use_encoder():
            if self.encoder_turbo and self.encoder_turbo["enabled"]:
                x = self.encoder(x, self.dummy_mask, h, w)
            else:
                res = x
                if self.window_attention:
                    x = self.norm1(x)
                    x = to4d(x, h, w)
                    x = x.permute(0, 2, 3, 1)
                    pad_l = pad_t = 0
                    pad_r = (self.window_size - w % self.window_size) % self.window_size
                    pad_b = (self.window_size - h % self.window_size) % self.window_size
                    x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
                    B, H, W, C = x.shape
                    x = x.reshape(B, H // self.window_size, self.window_size,
                                  W // self.window_size, self.window_size, C)
                    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B * H // self.window_size * W
                                                            // self.window_size, self.window_size * self.window_size, C)
                    x = self.gamma_1 * self.attn(x)
                    x = x.contiguous().view(-1, H // self.window_size, W // self.window_size, self.window_size, self.window_size, C) # noqa
                    x = x.permute(0, 1, 3, 2, 4, 5)  # (B, H//win, win, W//win, win, C)
                    B, Hp, p1, Wp, p2, C = x.shape
                    x = x.reshape(B, Hp * p1, Wp * p2, C)
                    if pad_r > 0 or pad_b > 0:
                        x = x[:, :h, :w, :].contiguous()
                    x = x.reshape(B, h * w, C)
                else:
                    x = self.norm1(x)
                    x = self.gamma_1 * self.attn(x)

                x = res + self.drop_path(x)
                if self.init_values != -1:
                    x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
                else:
                    x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            # x = self.residual(x) + self.drop_path(self.attn(self.norm1(x), head=head, mask_type=mask_type))
            residual = to3d(self.residual(to4d(x, h, w)))
            x = self.norm1(x)
            x = to3d(self.ds(to4d(x, h, w)))
            h, w = math.ceil(h / 2), math.ceil(w / 2)
            if self.window_attention:
                x = to4d(x, h, w)
                x = x.permute(0, 2, 3, 1)
                pad_l = pad_t = 0
                pad_r = (self.window_size - w % self.window_size) % self.window_size
                pad_b = (self.window_size - h % self.window_size) % self.window_size
                x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
                B, H, W, C = x.shape
                x = x.reshape(B, H // self.window_size, self.window_size, W // self.window_size, self.window_size, C)
                x = x.permute(0, 1, 3, 2, 4, 5).reshape(B * H // self.window_size * W
                                                        // self.window_size, self.window_size * self.window_size, C)
                bs, seql, hs = x.shape
                x = self.attn(x)
                C = x.shape[-1]
                x = x.contiguous().view(-1, H // self.window_size, W // self.window_size, self.window_size, self.window_size, C) # noqa
                x = x.permute(0, 1, 3, 2, 4, 5)  # (B, H//win, win, W//win, win, C)
                B, Hp, p1, Wp, p2, C = x.shape
                x = x.reshape(B, Hp * p1, Wp * p2, C)
                if pad_r > 0 or pad_b > 0:
                    x = x[:, :h, :w, :].contiguous()
                x = x.reshape(B, h * w, C)
            else:
                x = self.attn(x)
            if self.init_values != -1:
                x = residual + self.gamma_1 * x
            else:
                x = residual + x
        return x


class VisionTransformer(nn.Module):
    def __init__(self, repeats, expansion, channels, strides=[1, 2, 2, 2, 1, 2], num_classes=1000, drop_path_rate=0.2,
                 input_size=224, weight_init='', head_dim=48, final_drop=0.0, init_values=1e-6, conv_embedding=1,
                 block_ops=[MBConv3x3] * 3 + [Block] * 3, use_checkpoint=False, stem_dim=32, out_layers=[1, 2, 4, 5],
                 out_strides=[4, 8, 16, 32], normalize={'type': 'sync_bn'}, frozen_layers=[],
                 window_attention=False, window_size=[14, 7], window_stride=2, encoder_turbo=None, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.out_planes = channels[-5:]
        self.out_planes.pop(2)
        self.use_checkpoint = use_checkpoint
        self.repeats = repeats
        self.out_layers = out_layers
        self.out_strides = out_strides
        num_out = len(self.out_layers)
        self.out_strides = self.out_strides[-num_out:]
        self.out_planes = self.out_planes[-num_out:]
        self.window_attention = window_attention
        self.window_size = window_size
        self.frozen_layers = frozen_layers
        self.encoder_turbo = encoder_turbo
        _, bn = build_norm_layer(stem_dim, normalize)
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            bn,
            nn.GELU(),
        )

        # repeats =   [1, 2, 2, 3, 5, 8]
        # strides =   [1, 2, 2, 2, 1, 2]
        # expansion = [1, 6, 6, 4, 4, 4]
        # channels =  [16, 32, 48, 96, 128, 192]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(repeats))]  # stochastic depth decay rule
        dpr.reverse()
        self.scale = []

        cin = stem_dim
        blocks = []
        seq_l = (input_size // 2) ** 2
        window_size = window_size[0]
        self.conv_num = sum(repeats[:3])
        for stage in range(len(strides)):
            cout = channels[stage]
            # block_op = MBConv3x3 if stage < 3 else Block
            block_op = block_ops[stage]
            logger.info(f'stage {stage}, cin {cin}, cout {cout}, s {strides[stage]}, e {expansion[stage]} b {block_op}')
            seq_l = seq_l // (strides[stage] ** 2)
            for i in range(repeats[stage]):
                stride = strides[stage] if i == 0 else 1
                window = window_attention and (not (i + 1) % window_stride == 0)
                # window_size = window_size // stride
                if stride == 2:
                    window_size = self.window_size[1]
                blocks.append(block_op(cin, cout, stride=stride, mlp_ratio=expansion[stage],
                                       drop_path=dpr.pop(), seq_l=seq_l, head_dim=head_dim,
                                       init_values=init_values,
                                       conv_embedding=conv_embedding,
                                       window_attention=window,
                                       window_size=window_size,
                                       encoder_turbo=encoder_turbo,
                                       normalize=normalize))

                cin = cout
                self.scale.append(stride)
        self.blocks = nn.Sequential(*blocks)

        head_dim = 1280
        _, bn = build_norm_layer(head_dim, normalize)
        self.head = nn.Sequential(
            nn.Conv2d(cout, head_dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
            bn,
            nn.GELU(),
        )
        self.final_drop = nn.Dropout(final_drop) if final_drop > 0.0 else nn.Identity()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(head_dim, num_classes)

        head_bias = -math.log(self.num_classes) if 'nlhb' in weight_init else 0.
        # trunc_normal_(self.pos_embed, std=.02)
        # Weight init
        assert weight_init in ('jax', 'jax_nlhb', 'nlhb', '')
        if weight_init.startswith('jax'):
            # leave cls token as zeros to match jax impl
            for n, m in self.named_modules():
                _init_vit_weights(m, n, head_bias=head_bias, jax_impl=True)
        else:
            # trunc_normal_(self.cls_token, std=.02)
            self.apply(_init_vit_weights)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)

    @torch.jit.ignore
    def no_weight_decay(self):
        # return {'pos_embed', 'cls_token', 'dist_token'}
        return {'pos_embed', 'dist_token'}

    def get_outplanes(self):
        """
        get dimension of the output tensor
        """
        return self.out_planes

    def get_outstrides(self):
        return self.out_strides

    def opt_checkpoint(self, i):
        if self.encoder_turbo and self.encoder_turbo["fp16"] and self.encoder_turbo["batch_size"] == 1:
            if i > 50:
                return True
            else:
                return False

        return True

    def forward(self, input):
        # [2, 3, 6, 6, 6, 12]
        # out_index = [sum(repeats[:i+1])-1 for i in range(len(repeats))]
        x = input['image']
        x = self.stem(x)
        out = []
        b, c, h, w = x.shape
        for i, blk in enumerate(self.blocks):
            # if i == self.conv_num:
            if isinstance(blk, MBConv3x3) or isinstance(blk, FusedMBConv3x3):
                x = to4d(x, h, w)
                b, c, h, w = x.shape
            if isinstance(blk, Block) or isinstance(blk, MixerBlock):
                x = to3d(x)
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, *(x, Variable(torch.Tensor([h])), Variable(torch.Tensor([w]))))
            else:
                x = blk(x, h, w)
            h = math.ceil(h / self.scale[i])
            w = math.ceil(w / self.scale[i])
            if i in self.out_layers:
                out.append(to4d(x, h, w))

        features = out
        return {'features': features, 'strides': self.get_outstrides()}

    def freeze_layer(self):
        s1 = sum(self.repeats[:2])
        s2 = sum(self.repeats[:3])
        s3 = sum(self.repeats[:5])
        s4 = sum(self.repeats)
        layers = [
            self.stem, self.blocks[0:s1], self.blocks[s1:s2], self.blocks[s2:s3], self.blocks[s3:s4]
        ]
        for layer_idx in self.frozen_layers:
            layer = layers[layer_idx]
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        """
        Sets the module in training mode.
        This has any effect only on modules such as Dropout or BatchNorm.

        Returns:
            Module: self
        """
        self.training = mode
        for module in self.children():
            module.train(mode)
        self.freeze_layer()
        return self


def _init_vit_weights(m, n: str = '', head_bias: float = 0., jax_impl: bool = False):
    if isinstance(m, nn.Linear):
        if n.startswith('head'):
            nn.init.zeros_(m.weight)
            nn.init.constant_(m.bias, head_bias)
        elif n.startswith('pre_logits'):
            lecun_normal_(m.weight)
            nn.init.zeros_(m.bias)
        else:
            if jax_impl:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    if 'mlp' in n:
                        nn.init.normal_(m.bias, std=1e-6)
                    else:
                        nn.init.zeros_(m.bias)
            else:
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    elif jax_impl and isinstance(m, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


@MODULE_ZOO_REGISTRY.register("mb4")
def MB4(**kwargs):
    repeats = [2, 3, 6, 6, 6, 12]
    expansion = [1, 4, 6, 3, 2, 5]
    channels = [32, 64, 128, 192, 192, 384]
    final_drop = 0.0
    block_ops = [FusedMBConv3x3] * 2 + [MBConv3x3] * 2 + [Block] * 2

    logger.info(f'channels {channels}, repeats {repeats}, expansion {expansion}')
    model_kwargs = dict(repeats=repeats, expansion=expansion, channels=channels,
                        block_ops=block_ops, final_drop=final_drop, input_size=256, **kwargs)
    model = VisionTransformer(**model_kwargs)
    return model


@MODULE_ZOO_REGISTRY.register("mb4_gvm")
def MB4_gvm(**kwargs):
    repeats = [2, 3, 6, 6, 6, 12]
    expansion = [1, 4, 6, 3, 2, 5]
    channels = [32, 64, 128, 192, 192, 384]
    final_drop = 0.0
    block_ops = [MBConv3x3] * 4 + [Block] * 2

    logger.info(f'channels {channels}, repeats {repeats}, expansion {expansion}')
    model_kwargs = dict(repeats=repeats, expansion=expansion, channels=channels,
                        block_ops=block_ops, final_drop=final_drop, input_size=256, **kwargs)
    model = VisionTransformer(**model_kwargs)
    return model


@MODULE_ZOO_REGISTRY.register("mb7")
def MB7(**kwargs):
    repeats = [2, 4, 8, 8, 8, 16]
    expansion = [1, 4, 6, 3, 2, 5]
    channels = [48, 96, 160, 256, 256, 512]
    final_drop = 0.0
    block_ops = [MBConv3x3] * 4 + [Block] * 2

    logger.info(f'channels {channels}, repeats {repeats}, expansion {expansion}')
    model_kwargs = dict(repeats=repeats, expansion=expansion, channels=channels,
                        block_ops=block_ops, final_drop=final_drop, input_size=256, head_dim=32, num_classes=21842, **kwargs) # noqa
    model = VisionTransformer(**model_kwargs)
    return model


@MODULE_ZOO_REGISTRY.register("mb15")
def MB15(**kwargs):  # width 1.3, depth 1.6 # 700M, 9M
    repeats = [6, 9, 18, 18, 18, 36]
    expansion = [1, 4, 6, 3, 2, 5]
    # channels = [128, 264, 448, 704, 704, 1408]
    channels = [104, 216, 384, 576, 576, 1152]
    final_drop = 0.0
    block_ops = [FusedMBConv3x3] * 4 + [Block] * 2
    logger.info(f'channels {channels}, repeats {repeats}, expansion {expansion}')
    model_kwargs = dict(repeats=repeats, expansion=expansion, channels=channels,
                        block_ops=block_ops, final_drop=final_drop,
                        stem_dim=64, head_dim=64,
                        num_classes=1000, **kwargs)
    model = VisionTransformer(**model_kwargs)
    return model


def transform_position_torch_ls(layer_name, is_weight):
    # key_ls = "module.blocks." + str(layer_num) +".encoder.para"
    index = 0
    need_reshape = False
    if layer_name == "norm1":
        if is_weight:
            index = 4
        else:
            index = 5
    elif layer_name == "norm2":
        if is_weight:
            index = 10
        else:
            index = 11
    elif layer_name == "fc1":
        if is_weight:
            index = 6
            need_reshape = True
        else:
            index = 7
    elif layer_name == "fc2":
        if is_weight:
            index = 8
            need_reshape = True
        else:
            index = 9
    elif layer_name == "qkv":
        if is_weight:
            index = 0
            need_reshape = True
        else:
            index = 1
    elif layer_name == "proj":
        if is_weight:
            index = 2
            need_reshape = True
        else:
            index = 3
    return index, need_reshape


def cat_tensor(tensor_list):
    return torch.cat((tensor_list[0],
                      tensor_list[1],
                      tensor_list[2],
                      tensor_list[3],
                      tensor_list[4],
                      tensor_list[5],
                      tensor_list[6],
                      tensor_list[7],
                      tensor_list[8],
                      tensor_list[9],
                      tensor_list[10],
                      tensor_list[11],
                      tensor_list[12],
                      tensor_list[13]), 0)


def transform_origin2ls_ckpt(state_dict, start_layer, skip_layer, end_layer):
    # index_of_layernum = ?
    index_of_paraname = -2
    index_of_wb = -1
    no_prefix_blocks = False
    if "blocks." + str(start_layer) + ".pos_embed.weight" in state_dict:
        no_prefix_blocks = True
    state_dict_ls = {}
    for i in range(start_layer, end_layer + 1):
        if i != skip_layer:
            if no_prefix_blocks:
                # initial an empty list to store the tensor, total 14
                state_dict_ls["blocks." + str(i) + ".encoder.para"] = [1] * 14
                index_of_layernum = 1
            else:
                state_dict_ls["module.blocks." + str(i) + ".encoder.para"] = [1] * 14
                index_of_layernum = 2
    for key in state_dict:
        if "blocks" in key:
            # split the info of layers
            args_para = key.split(".")
            layer_num = int(args_para[index_of_layernum])
            layer_name = args_para[index_of_paraname]
            # skip_layer 层不做workspace merge
            if layer_num >= start_layer and layer_num != skip_layer:
                if no_prefix_blocks:
                    key_ls = "blocks." + str(layer_num) + ".encoder.para"
                else:
                    key_ls = "module.blocks." + str(layer_num) + ".encoder.para"
                if layer_name != "pos_embed" and args_para[-1] != "gamma_1" and args_para[-1] != "gamma_2":
                    # gamma12 index 特殊 没有weight 和bias 所以名字是最后一项 别的都是倒数第二项 （qkv和proj前面又有attn，所以只能用倒数区分）
                    # do the merge into enco layer in right position
                    is_weight = (args_para[index_of_wb] == "weight")

                    # key_ls = "module.blocks." + str(layer_num) +".encoder.para"

                    index, need_reshape = transform_position_torch_ls(layer_name, is_weight)
                    # value need to be input into list
                    value_input = state_dict[key]
                    if need_reshape:
                        value_input = torch.reshape(value_input, (-1,))
                    state_dict_ls[key_ls][index] = value_input

                elif args_para[-1] == "gamma_1":
                    # add gamma1 and gamm2
                    # gamma1和gamma2已经加入enc
                    state_dict_ls[key_ls][12] = state_dict[key]
                elif args_para[-1] == "gamma_2":
                    state_dict_ls[key_ls][13] = state_dict[key]
                elif layer_name == "pos_embed":
                    # add pos_embed
                    state_dict_ls[key] = state_dict[key]
                if all(type(tensor_sub) == torch.Tensor for tensor_sub in state_dict_ls[key_ls]) and len(state_dict_ls[key_ls]) == 14: # noqa
                    # cat之后也符合每一元素都是tensor 所以要限定长度是14个tensor的时候做cat
                    state_dict_ls[key_ls] = cat_tensor(state_dict_ls[key_ls])
            else:
                # add other layers in stage 1,2,3 and layer 69 at the beginning of stage 5
                state_dict_ls[key] = state_dict[key]
        else:
            # add other layer not start with the name "module.blocks"
            state_dict_ls[key] = state_dict[key]
    return state_dict_ls


def split_tensor(tensor_value, hs, stage):
    size = []
    # stage with 4 and 5 need to apply two different size list
    if stage == 4:
        size = [3 * hs * hs,  # qkv w
                3 * hs,  # qkv b
                hs * hs,  # proj w
                hs,  # proj b
                hs,  # norm 1 w
                hs,  # norm 1 b
                2 * hs * hs,  # fc1 w
                2 * hs,  # fc1 b
                2 * hs * hs,  # fc2 w
                hs,  # fc2 b
                hs,  # norm2 w
                hs,  # norm2 b
                hs,  # gamma_1
                hs]  # gamma_2
    elif stage == 5:
        size = [3 * hs * hs,  # qkv w
                3 * hs,  # qkv b
                hs * hs,  # proj w
                hs,  # proj b
                hs,  # norm 1 w
                hs,  # norm 1 b
                5 * hs * hs,  # fc1 w
                5 * hs,  # fc1 b
                5 * hs * hs,  # fc2 w
                hs,  # fc2 b
                hs,  # norm2 w
                hs,  # norm2 b
                hs,  # gamma_1
                hs]  # gamma_2
    qkv_w, qkv_b, proj_w, proj_b, norm1_w, norm1_b,\
        fc1_w, fc1_b, fc2_w, fc2_b, norm2_w, norm2_b, gamma_1, gamma_2 = torch.split(tensor_value, size, 0)
    # rebuild into dimension 2
    qkv_w = qkv_w.view(3 * hs, hs)
    proj_w = proj_w.view(hs, hs)
    if stage == 4:
        fc1_w = fc1_w.view(2 * hs, hs)
        fc2_w = fc2_w.view(hs, 2 * hs)
    elif stage == 5:
        fc1_w = fc1_w.view(5 * hs, hs)
        fc2_w = fc2_w.view(hs, 5 * hs)
    return [(".attn.qkv.weight", qkv_w), (".attn.qkv.bias", qkv_b),
            (".attn.proj.weight", proj_w), (".attn.proj.bias", proj_b),
            (".norm1.weight", norm1_w), (".norm1.bias", norm1_b),
            (".mlp.fc1.weight", fc1_w), (".mlp.fc1.bias", fc1_b),
            (".mlp.fc2.weight", fc2_w), (".mlp.fc2.bias", fc2_b),
            (".norm2.weight", norm2_w), (".norm2.bias", norm2_b),
            (".gamma_1", gamma_1), (".gamma_2", gamma_2)]


def transform_ls2origin_ckpt(state_dict_ls, skip_layer, hs_stage_4, hs_stage_5):
    # transform back
    no_prefix_blocks = False
    if "blocks." + str(skip_layer) + ".pos_embed.weight" in state_dict_ls:
        no_prefix_blocks = True
        index_of_layer = 1
    else:
        index_of_layer = 2
    state_dict_new = {}
    for key in state_dict_ls:
        if "encoder.para" not in key:
            state_dict_new[key] = state_dict_ls[key]
        else:
            # repeats4层和repeats5层（去除 skip_layer 层）的hidden size不一样，并且部分layer的size也不相同，为了方便，分开拆
            args_info = key.split(".")
            layer_num = args_info[index_of_layer]
            grads = []
            hidden_size = 0
            stage = 4
            if int(layer_num) < skip_layer:
                hidden_size = hs_stage_4
                stage = 4

            elif int(layer_num) > skip_layer:
                hidden_size = hs_stage_5
                stage = 5

            grads = split_tensor(state_dict_ls[key], hidden_size, stage)

            for i in range(len(grads)):
                if no_prefix_blocks:
                    key_new = "blocks." + layer_num + grads[i][0]
                else:
                    key_new = "module.blocks." + layer_num + grads[i][0]
                state_dict_new[key_new] = grads[i][1]

    return state_dict_new
