import math
import torch
import torch.nn as nn
from typing import Sequence
from collections import OrderedDict
from up.utils.model.utils import DropPath
from up.utils.model.initializer import trunc_normal_


__all__ = ['VisionTransformer',
           'vit_base_patch32_224',
           'vit_base_patch16_224',
           'vit_large_patch16_224',
           'vit_huge_patch14_224',
           'deit_tiny_patch16_224',
           'deit_small_patch16_224',
           'deit_base_patch16_224']


class GELU(nn.Module):
    """
    Gaussian Error Linear Units, based on
    `"Gaussian Error Linear Units (GELUs)" <https://arxiv.org/abs/1606.08415>`
    """

    def __init__(self, approximate=True):
        super(GELU, self).__init__()
        self.approximate = approximate

    def forward(self, x):
        if self.approximate:
            cdf = 0.5 * (1.0 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
            return x * cdf
        else:
            return x * (torch.erf(x / math.sqrt(2)) + 1) / 2


class FeedForward(nn.Module):

    def __init__(self, dim, hidden_dim, dropout=0.1, activation=GELU):
        super(FeedForward, self).__init__()

        self.mlp1 = nn.Linear(dim, hidden_dim)
        self.act = activation()
        self.mlp2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.mlp1(x)
        x = self.act(x)
        x = self.dropout(x)

        x = self.mlp2(x)
        x = self.dropout(x)
        return x


class MultiHeadAttention(nn.Module):

    def __init__(self,
                 dim,
                 heads=8,
                 dropout=0.1,
                 attention_dropout=0.1,
                 qkv_bias=True,):
        super(MultiHeadAttention, self).__init__()
        assert dim % heads == 0
        self.heads = heads
        self.scale = (dim // heads) ** -0.5  # 1/ sqrt(d_k)

        self.to_qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.to_out = nn.Linear(dim, dim, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.attention_dropout = nn.Dropout(attention_dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.to_qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attention_dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.to_out(x)
        x = self.dropout(x)
        return x


class Encoder1DBlock(nn.Module):
    def __init__(self, hidden_dim, mlp_dim, heads,
                 dropout, attention_dropout, drop_path, qkv_bias, activation, norm_layer=nn.LayerNorm):
        super(Encoder1DBlock, self).__init__()
        self.norm1 = norm_layer(hidden_dim)
        self.attention = MultiHeadAttention(hidden_dim, heads, dropout, attention_dropout, qkv_bias)
        self.norm2 = norm_layer(hidden_dim)
        self.feedforward = FeedForward(hidden_dim, mlp_dim, dropout, activation=activation)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else None

    def forward(self, x):

        residual = x
        x = self.norm1(x)
        x = self.attention(x)
        if self.drop_path is not None:
            x = self.drop_path(x)
        x = x + residual

        y = self.norm2(x)
        y = self.feedforward(y)
        if self.drop_path is not None:
            y = self.drop_path(y)

        return x + y


class Encoder(nn.Module):
    def __init__(self,
                 hidden_dim,
                 depth,
                 mlp_dim,
                 heads,
                 dropout=0.1,
                 attention_dropout=0.1,
                 drop_path=0.1,
                 qkv_bias=True,
                 activation=GELU):
        super(Encoder, self).__init__()

        encoder_layer = OrderedDict()
        for d in range(depth):
            encoder_layer['encoder_{}'.format(d)] = \
                Encoder1DBlock(hidden_dim, mlp_dim, heads, dropout, attention_dropout, drop_path, qkv_bias, activation)
        self.encoders = nn.Sequential(encoder_layer)
        self.encoder_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = self.encoders(x)
        x = self.encoder_norm(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(self,
                 image_size,
                 patch_size,
                 patch_stride,
                 num_classes,
                 hidden_dim,
                 depth,
                 mlp_dim,
                 in_channel=3,
                 heads=8,
                 dropout=0.1,
                 attention_dropout=0.1,
                 classifier='token',
                 drop_path=0.1,
                 activation='gelu',
                 qkv_bias=True,
                 out_indices=-1,
                 last_norm=True):
        r"""
        Arguments:

        - in_channel (:obj:`int`): input channels
        - image_size (:obj:`int`): image size
        - patch_size (:obj:`int`): size of a patch
        - patch_stride (:obj:`int`): number of patch stride
        - num_classes (:obj:`int`): number of classes
        - hidden_dim (:obj:`int`): embedding dimension of tokens
        - depth (:obj:`int`): number of encoder blocks
        - mlp_dim (:obj:`int`): hidden dimension in feedforward layer
        - heads (:obj:`int`): number of heads in multihead-attention
        - dropout (:obj:`float`): dropout rate after linear
        - drop_path (:obj:`float`): droppath rate after attention and feedforward
        - attention_dropout (:obj:`float`): dropout rate after softmax in Acaled Dot-Product Attention
        - classifier (:obj:`str`): classifier type
        - representation_size: if not ``None``, add extra linear/tanh after representation
        - activation (:obj:`Moudle`): ReLU or GELU
        - qkv_bias (:obj:`bool`): whether assign biases for qkv linear
        """

        super(VisionTransformer, self).__init__()
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        num_patches = (image_size // patch_size) ** 2

        self.patch_size = patch_size
        self.num_patches = num_patches
        self.num_classes = num_classes
        self.num_layers = depth
        self.classifier = classifier
        self.embedding = nn.Conv2d(in_channel, hidden_dim, kernel_size=patch_size, stride=patch_size, padding=0)
        self.last_norm = last_norm

        if classifier == 'token':
            self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, hidden_dim))
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            # trunc_normal_(self.cls_token, std=0.02)
        elif classifier == 'gap':
            self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, hidden_dim))
        # trunc_normal_(self.pos_embedding, std=0.02)

        if activation == 'gelu':
            activation = GELU
        elif activation == 'relu':
            activation = nn.ReLU
        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = self.num_layers + index
                assert out_indices[i] >= 0, f'Invalid out_indices {index}'
        self.out_indices = out_indices
        self.layers = OrderedDict()
        for i_layer in range(self.num_layers):
            self.layers['encoder_{}'.format(i_layer)] = \
                Encoder1DBlock(hidden_dim, mlp_dim, heads, dropout, attention_dropout, drop_path, qkv_bias, activation)
        self.layers = nn.Sequential(self.layers)
        # self.transformer = Encoder(
        #     hidden_dim, depth, mlp_dim, heads, dropout, attention_dropout, drop_path, qkv_bias, activation)
        self.encoder_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def get_param_no_wd(self, fc=False, norm=False):
        no_wd = [self.pos_embedding, self.cls_token]
        no_wd.extend([self.embedding.weight, self.embedding.bias])
        for m in self.modules():
            if fc and isinstance(m, nn.Linear) and m.bias is not None:
                no_wd.append(m.bias)
            elif norm and isinstance(m, nn.LayerNorm):
                no_wd.append(m.bias)
                no_wd.append(m.weight)
        return no_wd

    def forward(self, input):
        img = input['image']
        x = self.embedding(img)
        patch_resolution = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)
        # x shape: [B, N, K]

        if self.classifier == 'token':
            cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            # x shape: [B, N+1, K]

        x += self.pos_embedding
        x = self.dropout(x)

        # transformer main body
        outs = []
        out_strides = []
        for i, layer in enumerate(self.layers):
            x = layer(x)

            if i == len(self.layers) - 1 and self.encoder_norm and self.last_norm:
                x = self.encoder_norm(x)

            if i in self.out_indices:
                B, _, C = x.shape
                patch_token = x[:, 1:].reshape(B, *patch_resolution, C)
                patch_token = patch_token.permute(0, 3, 1, 2)
                cls_token = x[:, 0]
                if self.classifier == 'token':
                    out = [patch_token, cls_token]
                else:
                    out = patch_token
                outs.append(out)
                out_strides.append(self.patch_size)

        return {'features': outs, 'strides': out_strides}

        # x = self.transformer(x)

        # # type of classifier
        # if self.classifier == 'token':
        #     x = x[:, 0]
        # elif self.classifier == 'gap':
        #     x = torch.mean(x, dim=1, keepdim=False)

        # if self.representation_size is None:
        #     # finetune on new task
        #     return self.fc(x)
        # else:
        #     # train from scratch
        #     return self.fc(self.tanh(self.pre_logits(x)))


def vit_base_patch32_224(**kwargs):
    default_kwargs = {
        'image_size': 224,
        'patch_size': 32,
        'patch_stride': 32,
        'num_classes': 1000,
        'hidden_dim': 768,
        'depth': 12,
        'mlp_dim': 3072,
        'heads': 12,
        'in_channel': 3,
        'dropout': 0.1,
        'attention_dropout': 0.1,
        'classifier': 'token',
    }
    default_kwargs.update(kwargs)
    vit = VisionTransformer(**default_kwargs)
    return vit


def vit_base_patch16_224(pretrained=False, **kwargs):
    default_kwargs = {
        'image_size': 224,
        'patch_size': 16,
        'patch_stride': 16,
        'num_classes': 1000,
        'hidden_dim': 768,
        'depth': 12,
        'mlp_dim': 3072,
        'heads': 12,
        'in_channel': 3,
        'dropout': 0.1,
        'attention_dropout': 0.1,
        'classifier': 'token'
    }
    default_kwargs.update(kwargs)
    vit = VisionTransformer(**default_kwargs)
    return vit


def vit_large_patch16_224(pretrained=False, **kwargs):
    default_kwargs = {
        'image_size': 224,
        'patch_size': 16,
        'patch_stride': 16,
        'num_classes': 1000,
        'hidden_dim': 1024,
        'depth': 24,
        'mlp_dim': 4096,
        'heads': 16,
        'in_channel': 3,
        'dropout': 0.1,
        'attention_dropout': 0.1,
        'classifier': 'token'
    }
    default_kwargs.update(kwargs)
    vit = VisionTransformer(**default_kwargs)
    return vit


def vit_huge_patch14_224(**kwargs):
    default_kwargs = {
        'image_size': 224,
        'patch_size': 14,
        'patch_stride': 14,
        'num_classes': 1000,
        'hidden_dim': 1280,
        'depth': 32,
        'mlp_dim': 5120,
        'heads': 16,
        'in_channel': 3,
        'dropout': 0.1,
        'attention_dropout': 0.1,
        'classifier': 'token'
    }
    default_kwargs.update(kwargs)
    vit = VisionTransformer(**default_kwargs)
    return vit


def deit_tiny_patch16_224(pretrained=False, **kwargs):
    default_kwargs = {
        'image_size': 224,
        'patch_size': 16,
        'patch_stride': 16,
        'num_classes': 1000,
        'hidden_dim': 192,
        'depth': 12,
        'mlp_dim': 768,
        'heads': 3,
        'in_channel': 3,
        'dropout': 0.1,
        'attention_dropout': 0.1,
        'classifier': 'token'
    }
    default_kwargs.update(kwargs)
    vit = VisionTransformer(**default_kwargs)
    return vit


def deit_small_patch16_224(pretrained=False, **kwargs):
    default_kwargs = {
        'image_size': 224,
        'patch_size': 16,
        'patch_stride': 16,
        'num_classes': 1000,
        'hidden_dim': 384,
        'depth': 12,
        'mlp_dim': 1536,
        'heads': 6,
        'in_channel': 3,
        'dropout': 0.1,
        'attention_dropout': 0.1,
        'classifier': 'token'
    }
    default_kwargs.update(kwargs)
    vit = VisionTransformer(**default_kwargs)
    return vit


def deit_base_patch16_224(pretrained=False, **kwargs):
    """
    the same as vit_base_patch16_224
    """
    default_kwargs = {
        'image_size': 224,
        'patch_size': 16,
        'patch_stride': 16,
        'num_classes': 1000,
        'hidden_dim': 768,
        'depth': 12,
        'mlp_dim': 3072,
        'heads': 12,
        'in_channel': 3,
        'dropout': 0.1,
        'attention_dropout': 0.1,
        'classifier': 'token'
    }
    default_kwargs.update(kwargs)
    vit = VisionTransformer(**default_kwargs)
    return vit
