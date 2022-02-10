# Import from third library
import torch.nn as nn
from torch.nn import functional as F

# Import from eod
from eod.utils.general.registry_factory import MODULE_ZOO_REGISTRY
from eod.utils.model.initializer import initialize_from_cfg
from eod.utils.model.normalize import build_conv_norm


__all__ = ['CondinstHead']


def aligned_bilinear(tensor, factor):
    assert tensor.dim() == 4
    assert factor >= 1
    assert int(factor) == factor

    if factor == 1:
        return tensor

    h, w = tensor.size()[2:]
    tensor = F.pad(tensor, pad=(0, 1, 0, 1), mode="replicate")
    oh = factor * h + 1
    ow = factor * w + 1
    tensor = F.interpolate(
        tensor, size=(oh, ow),
        mode='bilinear',
        align_corners=True
    )
    tensor = F.pad(
        tensor, pad=(factor // 2, 0, factor // 2, 0),
        mode="replicate"
    )

    return tensor[:, :, :oh - 1, :ow - 1]


@MODULE_ZOO_REGISTRY.register('CondinstHead')
class CondinstHead(nn.Module):
    def __init__(
            self,
            inplanes,
            feat_planes,
            dense_points,
            mask_gen_params,
            feat_layers,
            branch_channels,
            branch_num_convs,
            branch_num_outputs,
            normalize=None,
            initializer=None):
        super(CondinstHead, self).__init__()
        self.controller_pred = nn.Conv2d(feat_planes, dense_points * mask_gen_params,
                                         kernel_size=3, stride=1, padding=1)
        self.mask_branch = MaskBranch(inplanes, feat_layers, branch_num_convs, branch_channels,
                                      branch_num_outputs, normalize, initializer)

    def forward(self, input):
        features = input['features']
        box_feature = input['box_feature']
        mlvl_controller = [self.controller_pred(x) for x in box_feature]
        mask_feats = self.mask_branch(features)
        return {'mlvl_controller': mlvl_controller, 'mask_feats': mask_feats}


class MaskBranch(nn.Module):
    def __init__(self, inplanes, feat_layers, num_convs, channels, num_outputs, normalize=None, initializer=None):
        super().__init__()
        self.feat_layers = feat_layers
        self.num_outputs = num_outputs
        if not isinstance(inplanes, list):
            inplanes = [inplanes] * feat_layers
        self.refine = nn.Sequential()
        for i in range(feat_layers):
            self.refine.add_module(
                f'{i}',
                build_conv_norm(
                    inplanes[i], channels, kernel_size=3, stride=1, padding=1, normalize=normalize, activation=True))

        self.tower = nn.Sequential()
        for i in range(num_convs):
            self.tower.add_module(
                f'{i}',
                build_conv_norm(
                    channels, channels, kernel_size=3, stride=1, padding=1, normalize=normalize, activation=True))
        self.tower.add_module(
            f'{num_convs}', nn.Conv2d(
                channels, max(
                    num_outputs, 1), kernel_size=3, stride=1, padding=1))
        initialize_from_cfg(self, initializer)

    def forward(self, features):
        for i in range(self.feat_layers):
            if i == 0:
                x = self.refine[i](features[i])
            else:
                x_p = self.refine[i](features[i])
                target_h, target_w = x.size()[2:]
                h, w = x_p.size()[2:]
                assert target_h % h == 0
                assert target_w % w == 0
                factor_h, factor_w = target_h // h, target_w // w
                assert factor_h == factor_w
                x_p = aligned_bilinear(x_p, factor_h)
                x = x + x_p
        mask_feats = self.tower(x)

        if self.num_outputs == 0:
            mask_feats = mask_feats[:, :self.num_outputs]

        return mask_feats


def build_condinst_mask_branch(cfg):
    return MaskBranch(cfg)
