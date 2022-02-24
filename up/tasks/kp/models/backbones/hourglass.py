import torch.nn as nn
import torch.nn.functional as F
from up.utils.general.registry_factory import MODULE_ZOO_REGISTRY
from up.utils.model.normalize import build_norm_layer

__all__ = ['HourglassNet', 'hourglass']


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, normalize={'type': 'solo_bn'}):
        super(Bottleneck, self).__init__()

        self.norm1_name, norm1 = build_norm_layer(inplanes, normalize, 1)
        self.add_module(self.norm1_name, norm1)
        self.norm2_name, norm2 = build_norm_layer(planes, normalize, 2)
        self.add_module(self.norm2_name, norm2)
        self.norm3_name, norm3 = build_norm_layer(planes, normalize, 3)
        self.add_module(self.norm3_name, norm3)

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        return getattr(self, self.norm3_name)

    def forward(self, x):
        residual = x

        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.norm3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


@MODULE_ZOO_REGISTRY.register('keypoint_hg')
class Hourglass(nn.Module):
    def __init__(self, block, num_blocks, planes, depth, normalize={'type': 'solo_bn'}):
        super(Hourglass, self).__init__()
        self.depth = depth
        self.block = block
        self.normalize = normalize
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.hg = self._make_hour_glass(block, num_blocks, planes, depth)

    def _make_residual(self, block, num_blocks, planes):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(planes * block.expansion, planes, normalize=self.normalize))
        return nn.Sequential(*layers)

    def _make_hour_glass(self, block, num_blocks, planes, depth):
        tmp_hg = []
        for i in range(depth):
            res = []
            for j in range(3):
                res.append(self._make_residual(block, num_blocks, planes))
            if i == 0:
                res.append(self._make_residual(block, num_blocks, planes))
            tmp_hg.append(nn.ModuleList(res))
        return nn.ModuleList(tmp_hg)

    def _hour_glass_forward(self, n, x):
        up1 = self.hg[n - 1][0](x)
        low1 = F.max_pool2d(x, 2, stride=2)
        low1 = self.hg[n - 1][1](low1)

        if n > 1:
            low2 = self._hour_glass_forward(n - 1, low1)
        else:
            low2 = self.hg[n - 1][3](low1)

        low3 = self.hg[n - 1][2](low2)
        up2 = self.upsample(low3)
        out = up1 + up2
        return out

    def forward(self, x):
        return self._hour_glass_forward(self.depth, x)


@MODULE_ZOO_REGISTRY.register('keypoint_hgnet')
class HourglassNet(nn.Module):
    def __init__(self, block, num_stacks=2, num_blocks=4, num_classes=16, has_bg=False, normalize={'type': 'solo_bn'}):

        super(HourglassNet, self).__init__()

        self.inplanes = 64
        self.num_feats = 128
        self.num_stacks = num_stacks
        self.num_classes = num_classes + int(has_bg)
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=True)
        self.norm1_name, norm1 = build_norm_layer(self.inplanes, normalize, 1)
        self.normalize = normalize
        self.add_module(self.norm1_name, norm1)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_residual(block, self.inplanes, 1)
        self.layer2 = self._make_residual(block, self.inplanes, 1)
        self.layer3 = self._make_residual(block, self.num_feats, 1)
        self.maxpool = nn.MaxPool2d(2, stride=2)

        # build hourglass modules
        ch = self.num_feats * block.expansion
        hg, res, fc, score, fc_, score_ = [], [], [], [], [], []
        for i in range(num_stacks):
            hg.append(Hourglass(block, num_blocks, self.num_feats, 4, normalize=normalize))
            res.append(self._make_residual(block, self.num_feats, num_blocks))
            fc.append(self._make_fc(ch, ch, i + 2))
            score.append(nn.Conv2d(ch, self.num_classes, kernel_size=1, bias=True))
            if i < num_stacks - 1:
                fc_.append(nn.Conv2d(ch, ch, kernel_size=1, bias=True))
                score_.append(
                    nn.Conv2d(
                        self.num_classes,
                        ch,
                        kernel_size=1,
                        bias=True))
        self.hg = nn.ModuleList(hg)
        self.res = nn.ModuleList(res)
        self.fc = nn.ModuleList(fc)
        self.score = nn.ModuleList(score)
        self.fc_ = nn.ModuleList(fc_)
        self.score_ = nn.ModuleList(score_)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def _make_residual(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=True),
            )

        layers = list()
        layers.append(block(self.inplanes, planes, stride, downsample, normalize=self.normalize))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, normalize=self.normalize))

        return nn.Sequential(*layers)

    def _make_fc(self, inplanes, outplanes, idx=0):
        norm_name, norm = build_norm_layer(self.inplanes, self.normalize, idx)
        self.add_module(norm_name, norm)
        conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=True)
        return nn.Sequential(
            conv,
            norm,
            self.relu,
        )

    def forward(self, x):
        if isinstance(x, dict):
            x = x['image']
        out = []
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.layer3(x)

        for i in range(self.num_stacks):
            y = self.hg[i](x)
            y = self.res[i](y)
            y = self.fc[i](y)
            score = self.score[i](y)
            out.append(score)
            if i < self.num_stacks - 1:
                fc_ = self.fc_[i](y)
                score_ = self.score_[i](score)
                x = x + fc_ + score_
        return {"features": out}


@MODULE_ZOO_REGISTRY.register("hourglass")
def hourglass(**kwargs):
    model = HourglassNet(Bottleneck, **kwargs)
    return model
