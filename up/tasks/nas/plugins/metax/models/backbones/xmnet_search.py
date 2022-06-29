import numpy as np
import collections
import torch
import torch.nn as nn
from .xmnet import MBConv, FusedConv
from up.utils.model.normalize import build_conv_norm
from up.utils.general.log_helper import default_logger as logger
from up.utils.model.initializer import initialize_from_cfg

try:
    from .ssds import ssd_entry
except BaseException:
    from ssds import ssd_entry

__all__ = ['xmnet_search']


BlockArgs = collections.namedtuple('BlockArgs', [
    'block', 'kernel_size', 'channel', 'repeat', 'expansion'
])


def get_block(name):
    if name == 'fuseconv':
        return FusedConv
    elif name == 'mbconv':
        return MBConv


class XMNetSearch(nn.Module):
    def __init__(self,
                 in_channels=3,
                 width_mult=1.0,
                 round_nearest=8,
                 input_size=224,
                 input_channel=32,
                 out_layers=[4],
                 out_stride=[16],
                 normalize={'type': 'solo_bn'},
                 initializer=None,
                 cells=None,
                 ssd_type='xmnet',
                 sub_cell_length=5,
                 **kwargs):
        super(XMNetSearch, self).__init__()
        input_channel = input_channel
        self.width_mult = width_mult
        self.out_strides = out_stride
        self.out_layers = out_layers

        '''
        fix the cells
        '''
        # cells = [1, 0, 0, 0, 0,
        #          1, 0, 1, 1, 3,
        #          1, 0, 2, 2, 5,
        #          1, 0, 2, 3, 2,
        #          1, 0, 2, 3, 1,
        #          0, 0, 3, 4, 4]
        assert cells is not None, "cells must not be None."
        self.cells = cells

        init_stride = [2, 2, 2, 1, 2]
        init_channels = [48, 80, 128, 128, 160]
        init_repeats = [2, 3, 3, 3, 6]
        outlayer_outchannel_map = {1: 0, 2: 1, 3: 3, 4: 4}

        # resolve cells
        self.SSD = ssd_entry(ssd_type + "SSD")

        blocks = []
        kernel_sizes = []
        channels = []
        repeats = []
        expansions = []

        for i in range(5):
            sub_cell_config = cells[i * sub_cell_length: i * sub_cell_length + sub_cell_length]
            sub_cell = self.SSD.resolve_SSD(cell_config=sub_cell_config,
                                            BlockArgs=BlockArgs)

            blocks.append(sub_cell.block)
            kernel_sizes.append(sub_cell.kernel_size)
            channels.append(self._make_divisible(init_channels[i] * sub_cell.channel))
            repeats.append(max(init_repeats[i] + sub_cell.repeat, 1))
            expansions.append(sub_cell.expansion)

        self.out_planes = [channels[outlayer_outchannel_map[i]] for i in self.out_layers]

        logger.info(f'>>>>>> blocks: {blocks}')
        logger.info(f'>>>>>> kernel_size: {kernel_sizes}')
        logger.info(f'>>>>>> expansions: {expansions}')
        logger.info(f'>>>>>> channels: {channels}')
        logger.info(f'>>>>>> repeats: {repeats}')

        self.repeats = repeats
        self.feat_idxs = [np.cumsum([1] + repeats)[idx] for idx in [0, 1, 2, 4, 5]]

        # building first layer
        input_channel = self._make_divisible(input_channel * min(1.0, width_mult), round_nearest)

        features = [
            build_conv_norm(
                in_channels,
                input_channel,
                kernel_size=3,
                stride=2,
                padding=1,
                normalize=normalize)]

        # stage 0
        cin = self._make_divisible(24 * width_mult)
        features.append(
            FusedConv(input_channel, input_channel, cin, normalize, stride=1, kernel_size=3)
        )
        for i in range(5):
            for j in range(repeats[i]):
                cout = channels[i]
                s = init_stride[i] if j == 0 else 1

                block = get_block(blocks[i])(
                    in_features=cin,
                    hidden_features=cin * expansions[i],
                    out_features=cout,
                    normalize=normalize,
                    stride=s,
                    kernel_size=kernel_sizes[i]
                )
                cin = cout
                features.append(block)

        # make it nn.Sequential
        self.features = nn.ModuleList(features)

        self.init_weights()
        if initializer is not None:
            initialize_from_cfg(self, initializer)

    def _make_divisible(self, v, divisor=8, min_value=None):
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v

    def init_weights(self, pretrained=None):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode="fan_out", nonlinearity="sigmoid")
                nn.init.zeros_(m.bias)

    def get_outstrides(self):
        """
        Returns:

            - out (:obj:`int`): number of channels of output
        """

        return torch.tensor(self.out_strides, dtype=torch.int)

    def get_outplanes(self):
        """
        get dimension of the output tensor
        """
        return self.out_planes

    def forward(self, input):
        x = input["image"]
        outs = []
        for idx in range(len(self.features)):
            x = self.features[idx](x)
            if idx in self.feat_idxs:
                outs.append(x)
        features = [outs[i] for i in self.out_layers]

        return {'features': features, 'strides': self.get_outstrides()}


def xmnet_search(**kwargs):
    model = XMNetSearch(**kwargs)
    return model
