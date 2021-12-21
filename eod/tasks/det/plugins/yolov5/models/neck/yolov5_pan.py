# Standard Library

# Import from third library
import torch
import torch.nn as nn

from eod.utils.model.initializer import initialize_from_cfg
from eod.utils.general.registry_factory import MODULE_ZOO_REGISTRY
from ..components import Concat, ConvBnAct, BottleneckCSP


__all__ = ['DarknetPAFPN']


@MODULE_ZOO_REGISTRY.register('DarknetPAFPN')
class DarknetPAFPN(nn.Module):
    """
    Stacked PA Feature Pyramid Network
    """

    def __init__(self,
                 inplanes,
                 csp2_block_nums,
                 outplanes=-1,
                 num_outs=3,
                 out_strides=[8, 16, 32],
                 normalize={'type': 'solo_bn'},
                 act_fn={'type': 'Hardswish'},
                 initializer=None,
                 downsample='conv'):
        """
        Arguments:
            - inplanes (:obj:`list` of :obj:`int`): input channel
            - outplanes (:obj:`list` of :obj:`int`): output channel
            - normalize (:obj:`dict`): config of Normalization Layer
            - initializer (:obj:`dict`): config for model parameter initialization
        """

        super(DarknetPAFPN, self).__init__()

        assert isinstance(inplanes, list), 'Only support list'
        assert isinstance(csp2_block_nums, int), 'csp2_block_nums must be an integer'
        assert num_outs >= len(inplanes), 'num_outs must be greater than or equal to inplanes length'
        assert num_outs == len(out_strides), 'num_outs must equal to strides length'

        self.inplanes = inplanes
        plane = inplanes[-1]
        self.out_align = False
        self.downsample = downsample
        out_planes_pre = inplanes + [plane for i in range(num_outs - len(inplanes))]
        if isinstance(outplanes, int) and outplanes <= 0:
            self.out_planes = out_planes_pre
        elif isinstance(outplanes, int):
            self.out_align = True
            self.out_planes = [outplanes] * num_outs
        elif isinstance(outplanes, list):
            assert len(outplanes) == num_outs
            self.out_align = True
            self.out_planes = outplanes
        else:
            raise NotImplementedError

        lines = len(inplanes) - 1
        same_plane_count = 0
        for i in range(lines):
            if inplanes[i] == inplanes[-1]:
                same_plane_count += 1

        self.up_sample_stream = nn.ModuleList()
        for i in range(lines):
            if i < same_plane_count:
                conv_block = ConvBnAct(plane, plane, 1, 1, normalize=normalize, act_fn=act_fn)
                self.up_sample_stream.append(conv_block)
            else:
                conv_block = ConvBnAct(plane, plane // 2, 1, 1, normalize=normalize, act_fn=act_fn)
                self.up_sample_stream.append(conv_block)
                plane = plane // 2
            upsample = nn.Upsample(None, 2, 'nearest')
            self.up_sample_stream.append(upsample)
            cat = Concat(1)
            self.up_sample_stream.append(cat)
            plane = plane * 2
            csp = BottleneckCSP(plane, plane // 2, nums=csp2_block_nums,
                                shortcut=False, normalize=normalize, act_fn=act_fn)
            self.up_sample_stream.append(csp)
            plane = plane // 2

        self.down_sample_stream = nn.ModuleList()
        for i in range(lines):
            conv_block = ConvBnAct(plane, plane, 3, 2, normalize=normalize, act_fn=act_fn)
            self.down_sample_stream.append(conv_block)
            cat = Concat(1)
            self.down_sample_stream.append(cat)
            plane = plane * 2
            if i >= lines - same_plane_count:
                csp = BottleneckCSP(plane, plane // 2, nums=csp2_block_nums,
                                    shortcut=False, normalize=normalize, act_fn=act_fn)
                plane = plane // 2
            else:
                csp = BottleneckCSP(plane, plane, nums=csp2_block_nums,
                                    shortcut=False, normalize=normalize, act_fn=act_fn)
            self.down_sample_stream.append(csp)

        # add conv layers on top of feature maps (RetinaNet)
        self.down_sample_convs = nn.ModuleList()
        for i in range(num_outs - len(inplanes)):
            conv_block = ConvBnAct(plane, plane, 3, 2, normalize=normalize, act_fn=act_fn)
            self.down_sample_convs.append(conv_block)
            if self.downsample == 'csp':
                csp = BottleneckCSP(plane, plane, nums=csp2_block_nums,
                                    shortcut=False, normalize=normalize, act_fn=act_fn)
                self.down_sample_convs.append(csp)

        # add out planes align
        self.out_align_convs = nn.ModuleList()
        if self.out_align:
            for i in range(len(out_planes_pre)):
                conv_block = ConvBnAct(out_planes_pre[i], self.out_planes[i], 3, 1, normalize=normalize, act_fn=act_fn)
                self.out_align_convs.append(conv_block)

        self.out_strides = out_strides

        # Init weights, biases
        if initializer is not None:
            initialize_from_cfg(self, initializer)

    def get_outplanes(self):
        """
        get dimension of the output tensor
        """
        return self.out_planes

    def forward(self, x):
        res = []
        features = x['features']
        assert isinstance(features, list), 'Only support list'

        f = features[-1]

        down_save = []
        for i, m in enumerate(self.up_sample_stream):
            t = type(m)
            if t is Concat:
                f = m([f, features[-(i // 4 + 2)]])
            elif t is ConvBnAct:
                f = m(f)
                down_save.append(f)
            else:
                f = m(f)

        res.append(f)
        for i, m in enumerate(self.down_sample_stream):
            t = type(m)
            if t is Concat:
                f = m([f, down_save[-(i // 3 + 1)]])
            elif t is BottleneckCSP:
                f = m(f)
                res.append(f)
            else:
                f = m(f)

        for i, m in enumerate(self.down_sample_convs):
            t = type(m)
            f = m(f)
            if self.downsample == 'csp' and t is BottleneckCSP:
                res.append(f)
            else:
                if self.downsample == 'conv':
                    res.append(f)

        if not self.out_align:
            return {'features': res, 'strides': self.get_outstrides()}
        else:
            feats = []
            for i, m in enumerate(self.out_align_convs):
                feats.append(m(res[i]))
            return {'features': feats, 'strides': self.get_outstrides()}

    def get_outstrides(self):
        return torch.tensor(self.out_strides, dtype=torch.int)
