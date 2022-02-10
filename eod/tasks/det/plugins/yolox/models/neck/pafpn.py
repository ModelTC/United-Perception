# Import from third library
import torch
import torch.nn as nn
import torch.nn.functional as F

from eod.utils.general.registry_factory import MODULE_ZOO_REGISTRY

from eod.tasks.det.plugins.yolov5.models.components import ConvBnAct
from eod.tasks.det.plugins.yolox.models.backbone.cspdarknet import CSPLayer, DWConv


__all__ = ['YoloxPAFPN']


@MODULE_ZOO_REGISTRY.register('YoloxPAFPN')
class YoloxPAFPN(nn.Module):
    ''' PA Feature Pyramid Network used in YOLOX '''

    def __init__(self,
                 inplanes,
                 depth=0.33,
                 out_strides=[8, 16, 32],
                 depthwise=False,
                 act_fn={'type': 'Silu'},
                 upsample='nearest',
                 align_corners=True,
                 normalize={'type': 'solo_bn'},
                 downsample_plane=256,
                 tocaffe_friendly=False):
        super(YoloxPAFPN, self).__init__()
        self.inplanes = inplanes
        in_channels = inplanes
        self.outplanes = inplanes
        self.out_strides = out_strides
        self.num_level = len(out_strides)
        self.tocaffe_friendly = tocaffe_friendly
        Conv = DWConv if depthwise else ConvBnAct

        self.upsample = upsample
        if upsample == 'nearest':
            align_corners = None
        self.align_corners = align_corners

        self.lateral_conv0 = ConvBnAct(
            int(in_channels[2]), int(in_channels[1]),
            kernel_size=1, stride=1, act_fn=act_fn, normalize=normalize
        )
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1]), int(in_channels[1]),
            n=round(3 * depth), depthwise=depthwise, act=act_fn, normalize=normalize
        )

        self.reduce_conv1 = ConvBnAct(
            int(in_channels[1]), int(in_channels[0]),
            kernel_size=1, stride=1, act_fn=act_fn, normalize=normalize,
        )
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0]), int(in_channels[0]),
            n=round(3 * depth), depthwise=depthwise, act=act_fn, normalize=normalize
        )

        # bottom-up conv
        self.bu_conv2 = Conv(
            int(in_channels[0]), int(in_channels[0]),
            kernel_size=3, stride=2, act_fn=act_fn, normalize=normalize
        )
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0]), int(in_channels[1]),
            n=round(3 * depth), depthwise=depthwise, act=act_fn, normalize=normalize
        )

        # bottom-up conv
        self.bu_conv1 = Conv(
            int(in_channels[1]), int(in_channels[1]),
            kernel_size=3, stride=2, act_fn=act_fn, normalize=normalize
        )
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1]), int(in_channels[2]),
            n=round(3 * depth), depthwise=depthwise, act=act_fn, normalize=normalize
        )

        for i in range(self.num_level - len(inplanes)):
            level = len(inplanes) + i
            self.add_module(
                f"p{level}_conv",
                ConvBnAct(self.outplanes[-1],
                          downsample_plane,
                          kernel_size=3,
                          stride=2,
                          act_fn=act_fn,
                          normalize=normalize))
            self.outplanes.append(downsample_plane)

    def forward(self, input):
        laterals = input['features']
        [x2, x1, x0] = laterals
        features = []

        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        if self.tocaffe_friendly:
            f_out0 = F.interpolate(fpn_out0,
                               #size=x1.shape[-2:],
                               scale_factor = 2,
                               mode=self.upsample,
                               align_corners=self.align_corners)
        else:
            f_out0 = F.interpolate(fpn_out0,
                               size=x1.shape[-2:],
                               mode=self.upsample,
                               align_corners=self.align_corners)
        f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16
        # features.append(f_out0)

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        if self.tocaffe_friendly:
            f_out1 = F.interpolate(fpn_out1,
                               scale_factor = 2,
                               mode=self.upsample,
                               align_corners=self.align_corners)
        else:
            f_out1 = F.interpolate(fpn_out1,
                               size=x2.shape[-2:],
                               mode=self.upsample,
                               align_corners=self.align_corners)
        f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8
        features.append(pan_out2)

        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16
        features.append(pan_out1)

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32
        features.append(pan_out0)
        x = features[-1]
        for i in range(self.num_level - len(self.inplanes)):
            level = i + len(self.inplanes)
            x = self.get_downsample(level)(x)
            features.append(x)

        return {'features': features, 'strides': self.get_outstrides()}

    def get_outplanes(self):
        return self.outplanes

    def get_outstrides(self):
        return torch.tensor(self.out_strides, dtype=torch.int)

    def get_downsample(self, idx):
        return getattr(self, f"p{idx}_conv")
