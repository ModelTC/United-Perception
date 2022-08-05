# Import from third library

import torch
from torch import nn
from ..components import RepBlock, SimConv, Transpose, net_info
from up.utils.general.registry_factory import MODULE_ZOO_REGISTRY

__all__ = ["RepPANNeck"]


@MODULE_ZOO_REGISTRY.register("RepPANNeck")
class RepPANNeck(nn.Module):
    """RepPANNeck Module
    EfficientRep is the default backbone of this model.
    RepPANNeck has the balance of feature fusion ability and hardware efficiency.
    """

    def __init__(
        self,
        depth_multiple=0.33,
        width_multiple=0.25,
        backbone_base_repeats=[1, 6, 12, 18, 6],
        backbone_base_channels=[64, 128, 256, 512, 1024],
        neck_base_repeats=[12, 12, 12, 12],
        neck_base_channels=[256, 128, 128, 256, 256, 512],
        inplanes=None,
        out_strides=[8, 16, 32],
        normalize={'type': 'solo_bn'},
        act_fn={'type': 'ReLU'},
    ):
        super().__init__()

        out_channels, num_repeats = net_info(depth_multiple, width_multiple, backbone_base_repeats,
                                             backbone_base_channels, neck_base_repeats, neck_base_channels)
        self.out_channels, self.num_repeats = out_channels, num_repeats
        self.out_strides = out_strides
        self.out_planes = []

        self.Rep_p4 = RepBlock(
            in_channels=self.out_channels[3] + self.out_channels[5],
            out_channels=self.out_channels[5],
            n=num_repeats[5],
            normalize=normalize,
            act_fn=act_fn,
        )

        self.Rep_p3 = RepBlock(
            in_channels=self.out_channels[2] + self.out_channels[6],
            out_channels=self.out_channels[6],
            n=num_repeats[6],
            normalize=normalize,
            act_fn=act_fn,
        )
        self.out_planes.append(self.out_channels[6])

        self.Rep_n3 = RepBlock(
            in_channels=self.out_channels[6] + self.out_channels[7],
            out_channels=self.out_channels[8],
            n=num_repeats[7],
            normalize=normalize,
            act_fn=act_fn,
        )
        self.out_planes.append(self.out_channels[8])

        self.Rep_n4 = RepBlock(
            in_channels=self.out_channels[5] + self.out_channels[9],
            out_channels=self.out_channels[10],
            n=num_repeats[8],
            normalize=normalize,
            act_fn=act_fn,
        )
        self.out_planes.append(self.out_channels[10])

        self.reduce_layer0 = SimConv(
            in_channels=self.out_channels[4],
            out_channels=self.out_channels[5],
            kernel_size=1,
            stride=1,
            normalize=normalize,
            act_fn=act_fn
        )

        self.upsample0 = Transpose(
            in_channels=self.out_channels[5],
            out_channels=self.out_channels[5],
        )

        self.reduce_layer1 = SimConv(
            in_channels=self.out_channels[5],
            out_channels=self.out_channels[6],
            kernel_size=1,
            stride=1,
            normalize=normalize,
            act_fn=act_fn
        )

        self.upsample1 = Transpose(
            in_channels=self.out_channels[6],
            out_channels=self.out_channels[6]
        )

        self.downsample2 = SimConv(
            in_channels=self.out_channels[6],
            out_channels=self.out_channels[7],
            kernel_size=3,
            stride=2,
            normalize=normalize,
            act_fn=act_fn
        )

        self.downsample1 = SimConv(
            in_channels=self.out_channels[8],
            out_channels=self.out_channels[9],
            kernel_size=3,
            stride=2,
            normalize=normalize,
            act_fn=act_fn
        )

    def forward(self, input):

        (x2, x1, x0) = input["features"]

        fpn_out0 = self.reduce_layer0(x0)
        upsample_feat0 = self.upsample0(fpn_out0)
        f_concat_layer0 = torch.cat([upsample_feat0, x1], 1)
        f_out0 = self.Rep_p4(f_concat_layer0)

        fpn_out1 = self.reduce_layer1(f_out0)
        upsample_feat1 = self.upsample1(fpn_out1)
        f_concat_layer1 = torch.cat([upsample_feat1, x2], 1)
        pan_out2 = self.Rep_p3(f_concat_layer1)

        down_feat1 = self.downsample2(pan_out2)
        p_concat_layer1 = torch.cat([down_feat1, fpn_out1], 1)
        pan_out1 = self.Rep_n3(p_concat_layer1)

        down_feat0 = self.downsample1(pan_out1)
        p_concat_layer2 = torch.cat([down_feat0, fpn_out0], 1)
        pan_out0 = self.Rep_n4(p_concat_layer2)

        outputs = [pan_out2, pan_out1, pan_out0]

        return {"features": outputs, "strides": torch.tensor(self.out_strides, dtype=torch.int)}

    def get_outplanes(self):
        return self.out_planes
