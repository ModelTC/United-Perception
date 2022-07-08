# Import from third library

import torch
from torch import nn
from ..components import RepBlock, SimConv, Transpose
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
        inplanes=None,
        num_repeats=None,
        out_strides=[8, 16, 32]
    ):
        super().__init__()

        self.out_channels = inplanes
        self.out_strides = out_strides

        assert inplanes is not None
        assert num_repeats is not None

        self.Rep_p4 = RepBlock(
            in_channels=self.out_channels[3] + self.out_channels[5],
            out_channels=self.out_channels[5],
            n=num_repeats[5],
        )

        self.Rep_p3 = RepBlock(
            in_channels=self.out_channels[2] + self.out_channels[6],
            out_channels=self.out_channels[6],
            n=num_repeats[6]
        )

        self.Rep_n3 = RepBlock(
            in_channels=self.out_channels[6] + self.out_channels[7],
            out_channels=self.out_channels[8],
            n=num_repeats[7],
        )

        self.Rep_n4 = RepBlock(
            in_channels=self.out_channels[5] + self.out_channels[9],
            out_channels=self.out_channels[10],
            n=num_repeats[8]
        )

        self.reduce_layer0 = SimConv(
            in_channels=self.out_channels[4],
            out_channels=self.out_channels[5],
            kernel_size=1,
            stride=1
        )

        self.upsample0 = Transpose(
            in_channels=self.out_channels[5],
            out_channels=self.out_channels[5],
        )

        self.reduce_layer1 = SimConv(
            in_channels=self.out_channels[5],
            out_channels=self.out_channels[6],
            kernel_size=1,
            stride=1
        )

        self.upsample1 = Transpose(
            in_channels=self.out_channels[6],
            out_channels=self.out_channels[6]
        )

        self.downsample2 = SimConv(
            in_channels=self.out_channels[6],
            out_channels=self.out_channels[7],
            kernel_size=3,
            stride=2
        )

        self.downsample1 = SimConv(
            in_channels=self.out_channels[8],
            out_channels=self.out_channels[9],
            kernel_size=3,
            stride=2
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
        return self.out_channels
