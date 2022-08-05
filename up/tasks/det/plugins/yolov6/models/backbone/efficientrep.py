# Import from third library
import torch
from torch import nn

from ..components import RepVGGBlock, RepBlock, SimSPPF, net_info
from up.utils.general.registry_factory import MODULE_ZOO_REGISTRY


__all__ = ["EfficientRep"]


@MODULE_ZOO_REGISTRY.register("EfficientRep")
class EfficientRep(nn.Module):
    '''EfficientRep Backbone
    EfficientRep is handcrafted by hardware-aware neural network design.
    With rep-style struct, EfficientRep is friendly to high-computation hardware(e.g. GPU).
    '''

    def __init__(
        self,
        depth_multiple=0.33,
        width_multiple=0.25,
        backbone_base_repeats=[1, 6, 12, 18, 6],
        backbone_base_channels=[64, 128, 256, 512, 1024],
        neck_base_repeats=[12, 12, 12, 12],
        neck_base_channels=[256, 128, 128, 256, 256, 512],
        input_channels=3,
        inplanes=None,
        out_layers=[2, 3, 4],
        out_strides=[8, 16, 32],
        normalize={'type': 'solo_bn'},
        act_fn={'type': 'ReLU'},
    ):
        super().__init__()

        self.out_layers = out_layers
        self.out_strides = out_strides
        out_channels, num_repeats = net_info(depth_multiple, width_multiple, backbone_base_repeats,
                                             backbone_base_channels, neck_base_repeats, neck_base_channels)
        self.out_channels, self.num_repeats = out_channels, num_repeats
        self.out_planes = [self.out_channels[i] for i in self.out_layers]

        self.stem = RepVGGBlock(
            in_channels=input_channels,
            out_channels=self.out_channels[0],
            kernel_size=3,
            stride=2,
            normalize=normalize,
            act_fn=act_fn,
        )

        self.ERBlock_2 = nn.Sequential(
            RepVGGBlock(
                in_channels=self.out_channels[0],
                out_channels=self.out_channels[1],
                kernel_size=3,
                stride=2,
                normalize=normalize,
                act_fn=act_fn,
            ),
            RepBlock(
                in_channels=self.out_channels[1],
                out_channels=self.out_channels[1],
                n=num_repeats[1],
                normalize=normalize,
                act_fn=act_fn,
            )
        )

        self.ERBlock_3 = nn.Sequential(
            RepVGGBlock(
                in_channels=self.out_channels[1],
                out_channels=self.out_channels[2],
                kernel_size=3,
                stride=2,
                normalize=normalize,
                act_fn=act_fn,
            ),
            RepBlock(
                in_channels=self.out_channels[2],
                out_channels=self.out_channels[2],
                n=num_repeats[2],
                normalize=normalize,
                act_fn=act_fn,
            )
        )

        self.ERBlock_4 = nn.Sequential(
            RepVGGBlock(
                in_channels=self.out_channels[2],
                out_channels=self.out_channels[3],
                kernel_size=3,
                stride=2,
                normalize=normalize,
                act_fn=act_fn,
            ),
            RepBlock(
                in_channels=self.out_channels[3],
                out_channels=self.out_channels[3],
                n=num_repeats[3],
                normalize=normalize,
                act_fn=act_fn,
            )
        )

        self.ERBlock_5 = nn.Sequential(
            RepVGGBlock(
                in_channels=self.out_channels[3],
                out_channels=self.out_channels[4],
                kernel_size=3,
                stride=2,
                normalize=normalize,
                act_fn=act_fn,
            ),
            RepBlock(
                in_channels=self.out_channels[4],
                out_channels=self.out_channels[4],
                n=num_repeats[4],
                normalize=normalize,
                act_fn=act_fn,
            ),
            SimSPPF(
                in_channels=self.out_channels[4],
                out_channels=self.out_channels[4],
                kernel_size=5,
                normalize=normalize,
                act_fn=act_fn
            )
        )

    def forward(self, x):
        x = x['image']
        outputs = []
        x = self.stem(x)
        outputs.append(x)
        x = self.ERBlock_2(x)
        outputs.append(x)
        x = self.ERBlock_3(x)
        outputs.append(x)
        x = self.ERBlock_4(x)
        outputs.append(x)
        x = self.ERBlock_5(x)
        outputs.append(x)

        features = [outputs[i] for i in self.out_layers]
        return {"features": features, "strides": torch.tensor(self.out_strides, dtype=torch.int)}

    def get_outplanes(self):
        return self.out_planes
