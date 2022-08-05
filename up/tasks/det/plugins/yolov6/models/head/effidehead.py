import torch
import torch.nn as nn
import math
from ..components import Conv
from up.utils.general.registry_factory import MODULE_ZOO_REGISTRY

__all__ = ['Effidehead']


@MODULE_ZOO_REGISTRY.register('Effidehead')
class Effidehead(nn.Module):
    '''Efficient Decoupled Head
    With hardware-aware degisn, the decoupled head is optimized with
    hybridchannels methods.
    '''

    def __init__(self,
                 inplanes,
                 num_classes=81,
                 anchors=1,
                 num_layers=3,
                 class_activation='sigmoid',
                 stride=[8, 16, 32],
                 prior_prob=1e-2,
                 normalize={'type': 'solo_bn'},
                 act_fn={'type': 'Silu'}):  # detection layer
        super().__init__()

        assert inplanes is not None
        self.out_channels = inplanes
        num_classes = {'sigmoid': -1, 'softmax': 0}[class_activation] + num_classes
        head_layers = build_effidehead_layer(self.out_channels, anchors, num_classes, normalize, act_fn)
        self.nl = num_layers  # number of detection layers
        if isinstance(anchors, (list, tuple)):
            self.na = len(anchors[0]) // 2
        else:
            self.na = anchors
        self.anchors = anchors
        self.prior_prob = prior_prob
        self.stride = torch.tensor(stride)

        # Init decouple head
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()

        # Efficient decoupled head layers
        for i in range(num_layers):
            idx = i * 6
            self.stems.append(head_layers[idx])
            self.cls_convs.append(head_layers[idx + 1])
            self.reg_convs.append(head_layers[idx + 2])
            self.cls_preds.append(head_layers[idx + 3])
            self.reg_preds.append(head_layers[idx + 4])
            self.obj_preds.append(head_layers[idx + 5])

        self.initialize_biases()

    def initialize_biases(self):
        for conv in self.cls_preds:
            b = conv.bias.view(self.na, -1)
            b.data.fill_(-math.log((1 - self.prior_prob) / self.prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        for conv in self.obj_preds:
            b = conv.bias.view(self.na, -1)
            b.data.fill_(-math.log((1 - self.prior_prob) / self.prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward_net(self, x):
        mlvl_preds = []
        for i in range(self.nl):
            feat = self.stems[i](x[i])
            cls_feat = self.cls_convs[i](feat)
            cls_output = self.cls_preds[i](cls_feat)
            reg_feat = self.reg_convs[i](feat)
            reg_output = self.reg_preds[i](reg_feat)
            obj_output = self.obj_preds[i](reg_feat)

            mlvl_preds.append((cls_output, reg_output, obj_output))

        return mlvl_preds

    def forward(self, input):
        features = input['features']
        mlvl_raw_preds = self.forward_net(features)
        output = {}
        output['preds'] = mlvl_raw_preds
        return output


def build_effidehead_layer(channels_list, num_anchors, num_classes, normalize, act_fn):
    head_layers = nn.Sequential(
        # stem0
        Conv(
            in_channels=channels_list[0],
            out_channels=channels_list[0],
            kernel_size=1,
            stride=1,
            normalize=normalize,
            act_fn=act_fn
        ),
        # cls_conv0
        Conv(
            in_channels=channels_list[0],
            out_channels=channels_list[0],
            kernel_size=3,
            stride=1,
            normalize=normalize,
            act_fn=act_fn
        ),
        # reg_conv0
        Conv(
            in_channels=channels_list[0],
            out_channels=channels_list[0],
            kernel_size=3,
            stride=1,
            normalize=normalize,
            act_fn=act_fn
        ),
        # cls_pred0
        nn.Conv2d(
            in_channels=channels_list[0],
            out_channels=num_classes * num_anchors,
            kernel_size=1
        ),
        # reg_pred0
        nn.Conv2d(
            in_channels=channels_list[0],
            out_channels=4 * num_anchors,
            kernel_size=1
        ),
        # obj_pred0
        nn.Conv2d(
            in_channels=channels_list[0],
            out_channels=1 * num_anchors,
            kernel_size=1
        ),
        # stem1
        Conv(
            in_channels=channels_list[1],
            out_channels=channels_list[1],
            kernel_size=1,
            stride=1,
            normalize=normalize,
            act_fn=act_fn
        ),
        # cls_conv1
        Conv(
            in_channels=channels_list[1],
            out_channels=channels_list[1],
            kernel_size=3,
            stride=1,
            normalize=normalize,
            act_fn=act_fn
        ),
        # reg_conv1
        Conv(
            in_channels=channels_list[1],
            out_channels=channels_list[1],
            kernel_size=3,
            stride=1,
            normalize=normalize,
            act_fn=act_fn
        ),
        # cls_pred1
        nn.Conv2d(
            in_channels=channels_list[1],
            out_channels=num_classes * num_anchors,
            kernel_size=1
        ),
        # reg_pred1
        nn.Conv2d(
            in_channels=channels_list[1],
            out_channels=4 * num_anchors,
            kernel_size=1
        ),
        # obj_pred1
        nn.Conv2d(
            in_channels=channels_list[1],
            out_channels=1 * num_anchors,
            kernel_size=1
        ),
        # stem2
        Conv(
            in_channels=channels_list[2],
            out_channels=channels_list[2],
            kernel_size=1,
            stride=1
        ),
        # cls_conv2
        Conv(
            in_channels=channels_list[2],
            out_channels=channels_list[2],
            kernel_size=3,
            stride=1,
            normalize=normalize,
            act_fn=act_fn
        ),
        # reg_conv2
        Conv(
            in_channels=channels_list[2],
            out_channels=channels_list[2],
            kernel_size=3,
            stride=1,
            normalize=normalize,
            act_fn=act_fn
        ),
        # cls_pred2
        nn.Conv2d(
            in_channels=channels_list[2],
            out_channels=num_classes * num_anchors,
            kernel_size=1
        ),
        # reg_pred2
        nn.Conv2d(
            in_channels=channels_list[2],
            out_channels=4 * num_anchors,
            kernel_size=1
        ),
        # obj_pred2
        nn.Conv2d(
            in_channels=channels_list[2],
            out_channels=1 * num_anchors,
            kernel_size=1
        )
    )
    return head_layers
