import torch.nn as nn
from up.utils.general.registry_factory import MODULE_ZOO_REGISTRY
from up.utils.model.normalize import build_norm_layer

__all__ = ['KeypointFPN']


@MODULE_ZOO_REGISTRY.register('kp_fpn')
class KeypointFPN(nn.Module):
    def __init__(self,
                 inplanes,
                 outplanes,
                 num_classes,
                 normalize={'type': 'solo_bn'}):
        super(KeypointFPN, self).__init__()
        self.num_classes = num_classes
        self.relu = nn.ReLU(inplace=True)

        self.upsample5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.outplanes = outplanes

        self.uprefine5 = nn.Conv2d(
            outplanes,
            outplanes,
            kernel_size=1,
            stride=1,
            padding=0)
        self.uprefine4 = nn.Conv2d(
            outplanes,
            outplanes,
            kernel_size=1,
            stride=1,
            padding=0)
        self.uprefine3 = nn.Conv2d(
            outplanes,
            outplanes,
            kernel_size=1,
            stride=1,
            padding=0)

        self.c5_lateral = nn.Conv2d(
            inplanes[3],
            outplanes,
            kernel_size=3,
            stride=1,
            padding=1)
        self.c4_lateral = nn.Conv2d(
            inplanes[2],
            # inplanes[0]//2,
            outplanes,
            kernel_size=1,
            stride=1,
            padding=0)
        self.c3_lateral = nn.Conv2d(
            inplanes[1],
            # inplanes[0]//4,
            outplanes,
            kernel_size=1,
            stride=1,
            padding=0)
        self.c2_lateral = nn.Conv2d(
            inplanes[0],
            # inplanes[0]//8,
            outplanes,
            kernel_size=1,
            stride=1,
            padding=0)

        self.conv_after_sum4 = nn.Conv2d(
            outplanes,
            outplanes,
            kernel_size=3,
            stride=1,
            padding=1)
        self.bn_s4 = build_norm_layer(outplanes, normalize)[1]
        self.conv_after_sum3 = nn.Conv2d(
            outplanes,
            outplanes,
            kernel_size=3,
            stride=1,
            padding=1)
        self.bn_s3 = build_norm_layer(outplanes, normalize)[1]
        self.conv_after_sum2 = nn.Conv2d(
            outplanes,
            outplanes,
            kernel_size=3,
            stride=1,
            padding=1)
        self.bn_s2 = build_norm_layer(outplanes, normalize)[1]

    def forward(self, input):
        c2, c3, c4, c5 = input['features']
        p5 = self.relu(self.c5_lateral(c5))

        p5_up = self.uprefine5(self.upsample5(p5))
        c4_lat = self.c4_lateral(c4)
        p4 = self.relu(self.bn_s4(self.conv_after_sum4(c4_lat + p5_up)))

        p4_up = self.uprefine4(self.upsample4(p4))
        c3_lat = self.c3_lateral(c3)
        p3 = self.relu(self.bn_s3(self.conv_after_sum3(c3_lat + p4_up)))

        p3_up = self.uprefine3(self.upsample3(p3))
        c2_lat = self.c2_lateral(c2)
        p2 = self.relu(self.bn_s2(self.conv_after_sum2(c2_lat + p3_up)))

        out = [p5, p4, p3, p2]

        return {"features": out}

    def get_outplanes(self):
        """
        Return:
            - outplanes (:obj:`list` of :obj:`int`)
        """
        return self.outplanes
