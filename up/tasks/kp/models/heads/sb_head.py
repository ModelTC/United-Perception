# Import from third library
import torch.nn as nn
from up.utils.general.registry_factory import MODULE_ZOO_REGISTRY
from up.utils.model.normalize import build_norm_layer

__all__ = ['KeypointSBHead']


@MODULE_ZOO_REGISTRY.register('keypoint_sb')
class KeypointSBHead(nn.Module):
    """
    keypoint head for solo top-down keypoint detection method.(HKD)
    """
    def __init__(self,
                 inplanes,
                 mid_channels,
                 num_classes,
                 has_bg,
                 normalize={'type': 'solo_bn'}):
        super(KeypointSBHead, self).__init__()
        self.has_bg = has_bg
        self.num_classes = num_classes

        self.relu = nn.ReLU(inplace=True)

        self.upsample5 = nn.ConvTranspose2d(
            inplanes[0], mid_channels, kernel_size=4, stride=2, padding=1)
        self.upsample4 = nn.ConvTranspose2d(
            mid_channels, mid_channels, kernel_size=4, stride=2, padding=1)
        self.upsample3 = nn.ConvTranspose2d(
            mid_channels, mid_channels, kernel_size=4, stride=2, padding=1)

        self.bn_s5 = build_norm_layer(mid_channels, normalize)[1]
        self.bn_s4 = build_norm_layer(mid_channels, normalize)[1]
        self.bn_s3 = build_norm_layer(mid_channels, normalize)[1]
        if has_bg:
            self.predict3 = nn.Conv2d(mid_channels, num_classes + 1, kernel_size=1, stride=1, padding=0)
        else:
            self.predict3 = nn.Conv2d(mid_channels, num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, input):
        output = {}
        output['pred'] = self.forward_net(input)
        output['num_classes'] = self.num_classes
        output['has_bg'] = self.has_bg
        return output

    def forward_net(self, input):
        c5 = input['features'][0]
        p5 = self.relu(self.bn_s5(self.upsample5(c5)))
        p4 = self.relu(self.bn_s4(self.upsample4(p5)))
        p3 = self.relu(self.bn_s3(self.upsample3(p4)))
        pred = self.predict3(p3)
        return pred
