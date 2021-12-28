import torch.nn as nn
from eod.utils.model.normalize import build_norm_layer
from eod.utils.general.registry_factory import MODULE_ZOO_REGISTRY

__all__ = ['UNet_Encoder']


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 normalize={'type': 'solo_bn'},
                 mid_channels=None):
        super().__init__()
        self._normalize = normalize
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            build_norm_layer(mid_channels, normalize)[1],
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            build_norm_layer(out_channels, normalize)[1],
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 normalize={'type': 'solo_bn'}):
        super().__init__()
        self._normalize = normalize
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, normalize=normalize)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


@MODULE_ZOO_REGISTRY.register('unet_encoder')
class UNet_Encoder(nn.Module):
    def __init__(self,
                 in_channels,
                 bilinear=True,
                 normalize={'type': 'solo_bn'}):
        super(UNet_Encoder, self).__init__()
        self.inplanes = in_channels
        self.bilinear = bilinear
        self.inc = DoubleConv(in_channels, 64, normalize)
        self.down1 = Down(64, 128, normalize)
        self.down2 = Down(128, 256, normalize)
        self.down3 = Down(256, 512, normalize)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor, normalize)

    def forward(self, input):
        img = input["image"]
        size = img.size()[2:]

        x1 = self.inc(img)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        res = {}
        res["features"] = [x1, x2, x3, x4, x5]
        res["size"] = size
        return res

    def get_outplanes(self):
        return self.inplanes

    def get_auxplanes(self):
        return self.inplanes // 2


def unet_encoder(**kwargs):
    model = UNet_Encoder(**kwargs)
    return model
