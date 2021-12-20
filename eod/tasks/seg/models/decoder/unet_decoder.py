import torch
import torch.nn as nn
from torch.nn import functional as F
from eod.utils.model.normalize import build_norm_layer
from eod.utils.general.registry_factory import MODULE_ZOO_REGISTRY
from eod.models.losses import build_loss

__all__ = ['UNet_Decoder']


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels=None,
                 normalize={'type': 'solo_bn'}):
        super().__init__()
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


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 bilinear=True,
                 normalize={'type': 'solo_bn'}
                 ):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, normalize)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, normalize)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW    # 4， 2048，34， 34
        diffY = x2.size()[2] - x1.size()[2]  #
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


@MODULE_ZOO_REGISTRY.register('unet_decoder')
class UNet_Decoder(nn.Module):
    def __init__(self,
                 inplanes,
                 num_classes=19,
                 bilinear=True,
                 normalize={'type': 'solo_bn'},
                 with_aux=False,
                 loss=None):
        super(UNet_Decoder, self).__init__()
        self.prefix = self.__class__.__name__
        self.n_channels = inplanes
        self.n_classes = num_classes
        self.bilinear = bilinear
        self.with_aux = with_aux
        factor = 2 if bilinear else 1
        self.up1 = Up(1024, 512 // factor, bilinear, normalize)
        self.up2 = Up(512, 256 // factor, bilinear, normalize)
        self.up3 = Up(256, 128 // factor, bilinear, normalize)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, num_classes)
        self.loss = build_loss(loss)

    def forward(self, x):
        x1, x2, x3, x4, x5 = x['features']
        gt_seg = x['gt_seg']
        x_up3 = self.up1(x5, x4)
        x_up2 = self.up2(x_up3, x3)
        x_up1 = self.up3(x_up2, x2)
        x_out = self.up4(x_up1, x1)
        pred = self.outc(x_out)

        if self.training:
            loss = self.loss(pred, gt_seg)
            return {f"{self.prefix}.loss": loss, "blob_pred": pred}
        else:
            return {"blob_pred": pred}
