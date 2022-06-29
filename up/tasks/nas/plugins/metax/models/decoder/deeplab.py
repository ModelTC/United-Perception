import torch.nn as nn
from torch.nn import functional as F
from up.utils.model.normalize import build_norm_layer
from up.utils.general.registry_factory import MODULE_ZOO_REGISTRY
from up.models.losses import build_loss
from up.tasks.seg.models.components import Aux_Module, ASPP

__all__ = ["XMNetDeeplabv3"]


@MODULE_ZOO_REGISTRY.register('xmnet_deeplabv3')
class XMNetDeeplabv3(nn.Module):
    """
    Reference:
        Chen, Liang-Chieh, et al. *"Rethinking Atrous Convolution for Semantic Image Segmentation."*
    """
    def __init__(self,
                 inplanes,
                 aux_inplanes,
                 num_classes=19,
                 inner_planes=256,
                 dilations=(12, 24, 36),
                 with_aux=True,
                 normalize={'type': 'solo_bn'},
                 loss=None):
        super(XMNetDeeplabv3, self).__init__()
        self.prefix = self.__class__.__name__
        self.with_aux = with_aux
        self.aspp = ASPP(inplanes, inner_planes=inner_planes, normalize=normalize, dilations=dilations)
        self.head = nn.Sequential(
            nn.Conv2d(self.aspp.get_outplanes(), 256, kernel_size=3, padding=1, dilation=1, bias=False),
            build_norm_layer(256, normalize)[1],
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True))
        if self.with_aux:
            self.aux_layer = Aux_Module(aux_inplanes, num_classes, normalize=normalize)
        self.loss = build_loss(loss)

    def forward(self, x):
        x1, x2, x3, x4 = x['features']
        size = x['size']
        aspp_out = self.aspp(x4)
        pred = self.head(aspp_out)

        pred = F.upsample(pred, size=size, mode='bilinear', align_corners=True)
        if self.training and self.with_aux:
            gt_seg = x['gt_semantic_seg']
            aux_pred = self.aux_layer(x3)
            aux_pred = F.upsample(aux_pred, size=size, mode='bilinear', align_corners=True)
            pred = pred, aux_pred
            loss = self.loss(pred, gt_seg)
            return {f"{self.prefix}.loss": loss, "blob_pred": pred[0]}
        else:
            return {"blob_pred": pred}
