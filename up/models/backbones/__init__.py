from .mobilenet_v2 import mobilenetv2  # noqa: F401
from .mobilenet_v3 import mobilenetv3  # noqa: F401
from .resnet import (resnet101,  # noqa
                     resnet152,
                     resnet18,
                     resnet34,
                     resnet50,
                     resnet_custom)
from .regnet import (  # noqa: F401
    regnetx_200m, regnetx_400m, regnetx_600m, regnetx_800m,
    regnetx_1600m, regnetx_3200m, regnetx_4000m, regnetx_6400m,
    regnety_200m, regnety_400m, regnety_600m, regnety_800m,
    regnety_1600m, regnety_3200m, regnety_4000m, regnety_6400m,
)
from .efficientnet import (  # noqa: F401
    efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3,
    efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7
)
from .convnext import (  # noqa: F401
    convnext_tiny, convnext_small, convnext_base, convnext_large, convnext_xlarge
)

from .resnet_D import (resnet101_D,  # noqa
                       resnet152_D,
                       resnet18_D,
                       resnet34_D,
                       resnet50_D,
                       resnet_custom_D)

from .vision_transformer import (  # noqa
    vit_base_patch32_224, vit_base_patch16_224, vit_large_patch16_224, vit_huge_patch14_224,
    deit_tiny_patch16_224, deit_small_patch16_224, deit_base_patch16_224
)

from .swin_transformer import (swin_tiny,  # noqa: F401
                               swin_small,
                               swin_base_224,
                               swin_base_384,
                               swin_large_224,
                               swin_large_384)

from .cswin import (CSWin_64_12211_tiny_224,  # noqa: F401
                    CSWin_64_24322_small_224,
                    CSWin_96_24322_base_224,
                    CSWin_144_24322_large_224,
                    CSWin_96_24322_base_384,
                    CSWin_144_24322_large_384)

from .mb import (MB2_160,  # noqa: F401
                 MB2,
                 MB3,
                 MB4,
                 MB6,
                 MB7_new,
                 MB7_ckpt,
                 MB9,
                 MB9_ckpt,
                 MB12_ckpt)

from moco_vit import (vit_small,  # noqa: F401
                      vit_base,
                      vit_conv_small,
                      vit_conv_base)

from up.utils.general.registry_factory import MODULE_ZOO_REGISTRY

imported_vars = list(globals().items())

for var_name, var in imported_vars:
    if callable(var):
        MODULE_ZOO_REGISTRY.register(var_name, var)
