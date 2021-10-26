from eod.utils.general.registry_factory import LOSSES_REGISTRY
from .smooth_l1_loss import SmoothL1Loss  # noqa
from .focal_loss import SigmoidFocalLoss # noqa
from .entropy_loss import SigmoidCrossEntropyLoss # noqa
from .l1_loss import L1Loss # noqa
from .iou_loss import IOULoss # noqa


def build_loss(loss_cfg):
    return LOSSES_REGISTRY.build(loss_cfg)
