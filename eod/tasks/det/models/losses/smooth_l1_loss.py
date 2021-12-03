# Import from third library
import torch

# Import from local
from eod.models.losses.loss import BaseLoss, _reduce
from eod.utils.general.registry_factory import LOSSES_REGISTRY


__all__ = ['SmoothL1Loss']


def smooth_l1_loss(input, target, sigma, reduction='none', normalizer=None):
    beta = 1. / (sigma**2)
    diff = torch.abs(input - target)
    cond = diff < beta
    loss = torch.where(cond, 0.5 * diff**2 / beta, diff - 0.5 * beta)
    loss = _reduce(loss, reduction, normalizer=normalizer)
    return loss


@LOSSES_REGISTRY.register('smooth_l1_loss')
class SmoothL1Loss(BaseLoss):
    def __init__(self,
                 name='smooth_l1',
                 reduction='mean',
                 loss_weight=1.0,
                 sigma=1.0):
        BaseLoss.__init__(self,
                          name=name,
                          reduction=reduction,
                          loss_weight=loss_weight)
        self.sigma = sigma
        self.key_fields = []

    def forward(self, input, target, reduction, normalizer=None):
        return smooth_l1_loss(input, target, self.sigma, reduction, normalizer=normalizer)
