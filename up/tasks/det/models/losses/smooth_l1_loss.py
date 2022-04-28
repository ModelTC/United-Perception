# Import from third library
import torch
import numpy as np

# Import from local
from up.models.losses.loss import BaseLoss, _reduce
from up.utils.general.registry_factory import LOSSES_REGISTRY


__all__ = ['SmoothL1Loss']


def smooth_l1_loss(input, target, sigma, reduction='none', normalizer=None, code_weights=None, weights=None):
    beta = 1. / (sigma**2)
    diff = torch.abs(input - target)
    if code_weights is not None:
        diff = diff * code_weights.view(1, 1, -1)
    cond = diff < beta
    loss = torch.where(cond, 0.5 * diff**2 / beta, diff - 0.5 * beta)
    if weights is not None:
        assert weights.shape[0] == loss.shape[0] and weights.shape[1] == loss.shape[1]
        loss = loss * weights.unsqueeze(-1)
    loss = _reduce(loss, reduction, normalizer=normalizer)
    return loss


@LOSSES_REGISTRY.register('smooth_l1_loss')
class SmoothL1Loss(BaseLoss):
    def __init__(self,
                 name='smooth_l1',
                 reduction='mean',
                 loss_weight=1.0,
                 sigma=1.0,
                 code_weights=None
                 ):
        BaseLoss.__init__(self,
                          name=name,
                          reduction=reduction,
                          loss_weight=loss_weight)
        self.sigma = sigma
        self.key_fields = []
        self.code_weights = None
        if code_weights is not None:
            self.code_weights = np.array(code_weights, dtype=np.float32)
            self.code_weights = torch.from_numpy(self.code_weights).cuda()
        else:
            self.code_weights = None

    def forward(self, input, target, reduction, normalizer=None, weights=None):
        return smooth_l1_loss(input, target, self.sigma, reduction, normalizer, self.code_weights, weights)
