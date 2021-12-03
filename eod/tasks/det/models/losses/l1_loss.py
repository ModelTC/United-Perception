# Import from third library
import torch

# Import from local
from eod.models.losses.loss import BaseLoss, _reduce
from eod.utils.general.registry_factory import LOSSES_REGISTRY


__all__ = ['L1Loss']


def l1_loss(input, target, scale_type='linear', reduction='none', normalizer=None):
    if scale_type == 'linear':
        input = input
        target = target
    elif scale_type == 'log':
        input = torch.log(input)
        target = torch.log(target)
    else:
        raise NotImplementedError
    loss = torch.abs(input - target)
    loss = _reduce(loss, reduction=reduction, normalizer=normalizer)
    return loss


@LOSSES_REGISTRY.register('l1_loss')
class L1Loss(BaseLoss):
    def __init__(self,
                 name='l1_loss',
                 reduction='mean',
                 loss_weight=1.0,
                 scale_type='linear'):
        r"""
        Arguments:
            - name (:obj:`str`): name of the loss function
            - reduction (:obj:`str`): reduction type, choice of mean, none, sum
            - loss_weight (:obj:`float`): loss weight
            - scale_type index (:obj:`str`): choice of 'linear', 'log'
        """
        BaseLoss.__init__(self,
                          name=name,
                          reduction=reduction,
                          loss_weight=loss_weight)
        self.scale_type = scale_type
        self.key_fields = []

    def forward(self, input, target, reduction, normalizer=None):
        """
        Arguments:
            - input (:obj:`FloatTenosr`):
            - output (:obj:`FloatTenosr`): same shape as input
        """
        assert input.shape == target.shape
        loss = l1_loss(input, target, scale_type=self.scale_type, reduction=reduction,
                       normalizer=normalizer)
        return loss
