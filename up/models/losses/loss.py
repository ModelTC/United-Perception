# Import from third library
from torch.nn.modules.loss import _Loss


def _reduce(loss, reduction, **kwargs):
    if reduction == 'none':
        ret = loss
    elif reduction == 'mean':
        normalizer = loss.numel()
        if kwargs.get('normalizer', None):
            normalizer = kwargs['normalizer']
        ret = loss.sum() / normalizer
    elif reduction == 'sum':
        ret = loss.sum()
    else:
        raise ValueError(reduction + ' is not valid')
    return ret


class BaseLoss(_Loss):
    # do not use syntax like `super(xxx, self).__init__,
    # which will cause infinited recursion while using class decorator`
    def __init__(self,
                 name='base',
                 reduction='none',
                 loss_weight=1.0):
        r"""
        Arguments:
            - name (:obj:`str`): name of the loss function
            - reduction (:obj:`str`): reduction type, choice of mean, none, sum
            - loss_weight (:obj:`float`): loss weight
        """
        _Loss.__init__(self, reduction=reduction)
        self.loss_weight = loss_weight
        self.name = name

    def __call__(self, input, target, reduction_override=None, normalizer_override=None, **kwargs):
        r"""
        Arguments:
            - input (:obj:`Tensor`)
            - reduction (:obj:`Tensor`)
            - reduction_override (:obj:`str`): choice of 'none', 'mean', 'sum', override the reduction type
            defined in __init__ function

            - normalizer_override (:obj:`float`): override the normalizer when reduction is 'mean'
        """
        reduction = reduction_override if reduction_override else self.reduction
        assert (normalizer_override is None or reduction == 'mean'), \
            f'normalizer is not allowed when reduction is {reduction}'
        loss = _Loss.__call__(self, input, target, reduction, normalizer=normalizer_override, **kwargs)
        return loss * self.loss_weight

    def forward(self, input, target, reduction, normalizer=None, **kwargs):
        raise NotImplementedError
