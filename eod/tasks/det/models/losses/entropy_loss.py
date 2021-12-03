# Import from third library
import torch
import torch.nn.functional as F

# Import from local
from eod.models.losses.loss import BaseLoss, _reduce
from eod.utils.general.registry_factory import LOSSES_REGISTRY

__all__ = [
    'GeneralizedCrossEntropyLoss',
    'SoftMaxCrossEntropyLoss',
    'SigmoidCrossEntropyLossExpandLabel',
    'SigmoidCrossEntropyLoss',
    'BinaryCrossEntropyLoss',
    'apply_class_activation'
]


class GeneralizedCrossEntropyLoss(BaseLoss):
    def __init__(self,
                 name='generalized_cross_entropy_loss',
                 reduction='none',
                 loss_weight=1.0,
                 activation_type='softmax',
                 ignore_index=-1,):
        BaseLoss.__init__(self,
                          name=name,
                          reduction=reduction,
                          loss_weight=loss_weight)
        self.activation_type = activation_type
        self.ignore_index = ignore_index


@LOSSES_REGISTRY.register('softmax_cross_entropy')
class SoftMaxCrossEntropyLoss(GeneralizedCrossEntropyLoss):
    def __init__(self,
                 class_dim,
                 name='softmax_cross_entropy',
                 reduction='mean',
                 loss_weight=1.0,
                 ignore_index=-1):
        """
        Arguments:
            - name (:obj:`str`): name of the loss function
            - class_dim (:obj:`int`): choices are [1, -1]
            - reduction (:obj:`str`): reduction type, choice of mean, none, sum
            - loss_weight (:obj:`float`): loss weight
            - ignore index (:obj:`int`): ignore index in label
        """
        activation_type = 'softmax'
        assert class_dim in [1, -1], class_dim
        self.class_dim = class_dim
        GeneralizedCrossEntropyLoss.__init__(self,
                                             name=name,
                                             reduction=reduction,
                                             loss_weight=loss_weight,
                                             activation_type=activation_type,
                                             ignore_index=ignore_index)

    def forward(self, input, target, reduction, normalizer=None):
        """Refer to `official documents <https://pytorch.org/docs/1.3.1/nn.functional.html#torch.nn.functional.cross_entropy>`_
        """
        if self.class_dim == -1:
            input = input.reshape(target.numel(), -1)
            target = target.reshape(-1)
        else:
            N, C = input.shape[:2]
            D = input.numel() // N // C
            input = input.reshape(N, C, D)
            target = target.reshape(N, D)
        if normalizer is None:
            return F.cross_entropy(input, target, reduction=reduction, ignore_index=self.ignore_index)
        else:
            loss = F.cross_entropy(input, target, reduction='none', ignore_index=self.ignore_index)
            return _reduce(loss, reduction, normalizer=normalizer)


@LOSSES_REGISTRY.register('sigmoid_cross_entropy')
class SigmoidCrossEntropyLoss(GeneralizedCrossEntropyLoss):
    def __init__(self,
                 name='sigmoid_cross_entropy',
                 reduction='mean',
                 loss_weight=1.0,
                 ignore_index=-1,
                 init_prior=0.5):
        """
        Arguments:
            - name (:obj:`str`): name of the loss function
            - reduction (:obj:`str`): reduction type, choice of mean, none, sum
            - loss_weight (:obj:`float`): loss weight
            - ignore index (:obj:`int`): ignore index in label
        """
        activation_type = 'sigmoid'
        GeneralizedCrossEntropyLoss.__init__(self,
                                             name=name,
                                             reduction=reduction,
                                             loss_weight=loss_weight,
                                             activation_type=activation_type,
                                             ignore_index=ignore_index)
        self.init_prior = init_prior

    def forward(self, input, target, reduction, normalizer=None):
        input = input.reshape(-1)
        target = target.reshape(-1)
        sample_mask = (target != self.ignore_index)
        loss = F.binary_cross_entropy_with_logits(input, target.float(), reduction='none')

        # Don't multiply sample mask if no sample is ignored
        # this is for performance optimization
        if (~sample_mask).sum() > 0:
            loss = loss * sample_mask.float()

        if normalizer is None:
            normalizer = max(1, sample_mask.sum().item())

        return _reduce(loss, reduction, normalizer=normalizer)


@LOSSES_REGISTRY.register('binary_cross_entropy')
class BinaryCrossEntropyLoss(GeneralizedCrossEntropyLoss):
    def __init__(self,
                 name='binary_cross_entropy',
                 reduction='mean',
                 loss_weight=1.0,
                 ignore_index=-1,
                 init_prior=0.5):
        """
        Arguments:
            - name (:obj:`str`): name of the loss function
            - reduction (:obj:`str`): reduction type, choice of mean, none, sum
            - loss_weight (:obj:`float`): loss weight
            - ignore index (:obj:`int`): ignore index in label
        """
        activation_type = 'none'
        GeneralizedCrossEntropyLoss.__init__(self,
                                             name=name,
                                             reduction=reduction,
                                             loss_weight=loss_weight,
                                             activation_type=activation_type,
                                             ignore_index=ignore_index)
        self.init_prior = init_prior

    def forward(self, input, target, reduction, normalizer=None):
        input = input
        target = target
        sample_mask = (target != self.ignore_index)
        loss = F.binary_cross_entropy(input, target, reduction='none')

        # Don't multiply sample mask if no sample is ignored
        # this is for performance optimization
        if (~sample_mask).sum() > 0:
            loss = loss * sample_mask.float()

        if normalizer is None:
            normalizer = max(1, sample_mask.sum().item())

        return _reduce(loss, reduction, normalizer=normalizer)


@LOSSES_REGISTRY.register('sigmoid_cross_entropy_with_softmax_label')
class SigmoidCrossEntropyLossExpandLabel(SigmoidCrossEntropyLoss):
    def __init__(self,
                 name='sigmoid_cross_entropy_expand_label',
                 reduction='mean',
                 loss_weight=1.0,
                 ignore_index=-1,
                 init_prior=0.5):
        activation_type = 'sigmoid'
        GeneralizedCrossEntropyLoss.__init__(self,
                                             name=name,
                                             reduction=reduction,
                                             loss_weight=loss_weight,
                                             activation_type=activation_type,
                                             ignore_index=ignore_index)
        self.init_prior = init_prior

    def forward(self, input, target, reduction, normalizer=None):
        """
        Arguments:
            - input (:obj:`FloatTensor`): [[M,N],C]
            - target (:obj:`LongTenosr`): [[M,N]]
        """
        assert input.dim() == 2 and target.dim() == 1
        expand_label_target = target.new_zeros((input.shape[0], input.shape[1] + 1))
        expand_label_target[torch.arange(input.size(0)), target] = 1
        expand_label_target = expand_label_target[:, 1:]

        return super(SigmoidCrossEntropyLossExpandLabel, self).forward(input, expand_label_target,
                                                                       reduction, normalizer=normalizer)


def apply_class_activation(input, activation, remove_background_channel=True):
    """
    """
    if activation == 'sigmoid':
        results = input.sigmoid()
    elif activation == 'softmax':
        results = F.softmax(input, dim=-1)
    else:
        raise NotImplementedError('{} is not supported'.format(activation))
    return results
