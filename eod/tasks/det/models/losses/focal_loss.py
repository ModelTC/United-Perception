# Import from third library
import torch
import torch.nn.functional as F


from .entropy_loss import GeneralizedCrossEntropyLoss
from eod.utils.general.registry_factory import LOSSES_REGISTRY
from eod.models.losses.loss import _reduce


__all__ = ['QualityFocalLoss', 'SigmoidFocalLoss', 'dynamic_normalizer']


@LOSSES_REGISTRY.register('quality_focal_loss')
class QualityFocalLoss(GeneralizedCrossEntropyLoss):
    """
    Quality focal loss: https://arxiv.org/abs/2006.04388,
    """
    def __init__(self,
                 gamma,
                 init_prior=0.01,
                 name='quality_focal_loss',
                 reduction='mean',
                 loss_weight=1.0,
                 ignore_index=-1,
                 dynamic_normalizer=False,
                 use_sigmoid=True):
        """
        Arguments:
            - name (:obj:`str`): name of the loss function
            - reduction (:obj:`str`): reduction type, choice of mean, none, sum
            - loss_weight (:obj:`float`): loss weight
            - gamma (:obj:`float`): hyparam
            - init_prior (:obj:`float`): init bias initialization
            - num_classes (:obj:`int`): num_classes total, 81 for coco
            - ignore index (:obj:`int`): ignore index in label
            - dynamic_normalizer (:obj:`bool`): flag of using dynamic_normalizer
        """
        self.use_sigmoid = use_sigmoid
        activation_type = 'sigmoid' if use_sigmoid else 'softmax'
        GeneralizedCrossEntropyLoss.__init__(self,
                                             name=name,
                                             reduction=reduction,
                                             loss_weight=loss_weight,
                                             activation_type=activation_type,
                                             ignore_index=ignore_index)
        self.init_prior = init_prior
        self.gamma = gamma
        self.dynamic_normalizer = dynamic_normalizer
        assert ignore_index == -1, 'only -1 is allowed for ignore index'

    def forward(self, input, target, reduction, normalizer=None, scores=None):
        """
        Arguments:
            - input (FloatTenosor): [[M, N,]C]
            - target (LongTenosor): [[M, N]]
        """
        assert reduction != 'none', 'Not Supported none reduction yet'
        input = input.reshape(target.numel(), -1)
        target = target.reshape(-1).int()
        sample_mask = torch.nonzero(target != self.ignore_index).reshape(-1)
        scores = scores.reshape(-1)
        normalizer = 1.0 if normalizer is None else normalizer
        normalizer = torch.Tensor([normalizer]).type_as(input).to(input.device)
        if self.dynamic_normalizer:
            normalizer = dynamic_normalizer(input, target, 0.5, self.gamma)
        # add for gflv2
        if self.use_sigmoid:
            func = F.binary_cross_entropy_with_logits
        else:
            func = F.binary_cross_entropy
        if sample_mask.size(0) != target.size(0):
            scores = scores[sample_mask]
            input = input[sample_mask, :]
            target = target[sample_mask]

        # loss of negative samples
        input_sigmoid = input.sigmoid() if self.use_sigmoid else input
        scale_factor = input_sigmoid
        loss = func(input, input.new_zeros(input.shape), reduction='none') * scale_factor.pow(self.gamma)

        # loss of positive samples
        pos_inds = (target > 0).nonzero().squeeze(1)
        pos_target = target[pos_inds].long() - 1 if self.use_sigmoid else target[pos_inds].long()
        # positive samples are supervised by bbox quality (IoU) score
        scale_factor = scores[pos_inds] - input_sigmoid[pos_inds, pos_target]
        loss[pos_inds, pos_target] = func(
            input[pos_inds, pos_target], scores[pos_inds], reduction='none') * scale_factor.abs().pow(self.gamma)

        return _reduce(loss, reduction=reduction, normalizer=normalizer)


@LOSSES_REGISTRY.register('sigmoid_focal_loss')
class SigmoidFocalLoss(GeneralizedCrossEntropyLoss):
    """
    Quality focal loss: https://arxiv.org/abs/2006.04388,
    """
    def __init__(self,
                 alpha=0.25,
                 gamma=2.0,
                 init_prior=0.01,
                 name='sigmoid_focal_loss',
                 reduction='mean',
                 loss_weight=1.0,
                 ignore_index=-1,
                 dynamic_normalizer=False,
                 use_sigmoid=True):
        """
        Arguments:
            - name (:obj:`str`): name of the loss function
            - reduction (:obj:`str`): reduction type, choice of mean, none, sum
            - loss_weight (:obj:`float`): loss weight
            - gamma (:obj:`float`): hyparam
            - init_prior (:obj:`float`): init bias initialization
            - num_classes (:obj:`int`): num_classes total, 81 for coco
            - ignore index (:obj:`int`): ignore index in label
            - dynamic_normalizer (:obj:`bool`): flag of using dynamic_normalizer
        """
        self.use_sigmoid = use_sigmoid
        activation_type = 'sigmoid' if use_sigmoid else 'softmax'
        GeneralizedCrossEntropyLoss.__init__(self,
                                             name=name,
                                             reduction=reduction,
                                             loss_weight=loss_weight,
                                             activation_type=activation_type,
                                             ignore_index=ignore_index)
        self.init_prior = init_prior
        self.gamma = gamma
        self.alpha = alpha
        self.dynamic_normalizer = dynamic_normalizer
        assert ignore_index == -1, 'only -1 is allowed for ignore index'

    def generate_onehot_target(self, input, target):
        # N * C
        N, C = input.size()
        onehot_target = input.new_zeros(N, C + 1)
        onehot_target[torch.arange(N), target] = 1
        return onehot_target[:, 1:]

    def forward(self, input, target, reduction, normalizer=None):
        """
        Arguments:
            - input (FloatTenosor): [[M, N,]C]
            - target (LongTenosor): [[M, N]]
        """
        assert reduction != 'none', 'Not Supported none reduction yet'
        input = input.reshape(target.numel(), -1)
        target = target.reshape(-1)
        normalizer = 1.0 if normalizer is None else normalizer
        normalizer = torch.Tensor([normalizer]).type_as(input).to(input.device)
        if self.dynamic_normalizer:
            normalizer = dynamic_normalizer(input, target.int(), self.alpha, self.gamma)
        sample_mask = torch.nonzero(target != self.ignore_index).reshape(-1)
        if sample_mask.size(0) != target.size(0):
            input = input[sample_mask, :]
            target = target[sample_mask]
        target = self.generate_onehot_target(input, target)
        p = torch.sigmoid(input)
        ce_loss = F.binary_cross_entropy_with_logits(input, target, reduction="none")
        p_t = p * target + (1 - p) * (1 - target)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
            loss = alpha_t * loss

        return _reduce(loss, reduction=reduction, normalizer=normalizer)


def dynamic_normalizer(input, target, alpha, gamma):
    def reduce_(tensor, gamma):
        return tensor.pow(gamma).sum()
    target = target.reshape(-1).long()
    input_p = input.detach().sigmoid()
    pos_mask = torch.nonzero(target >= 1).squeeze()
    valid_mask = torch.nonzero(target >= 0).squeeze()
    pos_normalizer = reduce_((1 - input_p[pos_mask, target[pos_mask] - 1]), gamma)
    neg_normalizer = reduce_(input_p[valid_mask], gamma) - reduce_(input_p[pos_mask, target[pos_mask] - 1], gamma)
    pos_normalizer *= alpha
    neg_normalizer *= 1 - alpha
    normalizer = torch.clamp(pos_normalizer + neg_normalizer, min=1)
    return normalizer
