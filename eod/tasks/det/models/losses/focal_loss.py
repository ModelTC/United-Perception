# Import from third library
import torch
import torch.nn.functional as F
import numpy as np

from .entropy_loss import GeneralizedCrossEntropyLoss
from eod.utils.general.registry_factory import LOSSES_REGISTRY
from eod.models.losses.loss import _reduce
from eod.extensions import (
    SigmoidFocalLossFunction,
    SoftmaxFocalLossFunction,
    CrossSigmoidFocalLossFunction
)


__all__ = ['QualityFocalLoss', 'SigmoidFocalLoss', 'dynamic_normalizer', 'CrossSigmoidFocalLoss']


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


@LOSSES_REGISTRY.register('softmax_focal_loss')
class SoftMaxFocalLoss(GeneralizedCrossEntropyLoss):
    def __init__(self,
                 gamma,
                 alpha,
                 num_classes,
                 init_prior,
                 name='softmax_focal_loss',
                 reduction='mean',
                 loss_weight=1.0,
                 ignore_index=-1,
                 ):
        """
        Arguments:
            - name (:obj:`str`): name of the loss function
            - reduction (:obj:`str`): reduction type, choice of mean, none, sum
            - loss_weight (:obj:`float`): loss weight
            - gamma (:obj:`float`): hyparam
            - alpha (:obj:`float`): hyparam
            - init_prior (:obj:`float`): init bias initialization
            - num_classes (:obj:`int`): num_classes total, 81 for coco
            - ignore index (:obj:`int`): ignore index in label
        """
        activation_type = 'softmax'
        GeneralizedCrossEntropyLoss.__init__(self,
                                             name=name,
                                             reduction=reduction,
                                             loss_weight=loss_weight,
                                             activation_type=activation_type,
                                             ignore_index=ignore_index)
        self.init_prior = init_prior
        self.num_channels = num_classes
        self.gamma = gamma
        self.alpha = alpha
        assert ignore_index == -1, 'only -1 is allowed for ignore index'

    def forward(self, input, target, reduction, normalizer=None):
        """
        Arguments:
            - input (FloatTenosor): [[M, N,]C]
            - target (LongTenosor): [[M, N]]
        """
        assert reduction != 'none', 'Not Supported none reduction yet'
        input = input.reshape(target.numel(), -1)
        target = target.reshape(-1).int()
        # normalizer always has a value because the API of `focal loss` need it.
        normalizer = 1.0 if normalizer is None else normalizer
        normalizer = torch.Tensor([normalizer]).type_as(input).to(input.device)
        loss = SoftmaxFocalLossFunction.apply(
            input, target, normalizer, self.gamma, self.alpha, self.num_channels)
        return loss


@LOSSES_REGISTRY.register('sigmoid_focal_loss')
class SigmoidFocalLoss(GeneralizedCrossEntropyLoss):
    def __init__(self,
                 num_classes,
                 alpha=0.25,
                 gamma=2.0,
                 init_prior=0.01,
                 name='sigmoid_focal_loss',
                 reduction='mean',
                 loss_weight=1.0,
                 ignore_index=-1,
                 dynamic_normalizer=False,
                 ):
        """
        Arguments:
            - name (:obj:`str`): name of the loss function
            - reduction (:obj:`str`): reduction type, choice of mean, none, sum
            - loss_weight (:obj:`float`): loss weight
            - gamma (:obj:`float`): hyparam
            - alpha (:obj:`float`): hyparam
            - init_prior (:obj:`float`): init bias initialization
            - num_classes (:obj:`int`): num_classes total, 81 for coco
            - ignore index (:obj:`int`): ignore index in label
            - dynamic_normalizer (:obj:`bool`): flag of using dynamic_normalizer
        """
        activation_type = 'sigmoid'
        GeneralizedCrossEntropyLoss.__init__(self,
                                             name=name,
                                             reduction=reduction,
                                             loss_weight=loss_weight,
                                             activation_type=activation_type,
                                             ignore_index=ignore_index)
        self.init_prior = init_prior
        self.num_channels = num_classes - 1
        self.gamma = gamma
        self.alpha = alpha
        self.dynamic_normalizer = dynamic_normalizer
        assert ignore_index == -1, 'only -1 is allowed for ignore index'

    def forward(self, input, target, reduction, normalizer=None):
        """
        Arguments:
            - input (FloatTenosor): [[M, N,]C]
            - target (LongTenosor): [[M, N]]
        """
        # assert reduction != 'none', 'Not Supported none reduction yet'
        input = input.reshape(target.numel(), -1)
        target = target.reshape(-1).int()
        normalizer = 1.0 if normalizer is None else normalizer
        normalizer = torch.Tensor([normalizer]).type_as(input).to(input.device)
        if self.dynamic_normalizer:
            normalizer = dynamic_normalizer(input, target, self.alpha, self.gamma)
        loss = SigmoidFocalLossFunction.apply(
            input, target, normalizer, self.gamma, self.alpha, self.num_channels, reduction)
        return loss


@LOSSES_REGISTRY.register('cross_sigmoid_focal_loss')
class CrossSigmoidFocalLoss(SigmoidFocalLoss):
    def __init__(self,
                 num_classes,
                 alpha,
                 gamma,
                 init_prior,
                 name='cross_sigmoid_focal_loss',
                 reduction='mean',
                 loss_weight=1.0,
                 ignore_index=-1,
                 dynamic_normalizer=False,
                 gt_class_to_avoid=None,
                 neg_targets=None):
        """
        Arguments:
            - name (:obj:`str`): name of the loss function
            - reduction (:obj:`str`): reduction type, choice of mean, none, sum
            - loss_weight (:obj:`float`): loss weight
            - gamma (:obj:`float`): hyparam
            - alpha (:obj:`float`): hyparam
            - init_prior (:obj:`float`): init bias initialization
            - num_classes (:obj:`int`): num_classes total, 81 for coco
            - ignore index (:obj:`int`): ignore index in label
            - gt_class_to_avoid (:obj: `list`): avoid label for gt_class
            - dynamic_normalizer (:obj:`bool`): flag of using dynamic_normalizer
        """
        SigmoidFocalLoss.__init__(
            self,
            num_classes,
            alpha,
            gamma,
            init_prior,
            name=name,
            reduction=reduction,
            loss_weight=loss_weight,
            ignore_index=ignore_index,
            dynamic_normalizer=dynamic_normalizer)
        if gt_class_to_avoid is None:
            # if gt_class_to avoid is not provided, we set -2 for no neg label to avoid.
            gt_class_to_avoid = [[-2]] * self.num_channels
        self.gt_class_to_avoid = self._convert2tensor(gt_class_to_avoid)
        self.neg_targets = neg_targets
        assert ignore_index == -1, 'only -1 is allowed for ignore index'

    def _convert2tensor(self, avoid_list):
        cls_num = len(avoid_list)
        data = []
        data.append(cls_num)
        for cls, neg in enumerate(avoid_list):
            data.append(1 + len(neg))
            data.append(cls + 1)
            data.extend(neg)
        data = np.require(data, dtype=np.int32)
        data = torch.from_numpy(data).cuda()
        return data

    def forward(self, input, target, reduction, normalizer=None):
        """
        Arguments:
            - input (FloatTenosor): [[M, N,]C]
            - target (LongTenosor): [[M, N]]
        """
        # assert reduction != 'none', 'Not Supported none reduction yet'
        input = input.reshape(target.numel(), -1)
        target = target.reshape(-1).int()
        normalizer = 1.0 if normalizer is None else normalizer
        normalizer = torch.Tensor([normalizer]).type_as(input).to(input.device)
        if self.dynamic_normalizer:
            normalizer = cross_dynamic_normalizer(input, target, self.alpha, self.gamma, self.neg_targets)
        return CrossSigmoidFocalLossFunction.apply(input, target, normalizer,
                                                   self.gamma, self.alpha,
                                                   self.num_channels,
                                                   self.gt_class_to_avoid,
                                                   reduction)


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


def cross_dynamic_normalizer(input, target, alpha, gamma, neg_target=None):
    def reduce_(tensor, gamma):
        return tensor.pow(gamma).sum()
    target = target.reshape(-1).long()
    input_p = input.detach().sigmoid()
    cls_num = input_p.size(1)
    pos_mask = torch.nonzero((target >= 1) & (target <= cls_num)).squeeze()
    valid_mask = input.new_zeros(input.size()).bool()
    valid_mask[pos_mask, :] = True
    for idx, nt in enumerate(neg_target):
        if isinstance(nt, list):
            for _nt in nt:
                valid_mask[torch.nonzero(_nt == target), idx] = True
        else:
            valid_mask[torch.nonzero(nt == target), idx] = True
    pos_normalizer = reduce_((1 - input_p[pos_mask, target[pos_mask] - 1]), gamma)
    neg_normalizer = reduce_(input_p[valid_mask], gamma) - reduce_(input_p[pos_mask, target[pos_mask] - 1], gamma)
    pos_normalizer *= alpha
    neg_normalizer *= 1 - alpha
    normalizer = torch.clamp(pos_normalizer + neg_normalizer, min=1)
    return normalizer
