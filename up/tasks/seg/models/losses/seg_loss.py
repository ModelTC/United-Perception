import torch
import torch.nn as nn
from torch.nn import functional as F
from up.utils.general.registry_factory import LOSSES_REGISTRY
from up.utils.general.log_helper import default_logger as logger

__all__ = ["CriterionOhem", "SegCrossEntropyLoss"]


@LOSSES_REGISTRY.register('seg_ohem')
class CriterionOhem(nn.Module):
    def __init__(self, aux_weight=0, thresh=0.7, min_kept=100000, ignore_index=255):
        super(CriterionOhem, self).__init__()
        self._aux_weight = aux_weight
        self._criterion1 = OhemCrossEntropy2dTensor(ignore_index, thresh, min_kept)
        self._criterion2 = OhemCrossEntropy2dTensor(ignore_index, thresh, min_kept)

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)
        if self._aux_weight > 0:  # require aux loss
            main_pred, aux_pred = preds
            main_h, main_w = main_pred.size(2), main_pred.size(3)
            aux_h, aux_w = aux_pred.size(2), aux_pred.size(3)
            assert len(preds) == 2 and main_h == aux_h and main_w == aux_w and main_h == h and main_w == w

            loss1 = self._criterion1(main_pred, target)
            loss2 = self._criterion2(aux_pred, target)
            loss = loss1 + self._aux_weight * loss2
        else:
            pred_h, pred_w = preds.size(2), preds.size(3)
            assert pred_h == h and pred_w == w
            loss = self._criterion1(preds, target)
        return loss


class OhemCrossEntropy2dTensor(nn.Module):
    """
        Ohem Cross Entropy Tensor Version
    """

    def __init__(self, ignore_index=255, thresh=0.7, min_kept=256,
                 use_weight=False):
        super(OhemCrossEntropy2dTensor, self).__init__()
        self.ignore_index = ignore_index
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        if use_weight:
            weight = torch.FloatTensor(
                [0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489,
                 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955,
                 1.0865, 1.1529, 1.0507])
            self.criterion = torch.nn.CrossEntropyLoss(reduction="elementwise_mean",
                                                       weight=weight,
                                                       ignore_index=ignore_index)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(reduction="elementwise_mean",
                                                       ignore_index=ignore_index)

    def forward(self, pred, target):
        b, c, h, w = pred.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_index)
        target = (target * valid_mask).long()
        num_valid = valid_mask.sum()
        prob = F.softmax(pred, dim=1)
        prob = (prob.transpose(0, 1)).reshape(c, -1)
        if self.min_kept > num_valid:
            logger.info('Labels: {}'.format(num_valid))
        elif num_valid > 0:
            prob = prob.masked_fill_(~valid_mask, 1)
            mask_prob = prob[
                target, torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                _, index = mask_prob.sort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
                kept_mask = mask_prob.le(threshold)
                target = target * kept_mask.long()
                valid_mask = valid_mask * kept_mask

        target = target.masked_fill_(~valid_mask, self.ignore_index)
        target = target.view(b, h, w)

        return self.criterion(pred, target)


@LOSSES_REGISTRY.register('seg_ce')
class SegCrossEntropyLoss(nn.Module):
    def __init__(self,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 ignore_index=255):
        super(SegCrossEntropyLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.cls_criterion = cross_entropy
        self.ignore_index = ignore_index

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function."""
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None
        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label.long(),
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            ignore_index=self.ignore_index)
        return loss_cls


def cross_entropy(pred,
                  label,
                  weight=None,
                  class_weight=None,
                  reduction='mean',
                  avg_factor=None,
                  ignore_index=255):
    """The wrapper function for :func:`F.cross_entropy`"""
    # class_weight is a manual rescaling weight given to each class.
    # If given, has to be a Tensor of size C element-wise losses
    loss = F.cross_entropy(
        pred,
        label,
        weight=class_weight,
        reduction='none',
        ignore_index=ignore_index)

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
        assert weight.dim() == loss.dim()
        if weight.dim() > 1:
            assert weight.size(1) == 1 or weight.size(1) == loss.size(1)
        loss = loss * weight
    if avg_factor is None:
        reduction_enum = F._Reduction.get_enum(reduction)
        if reduction_enum == 0:
            loss = loss
        elif reduction_enum == 1:
            loss = loss.mean()
        elif reduction_enum == 2:
            loss = loss.sum()
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')

    return loss
