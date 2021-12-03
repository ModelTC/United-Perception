import torch
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import math
from eod.utils.general.registry_factory import LOSSES_REGISTRY


__all__ = ['LabelSmoothCELoss', 'BCE_LOSS']


@LOSSES_REGISTRY.register('label_smooth_ce')
class LabelSmoothCELoss(_Loss):
    def __init__(self, smooth_ratio, num_classes):
        super(LabelSmoothCELoss, self).__init__()
        self.smooth_ratio = smooth_ratio
        self.v = self.smooth_ratio / num_classes

    def forward(self, input, label):
        one_hot = torch.zeros_like(input)
        one_hot.fill_(self.v)
        y = label.to(torch.long).view(-1, 1)
        one_hot.scatter_(1, y, 1 - self.smooth_ratio + self.v)

        loss = - torch.sum(F.log_softmax(input, 1) * (one_hot.detach())) / input.size(0)
        return loss


@LOSSES_REGISTRY.register('bce')
class BCE_LOSS(_Loss):
    def __init__(self):
        super().__init__()
        self.bce_loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, input, label):
        one_hot = torch.zeros_like(input).cuda()
        C = input.size(1)
        label = label.reshape(one_hot.shape[0], 1)
        one_hot.scatter_(1, label, 1)
        loss = self.bce_loss(input - math.log(C), one_hot) * C
        return loss


LOSSES_REGISTRY.register('ce', torch.nn.CrossEntropyLoss)
