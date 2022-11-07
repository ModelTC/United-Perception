import torch
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import math
from up.utils.general.registry_factory import LOSSES_REGISTRY
from up.utils.env.dist_helper import allreduce


__all__ = ['LabelSmoothCELoss', 'BCE_LOSS', 'CrossEntropyLoss', 'SoftTargetCrossEntropy']


@LOSSES_REGISTRY.register('label_smooth_ce')
class LabelSmoothCELoss(_Loss):
    def __init__(self, smooth_ratio, num_classes, loss_weight=1.0):
        super(LabelSmoothCELoss, self).__init__()
        self.smooth_ratio = smooth_ratio
        self.v = self.smooth_ratio / num_classes
        self.loss_weight = loss_weight

    def forward(self, input, label):
        if label.dim() == 1:
            one_hot = torch.zeros_like(input)
            one_hot.fill_(self.v)
            y = label.to(torch.long).view(-1, 1)
            one_hot.scatter_(1, y, 1 - self.smooth_ratio + self.v)
        elif label.dim() > 1:
            one_hot = label.clone()
            one_hot -= one_hot * (self.smooth_ratio - self.v)
            one_hot[one_hot == 0.] += self.v
        loss = - torch.sum(F.log_softmax(input, 1) * (one_hot.detach())) / input.size(0)

        return loss * self.loss_weight


@LOSSES_REGISTRY.register('bce')
class BCE_LOSS(_Loss):
    def __init__(self, loss_weight=1.0, bias=False):
        super().__init__()
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        self.loss_weight = loss_weight
        self.bias = bias

    def forward(self, input, label):
        C = input.size(1) if self.bias else 1
        if label.dim() == 1:
            one_hot = torch.zeros_like(input).cuda()
            label = label.reshape(one_hot.shape[0], 1)
            one_hot.scatter_(1, label, 1)
            loss = self.bce_loss(input, one_hot)
        elif label.dim() > 1:
            label = label.float()
            loss = self.bce_loss(input - math.log(C), label) * C
        return loss.mean() * self.loss_weight


@LOSSES_REGISTRY.register('ce')
class CrossEntropyLoss(_Loss):
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.loss_weight = loss_weight

    def forward(self, input, label):
        loss = self.ce_loss(input, label)
        return loss * self.loss_weight


@LOSSES_REGISTRY.register('softmax_eql')
class SoftmaxEQLLoss(_Loss):
    def __init__(self, num_classes, indicator='pos', loss_weight=1.0, tau=1.0, eps=1e-4):
        super(SoftmaxEQLLoss, self).__init__()
        self.loss_weight = loss_weight
        self.num_classes = num_classes
        self.tau = tau
        self.eps = eps

        assert indicator in ['pos', 'neg', 'pos_and_neg'], 'Wrong indicator type!'
        self.indicator = indicator

        # initial variables
        self.register_buffer('pos_grad', torch.zeros(num_classes))
        self.register_buffer('neg_grad', torch.zeros(num_classes))
        self.register_buffer('pos_neg', torch.ones(num_classes))

    def forward(self, input, label):
        if self.indicator == 'pos':
            indicator = self.pos_grad.detach()
        elif self.indicator == 'neg':
            indicator = self.neg_grad.detach()
        elif self.indicator == 'pos_and_neg':
            indicator = self.pos_neg.detach() + self.neg_grad.detach()
        else:
            raise NotImplementedError

        one_hot = F.one_hot(label, self.num_classes)
        self.targets = one_hot.detach()

        matrix = indicator[None, :].clamp(min=self.eps) / indicator[:, None].clamp(min=self.eps)
        factor = matrix[label.long(), :].pow(self.tau)

        cls_score = input + (factor.log() * (1 - one_hot.detach()))
        loss = F.cross_entropy(cls_score, label)
        return loss * self.loss_weight

    def collect_grad(self, grad):
        grad = torch.abs(grad)
        pos_grad = torch.sum(grad * self.targets, dim=0)
        neg_grad = torch.sum(grad * (1 - self.targets), dim=0)

        allreduce(pos_grad)
        allreduce(neg_grad)

        self.pos_grad += pos_grad
        self.neg_grad += neg_grad
        self.pos_neg = self.pos_grad / (self.neg_grad + 1e-10)


@LOSSES_REGISTRY.register('softmax_eqlv2')
class SoftmaxEQLV2Loss(_Loss):
    def __init__(self, num_classes, indicator='pos', loss_weight=1.0, tau=1.0, eps=1e-4):
        super(SoftmaxEQLV2Loss, self).__init__()
        self.loss_weight = loss_weight
        self.num_classes = num_classes
        self.tau = tau
        self.eps = eps

        assert indicator in ['pos', 'neg', 'pos_and_neg'], 'Wrong indicator type!'
        self.indicator = indicator

        # initial variables
        self.register_buffer('pos_grad', torch.zeros(num_classes))
        self.register_buffer('neg_grad', torch.zeros(num_classes))
        self.register_buffer('pos_neg', torch.ones(num_classes))

    def forward(self, input, label):
        if self.indicator == 'pos':
            indicator = self.pos_grad.detach()
        elif self.indicator == 'neg':
            indicator = self.neg_grad.detach()
        elif self.indicator == 'pos_and_neg':
            indicator = self.pos_neg.detach() + self.neg_grad.detach()
        else:
            raise NotImplementedError

        if label.dim() == 1:
            one_hot = F.one_hot(label, self.num_classes)
        else:
            one_hot = label.clone()
        self.targets = one_hot.detach()

        indicator = indicator / (indicator.sum() + self.eps)
        indicator = (indicator ** self.tau + 1e-9).log()
        cls_score = input + indicator[None, :]
        loss = F.cross_entropy(cls_score, label)
        return loss * self.loss_weight

    def collect_grad(self, grad):
        grad = torch.abs(grad)
        pos_grad = torch.sum(grad * self.targets, dim=0)
        neg_grad = torch.sum(grad * (1 - self.targets), dim=0)

        allreduce(pos_grad)
        allreduce(neg_grad)

        self.pos_grad += pos_grad
        self.neg_grad += neg_grad
        self.pos_neg = self.pos_grad / (self.neg_grad + 1e-10)


@LOSSES_REGISTRY.register('stce')
class SoftTargetCrossEntropy(_Loss):

    def __init__(self, loss_weight=1.0):
        super(SoftTargetCrossEntropy, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, input, label) -> torch.Tensor:
        loss = torch.sum(-label * F.log_softmax(input, dim=-1), dim=-1)
        return loss.mean() * self.loss_weight
