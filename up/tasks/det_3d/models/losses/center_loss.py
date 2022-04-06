import torch
from up.utils.general.registry_factory import LOSSES_REGISTRY
from up.tasks.det.models.losses.focal_loss import dynamic_normalizer
from up.models.losses.loss import BaseLoss, _reduce


@LOSSES_REGISTRY.register('center_focal_loss')
class CenterNetFocalLoss(BaseLoss):
    def __init__(self,
                 alpha=0.5,
                 gamma=2.0,
                 beta=4.0,
                 name='center_focal_loss',
                 reduction='mean',
                 loss_weight=1.0,
                 dynamic_normalizer=False,
                 activation_type='sigmoid',
                 ):
        BaseLoss.__init__(self, name=name, reduction=reduction, loss_weight=loss_weight)
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.loss_weight = loss_weight
        self.dynamic_normalizer = dynamic_normalizer
        self.activation_type = activation_type

    def forward(self, input, target, reduction, normalizer=None):
        normalizer = 1.0 if normalizer is None else normalizer
        normalizer = torch.Tensor([normalizer]).type_as(input).to(input.device)
        if self.dynamic_normalizer:
            normalizer = dynamic_normalizer(input, target, self.alpha, self.gamma)
        if self.activation_type == 'sigmoid':
            input = torch.clamp(input.sigmoid(), min=1e-4, max=1 - 1e-4)
        losses = self.center_focal_loss(input, target)
        losses = _reduce(losses, reduction, normalizer=normalizer)
        return losses

    def center_focal_loss(self, preds, target):
        zn = 1.0 - self.alpha
        zp = self.alpha
        pos_inds = target.eq(1).float()
        neg_inds = target.lt(1).float()
        neg_weights = torch.pow(1 - target, self.beta)

        # p = 1. / (1. + torch.exp(-preds))
        p = preds
        # (1 - p)**gamma * log(p)
        term1 = torch.pow((1 - p), self.gamma) * torch.log(p)
        # p**gamma * log(1 - p)
        term2 = torch.pow(p, self.gamma) * torch.log(1 - p)
        pos_loss = term1 * pos_inds * zp
        neg_loss = term2 * neg_weights * neg_inds * zn
        num_pos = pos_inds.float().sum()
        if num_pos == 0:
            losses = - neg_loss
        else:
            losses = - (pos_loss + neg_loss) / num_pos
        return losses
