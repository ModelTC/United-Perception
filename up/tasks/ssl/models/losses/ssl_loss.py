import torch
from torch.nn.modules.loss import _Loss
from up.utils.env.dist_helper import env
from up.utils.general.registry_factory import LOSSES_REGISTRY


__all__ = ['MoCoLoss', 'ContrastiveLoss', 'SimCLRLoss', 'SimSiamLoss']


@LOSSES_REGISTRY.register('moco_loss')
class MoCoLoss(_Loss):
    def __init__(self, loss_weight=1.0):
        super(MoCoLoss, self).__init__()
        self.loss_weight = loss_weight
        self.ce_loss = torch.nn.CrossEntropyLoss()

    def forward(self, input):
        logits = input['logits']
        if 'gt' in input:
            label = input['gt']
        else:
            label = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        loss = self.ce_loss(logits, label)
        return loss * self.loss_weight


@LOSSES_REGISTRY.register('contrastive_loss')
class ContrastiveLoss(_Loss):
    def __init__(self, tau, loss_weight=1.0):
        super().__init__()
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.loss_weight = loss_weight
        self.tau = tau

    def forward(self, input):
        rank = env.rank
        logits = input['logits']
        N = logits.shape[1]
        label_1 = (torch.arange(N, dtype=torch.long) + N * rank).cuda()
        label_2 = (torch.arange(N, dtype=torch.long) + N * rank).cuda()

        logit_1, logit_2 = logits[0, :], logits[1, :]

        loss_1 = self.ce_loss(logit_1, label_1)
        loss_2 = self.ce_loss(logit_2, label_2)
        loss = loss_1 + loss_2
        loss = 2 * self.tau * loss
        return loss * self.loss_weight


@LOSSES_REGISTRY.register('simclr_loss')
class SimCLRLoss(_Loss):
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.loss_weight = loss_weight

    def forward(self, input):
        logits = input['logits']
        label = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        loss = self.ce_loss(logits, label)
        return loss * self.loss_weight


@LOSSES_REGISTRY.register('simsiam_loss')
class SimSiamLoss(_Loss):
    def __init__(self, loss_weight=1.0):
        super(SimSiamLoss, self).__init__()
        self.loss_weight = loss_weight
        self.cos_loss = torch.nn.CosineSimilarity(dim=1)

    def forward(self, input):
        p1, p2 = input['p1'], input['p2']
        z1, z2 = input['z1'], input['z2']
        loss = -(self.cos_loss(p1, z2).mean() + self.cos_loss(p2, z1).mean()) * 0.5

        return loss * self.loss_weight


@LOSSES_REGISTRY.register('mae_loss')
class MAELoss(_Loss):
    def __init__(self, loss_weight=1.0, norm_pix_loss=False):
        super(MAELoss, self).__init__()
        self.norm_pix_loss = norm_pix_loss
        self.loss_weight = loss_weight

    def forward(self, input):
        """
        target: [N, L, p*p*3]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = input['target']
        pred = input['pred']
        mask = input['mask']
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss * self.loss_weight
