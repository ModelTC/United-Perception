import torch
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import math
from up.utils.general.registry_factory import LOSSES_REGISTRY


__all__ = ['MoCoLoss', 'SimSiamLoss']


@LOSSES_REGISTRY.register('loss_moco')
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


@LOSSES_REGISTRY.register('loss_simsiam')
class SimSiamLoss(_Loss):
    def __init__(self, loss_weight=1.0):
        super(SimSiamLoss, self).__init__()
        self.loss_weight = loss_weight
        self.cos_loss = torch.nn.CosineSimilarity(dim=1)

    def forward(self, input):
        p1, p2 = input['p1'], input['p2']
        z1, z2 = input['z1'], input['z2']
        loss = -(self.cos_loss(p1, z2).mean() 
               + self.cos_loss(p2, z1).mean()) * 0.5

        return loss * self.loss_weight

