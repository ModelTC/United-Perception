import torch
import torch.nn as nn
import torch.nn.functional as F
from eod.utils.general.registry_factory import LOSSES_REGISTRY


__all__ = ['KLLoss']


class KLLoss(nn.Module):
    """
    KL Divergence loss
    """
    def __init__(self, norm='softmax'):
        super(KLLoss, self).__init__()
        self.norm = norm

    def forward(self, s_features, t_features, **kwargs):
        loss = 0
        for s, t in zip(s_features, t_features):
            loss += self.kl(s, t)
        return loss

    def kl(self, pred_feas, target_feas):
        crit = nn.KLDivLoss(reduction='batchmean')
        relu = nn.ReLU()
        s = relu(pred_feas)
        t = relu(target_feas)
        if self.norm == 'softmax':
            s = F.log_softmax(s, dim=1)
            t = F.softmax(t, dim=1)
            t.detach_()
            # sum(y*(logy-logx))
            loss = crit(s, t)
        elif self.norm == 'l2':
            # sum(y*log(y/x))
            # use KLDivLoss for l2 norm may cause overflow
            loss = torch.sum(t / torch.sum(t) * (torch.log((t / torch.sum(t) + 1e-6) / (s / torch.sum(s) + 1e-6))))
        else:
            print('Invalid Normalization Function: {}'.format(self.norm))
            return None

        return loss
