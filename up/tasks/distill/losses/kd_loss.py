import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['CELoss', 'KDLoss', 'GKDLoss']


class CELoss(nn.Module):
    """
    Distilling the Knowledge in a Neural Network, NIPS2014.
    https://arxiv.org/pdf/1503.02531.pdf
    """
    def __init__(self, T=1):
        super().__init__()
        self.t = T

    def forward(self, s_preds, t_preds, **kwargs):
        loss = 0
        for s_pred, t_pred in zip(s_preds, t_preds):
            s = F.log_softmax(s_pred / self.t, dim=1)
            t = F.softmax(t_pred / self.t, dim=1)
            loss += torch.mean(torch.sum(- t * s, 1))
        return loss


class KDLoss(nn.Module):
    """
    Distilling the Knowledge in a Neural Network, NIPS2014.
    https://arxiv.org/pdf/1503.02531.pdf
    """
    def __init__(self, T=1):
        super().__init__()
        self.t = T

    def forward(self, s_preds, t_preds, **kwargs):
        loss = 0
        for s_pred, t_pred in zip(s_preds, t_preds):
            s = F.log_softmax(s_pred / self.t, dim=1)
            t = F.softmax(t_pred / self.t, dim=1)
            loss += F.kl_div(s, t, reduction='sum') * (self.t**2) / s_pred.shape[0]
        return loss


class GKDLoss(nn.Module):
    """
    Guided Knowledge Distillation Loss
    Use correctly classified soft labels to guide student.
    """

    def __init__(self, T=1):
        super().__init__()
        self.t = T

    def forward(self, s_preds, t_preds, **kwargs):
        label = kwargs['label']
        loss = 0
        for s_pred, t_pred in zip(s_preds, t_preds):
            s = F.log_softmax(s_pred / self.t, dim=1)
            t = F.softmax(t_pred / self.t, dim=1)
            t_argmax = torch.argmax(t, dim=1)
            mask = torch.eq(label, t_argmax).float()
            count = (mask[mask == 1]).size(0)
            mask = mask.unsqueeze(-1)
            correct_s = s.mul(mask)
            correct_t = t.mul(mask)
            correct_t[correct_t == 0.0] = 1.0

            loss += F.kl_div(correct_s, correct_t, reduction='sum') * (self.t**2) / count
        return loss
