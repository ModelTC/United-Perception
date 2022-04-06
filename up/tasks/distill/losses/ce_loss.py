import torch
import torch.nn as nn
import torch.nn.functional as F

# from up.utils.general.log_helper import default_logger as logger
from up.utils.general.registry_factory import MIMIC_LOSS_REGISTRY


@MIMIC_LOSS_REGISTRY.register('ce_loss')
class CELoss(nn.Module):
    """
    Distilling the Knowledge in a Neural Network, NIPS2014.
    https://arxiv.org/pdf/1503.02531.pdf
    """
    def __init__(self, T=1, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.t = T

    def forward(self, s_preds, t_preds, **kwargs):
        loss = 0
        for s_pred, t_pred in zip(s_preds, t_preds):
            s = F.log_softmax(s_pred / self.t, dim=1)
            t = F.softmax(t_pred / self.t, dim=1)
            loss += torch.mean(torch.sum(- t * s, 1))
        return loss * self.loss_weight


@MIMIC_LOSS_REGISTRY.register('bce_loss')
class BCELoss(nn.Module):
    """
    BCE Loss
    """
    def __init__(self, T=1.0, batch_mean=False, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.T = T
        self.batch_mean = batch_mean

    def forward(self, pred_s, pred_t, masks=None):
        assert isinstance(pred_s, list) and isinstance(pred_t, list), 'preds must be list!'
        if masks is not None:
            assert isinstance(masks, list) and len(masks) == len(pred_s), 'masks must be consistent with preds!'
        else:
            masks = [s.new_ones(s.shape).float().detach() for s in pred_s]

        total_loss = 0
        for idx, (s, t) in enumerate(zip(pred_s, pred_t)):
            loss = torch.sum(F.binary_cross_entropy(s, t, reduction='none') * masks[idx])
            if self.batch_mean:
                loss = loss / s.size(0)
            elif masks[idx].sum() != 0:
                loss = loss / masks[idx].sum()
            total_loss += loss
        return total_loss * self.loss_weight
