import torch
import torch.nn as nn

# from up.utils.general.log_helper import default_logger as logger
from up.utils.general.registry_factory import MIMIC_LOSS_REGISTRY


@MIMIC_LOSS_REGISTRY.register('l2_loss')
class L2Loss(nn.Module):
    """
    L2 Loss
    """
    def __init__(self, feat_norm=False, batch_mean=False, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.feat_norm = feat_norm
        self.batch_mean = batch_mean

    def forward(self, s_features, t_features, masks=None):
        assert isinstance(s_features, list) and isinstance(t_features, list), 'features must be list!'
        if masks is not None:
            assert isinstance(masks, list) and len(masks) == len(s_features), 'masks must be consistent with features!'
        else:
            masks = [s.new_ones(s.shape).float().detach() for s in s_features]
        total_loss = 0
        for idx, (s, t) in enumerate(zip(s_features, t_features)):
            if self.feat_norm:
                s = self.normalize_feature(s)
                t = self.normalize_feature(t)
                masks[idx] = masks[idx].view(s.size(0), -1)
            loss = torch.sum(torch.pow(torch.add(s, -1, t), 2) * masks[idx])
            if self.batch_mean:
                loss = loss / s.size(0)
            elif masks[idx].sum() != 0:
                loss = loss / masks[idx].sum()
            total_loss += loss
        return total_loss * self.loss_weight

    def normalize_feature(self, x, mult=1.0):
        x = x.view(x.size(0), -1)
        return x / x.norm(2, dim=1, keepdim=True) * mult
