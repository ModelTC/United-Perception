import torch
import torch.nn as nn


class L2Loss(nn.Module):
    """
    L2 Loss
    """
    def __init__(self, normalize=True, div_element=False):
        super().__init__()
        self.normalize = normalize
        self.div_element = div_element

    def forward(self, s_features, t_features, **kwargs):
        total_loss = 0
        for s, t in zip(s_features, t_features):
            if self.normalize:
                s = self.normalize_feature(s)
                t = self.normalize_feature(t)
            loss = torch.sum(torch.pow(torch.add(s, -1, t), 2))
            if self.div_element:
                loss = loss / s.numel()
            else:
                loss = loss / s.size(0)
            total_loss += loss
        return total_loss / len(s_features)

    def normalize_feature(self, x, mult=1.0):
        x = x.view(x.size(0), -1)
        return x / x.norm(2, dim=1, keepdim=True) * mult
