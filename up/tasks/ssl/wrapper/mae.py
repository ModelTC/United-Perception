import torch.nn as nn
from up.utils.general.registry_factory import (
    MODULE_WRAPPER_REGISTRY
)


@MODULE_WRAPPER_REGISTRY.register('MAE')
class MAE(nn.Module):
    """
    Build a MAE model.
    """
    def __init__(self, model):
        """
        model: encoder-decoder model (default: vit-based model)
        """
        super().__init__()
        self.model = model

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, input):
        return self.model(input)
