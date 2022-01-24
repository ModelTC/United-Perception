# Import from third library
import torch
import torch.nn as nn

# Import from local
from up.utils.general.registry_factory import LOSSES_REGISTRY


@LOSSES_REGISTRY.register('ohem_loss')
class OnlineHardExampleMingingLoss(nn.Module):
    def __init__(self, batch_size, criterion='sum'):
        """
        :param: batch_size/topk -> int, number of valid samples per image
        :param: criterion -> str, way of mining method, choices are:
                - sum: choose topk by sum of cls and loc
                - cls: choose topk by cls loss
                - loc: choose topk by loc loss
                - respectively: choose topk respectively
        """
        super().__init__()
        self.batch_size = batch_size
        self.criterion = criterion

    def forward(self, cls_loss, loc_loss, image_batch):

        total_batch_size = self.batch_size * image_batch

        if cls_loss.dim() == 2:
            cls_loss = cls_loss.sum(dim=1)
        loc_loss = loc_loss.sum(dim=1)

        if self.criterion == 'respectively':
            loc_loss, _ = torch.sort(loc_loss, descending=True)
            cls_loss, _ = torch.sort(cls_loss, descending=True)
            if total_batch_size < loc_loss.size(0):
                loc_loss = loc_loss[:total_batch_size]
                cls_loss = cls_loss[:total_batch_size]
            return cls_loss.mean(), loc_loss.mean()
        else:
            if self.criterion == 'sum':
                loss = cls_loss + loc_loss
            elif self.criterion == 'cls':
                loss = cls_loss
            elif self.criterion == 'loc':
                loss = loc_loss
            else:
                raise NotImplementedError

            sorted_ohem_loss, idx = torch.sort(loss, descending=True)
            keep_num = min(sorted_ohem_loss.size(0), total_batch_size)
            if keep_num < sorted_ohem_loss.size(0):
                keep_idx_cuda = idx[:keep_num]
                cls_loss = cls_loss[keep_idx_cuda]
                loc_loss = loc_loss[keep_idx_cuda]

            cls_loss = cls_loss.mean()
            loc_loss = loc_loss.mean()
            return cls_loss, loc_loss


def ohem_loss(cls_loss, loc_loss, num_img, cfg):
    """
    Arguments:
        - batch_size (int): number of sampled rois for bbox head training
        - cls_loss (FloatTenosr): [R[, C]]
        - loc_loss (FloatTensor): [R, 4]
        - num_img (int):
        - cfg (dict)

    Returns:
        cls_loss, loc_loss (FloatTensor)
    """
    ohem = OnlineHardExampleMingingLoss(cfg['batch_size'], cfg.get('criterion', 'sum'))
    return ohem(cls_loss, loc_loss, num_img)
