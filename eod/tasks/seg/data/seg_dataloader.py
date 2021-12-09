from torch.utils.data import DataLoader
import torch
from easydict import EasyDict

from eod.utils.general.registry_factory import DATALOADER_REGISTRY
from eod.data.samplers.batch_sampler import InfiniteBatchSampler


__all__ = ['SegDataLoader']


@DATALOADER_REGISTRY.register('seg_base')
class SegDataLoader(DataLoader):

    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None,
                 num_workers=0, pin_memory=False, drop_last=False,
                 ):
        super(SegDataLoader, self).__init__(
            dataset, batch_size, shuffle, sampler, batch_sampler, num_workers,
            self._collate_fn, pin_memory, drop_last)

    def _collate_fn(self, batch):
        images = torch.stack([_.image for _ in batch])
        gt_seg = [_.get('gt_seg', None) for _ in batch]
        image_info = [_.get('image_info', None) for _ in batch]
        output = EasyDict({
            'image': images,
            'image_info': image_info
        })
        if gt_seg[0] is not None:
            gt_seg = torch.stack(gt_seg)
        output.gt_seg = gt_seg
        return output

    def get_epoch_size(self):
        if isinstance(self.batch_sampler, InfiniteBatchSampler):
            return len(self.batch_sampler.batch_sampler)   # training
        return len(self.batch_sampler)
