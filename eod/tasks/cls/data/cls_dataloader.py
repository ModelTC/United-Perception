from torch.utils.data import DataLoader
import torch
import numpy as np
from easydict import EasyDict

from eod.utils.general.registry_factory import DATALOADER_REGISTRY
from eod.data.samplers.batch_sampler import InfiniteBatchSampler


__all__ = ['ClassDataLoader']


@DATALOADER_REGISTRY.register('cls_base')
class ClassDataLoader(DataLoader):

    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None,
                 num_workers=0, pin_memory=False, drop_last=False,
                 ):
        super(ClassDataLoader, self).__init__(
            dataset, batch_size, shuffle, sampler, batch_sampler, num_workers,
            self._collate_fn, pin_memory, drop_last)

    def _collate_fn(self, batch):
        images = torch.stack([_.image for _ in batch])
        gts = torch.from_numpy(np.array([_.gt for _ in batch]))
        output = EasyDict({
            'image': images,
            'gt': gts,
        })
        return output

    def get_epoch_size(self):
        if isinstance(self.batch_sampler, InfiniteBatchSampler):
            return len(self.batch_sampler.batch_sampler)   # training
        return len(self.batch_sampler)
