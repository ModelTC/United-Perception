from torch.utils.data import DataLoader
import torch
import numpy as np
from easydict import EasyDict

from up.utils.general.registry_factory import DATALOADER_REGISTRY, BATCHING_REGISTRY
from up.data.samplers.batch_sampler import InfiniteBatchSampler


__all__ = ['ClassDataLoader']


@DATALOADER_REGISTRY.register('cls_base')
class ClassDataLoader(DataLoader):

    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None,
                 num_workers=0, pin_memory=False, drop_last=False, batch_fn=None
                 ):
        super(ClassDataLoader, self).__init__(
            dataset, batch_size, shuffle, sampler, batch_sampler, num_workers,
            self._collate_fn, pin_memory, drop_last)
        if batch_fn is not None:
            self.batch_fn = BATCHING_REGISTRY.get(batch_fn['type'])(**batch_fn['kwargs'])
        else:
            self.batch_fn = None

    def _collate_fn(self, batch):
        images = torch.stack([_.image for _ in batch])
        if isinstance(batch[0].gt, int):
            gts = torch.from_numpy(np.array([_.gt for _ in batch]))
        elif isinstance(batch[0].gt, list):
            gts = torch.stack([torch.from_numpy(np.array(_.gt)) for _ in batch])
        elif batch[0].gt.dim() > 0:
            gts = torch.stack([_.gt for _ in batch])

        filenames = [_.get('filename', '') for _ in batch]

        output = EasyDict({
            'image': images,
            'gt': gts,
            'filenames': filenames,
        })
        if self.batch_fn is not None:
            output = self.batch_fn(output)

        return output

    def get_epoch_size(self):
        if isinstance(self.batch_sampler, InfiniteBatchSampler):
            return len(self.batch_sampler.batch_sampler)   # training
        return len(self.batch_sampler)
