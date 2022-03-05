from __future__ import division

# Import from third library
from torch.utils.data import DataLoader

from up.utils.general.registry_factory import DATALOADER_REGISTRY, BATCHING_REGISTRY
from up.data.samplers.batch_sampler import InfiniteBatchSampler


__all__ = ['PointDataLoader']


@DATALOADER_REGISTRY.register('point')
class PointDataLoader(DataLoader):
    def __init__(self,
                 dataset,
                 alignment=1,
                 batch_size=1,
                 shuffle=False,
                 sampler=None,
                 batch_sampler=None,
                 num_workers=0,
                 pin_memory=False,
                 drop_last=False,
                 pad_value=0,
                 pad_type='batch_pad',
                 worker_init=False):
        worker_init_fn = None
        if worker_init:
            from up.utils.env.gene_env import worker_init_reset_seed
            worker_init_fn = worker_init_reset_seed
        super(
            PointDataLoader,
            self).__init__(
            dataset,
            batch_size,
            shuffle,
            sampler,
            batch_sampler,
            num_workers,
            dataset.collate_batch,
            pin_memory,
            drop_last,
            worker_init_fn=worker_init_fn)
        self.pad = BATCHING_REGISTRY.get(pad_type)(alignment, pad_value)

    def get_data_size(self):
        return self.get_epoch_size()

    def get_epoch_size(self):
        if isinstance(self.batch_sampler, InfiniteBatchSampler):
            return len(self.batch_sampler.batch_sampler)   # training
        return len(self.batch_sampler)
