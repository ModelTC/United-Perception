# Standard Library
import bisect
import itertools

# Import from third library
import numpy as np
import torch
from torch.utils.data.sampler import BatchSampler
from eod.utils.general.registry_factory import BATCH_SAMPLER_REGISTRY


__all__ = ['BaseBatchSampler', 'AspectRatioGroupedBatchSampler', 'InfiniteBatchSampler']


@BATCH_SAMPLER_REGISTRY.register('base')
class BaseBatchSampler(BatchSampler):
    def __init__(self, sampler, batch_size, drop_last=False):
        super(BaseBatchSampler, self).__init__(sampler, batch_size, drop_last)


@BATCH_SAMPLER_REGISTRY.register('aspect_ratio_group')
class AspectRatioGroupedBatchSampler(BatchSampler):
    """Wraps another sampler to yield a mini-batch of indices.
    It enforces that elements from the same group should appear in groups of batch_size.
    It also tries to provide mini-batches which follows an ordering which is
    as close as possible to the ordering from the original sampler.

    """

    def __init__(self, sampler, batch_size, aspect_grouping, drop_last=False, training=True):
        """
        Arguments:
             - sampler (:obj:`sampler`): instance of sampler object
             - batch_size (:obj:`int`)
             - aspect_grouping (:obj:`list` of `float`): split point of aspect ration
        """
        super(AspectRatioGroupedBatchSampler, self).__init__(sampler, batch_size, drop_last)
        self.group_ids = self._get_group_ids(sampler.dataset, aspect_grouping)
        assert self.group_ids.dim() == 1
        self.training = training

        self.groups = torch.unique(self.group_ids).sort(0)[0]
        self._can_reuse_batches = False

    def _get_group_ids(self, dataset, aspect_grouping):
        assert isinstance(aspect_grouping, (list, tuple)), "aspect_grouping must be a list or tuple"
        assert hasattr(dataset, 'aspect_ratios'), 'Image aspect ratios are required'

        # devide images into groups by aspect ratios
        aspect_ratios = dataset.aspect_ratios
        aspect_grouping = sorted(aspect_grouping)
        group_ids = list(map(lambda y: bisect.bisect_right(aspect_grouping, y), aspect_ratios))
        return torch.as_tensor(group_ids)

    def _prepare_batches(self):
        sampled_ids = torch.as_tensor(list(self.sampler))
        sampled_group_ids = self.group_ids[sampled_ids]
        clusters = [sampled_ids[sampled_group_ids == i] for i in self.groups]
        target_batch_num = int(np.ceil(len(sampled_ids) / float(self.batch_size)))

        splits = [c.split(self.batch_size) for c in clusters if len(c) > 0]
        merged = tuple(itertools.chain.from_iterable(splits))

        # re-permuate the batches by the order that
        # the first element of each batch occurs in the original sampled_ids
        first_element_of_batch = [t[0].item() for t in merged]
        inv_sampled_ids_map = {v: k for k, v in enumerate(sampled_ids.tolist())}
        first_index_of_batch = torch.as_tensor(
            [inv_sampled_ids_map[s] for s in first_element_of_batch])
        permutation_order = first_index_of_batch.sort(0)[1].tolist()
        batches = [merged[i].tolist() for i in permutation_order]

        if self.training:
            # ensure number of batches in different gpus are the same
            if len(batches) > target_batch_num:
                batches = batches[:target_batch_num]
            assert len(batches) == target_batch_num, "Error, uncorrect target_batch_num!"
        return batches

    def __iter__(self):
        if self._can_reuse_batches:
            batches = self._batches
            self._can_reuse_batches = False
        else:
            batches = self._prepare_batches()
        self._batches = batches
        return iter(batches)

    def __len__(self):
        if getattr(self.sampler, 'static_size', False):
            return int(np.ceil(len(self.sampler) / float(self.batch_size)))
        if not hasattr(self, "_batches"):
            self._batches = self._prepare_batches()
            self._can_reuse_batches = True
        return len(self._batches)


class InfiniteBatchSampler(object):
    """Wraps a BatchSampler, resampling a specified number of iterations"""

    def __init__(self, batch_sampler, start_iter=0):
        """
        Arguments:
             - batch_sampler (:obj:`sampler`): instance of sampler object
        """
        self.batch_sampler = batch_sampler
        self.batch_size = self.batch_sampler.batch_size
        self.start_iter = start_iter

    def __iter__(self):

        cur_iter = self.start_iter
        len_dataset = len(self.batch_sampler)
        while True:
            if hasattr(self.batch_sampler.sampler, 'set_epoch'):
                self.batch_sampler.sampler.set_epoch(cur_iter // len_dataset)
            for batch in self.batch_sampler:
                cur_iter += 1
                yield batch

    def __len__(self):
        raise ValueError('Infinite sampler has no length')

    def set_start_iter(self, start_iter):
        self.start_iter = start_iter
