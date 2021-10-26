# Standard Library
import math

# Import from third library
import torch
from torch.utils.data.sampler import Sampler

from eod.utils.env.dist_helper import env
from eod.utils.general.registry_factory import SAMPLER_REGISTRY


__all__ = ['DistributedSampler', 'LocalSampler', 'TestDistributedSampler']


@SAMPLER_REGISTRY.register('dist')
class DistributedSampler(Sampler):
    """
    Sampler that restricts data loading to a subset of the dataset.

    .. note:
        Dataset is assumed to be of constant size.

    Arguments:
        dataset (Dataset): dataset used for sampling.
        num_replicas (int): number of processes participating in distributed training, optional.
        rank (int): rank of the current process within num_replicas, optional.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, fix_seed=False):
        """
        Arguments:
             - dataset (:obj:`dataset`): instance of dataset object
        """
        if num_replicas is None:
            num_replicas = env.world_size
        if rank is None:
            rank = env.rank

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.fix_seed = fix_seed

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch * (not self.fix_seed))
        indices = list(torch.randperm(len(self.dataset), generator=g))

        # add extra samples to make it evenly divisible
        # indices += indices[:(self.total_size - len(indices))]
        padding_size = self.total_size - len(indices)
        if padding_size <= len(indices):
            indices += indices[:padding_size]
        else:
            indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        assert len(indices) == self.total_size

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset:offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


@SAMPLER_REGISTRY.register('local')
class LocalSampler(Sampler):
    def __init__(self, dataset, rank=None):
        if rank is None:
            rank = env.rank
        self.dataset = dataset
        self.rank = rank
        self.epoch = 0
        self.num_samples = len(self.dataset)

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch + self.rank)
        indices = list(torch.randperm(self.num_samples, generator=g))
        return iter(indices)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.num_samples


@SAMPLER_REGISTRY.register('dist_test')
class TestDistributedSampler(Sampler):
    """
    Sampler that restricts data loading to a subset of the dataset, but won't align the total data
    size to be divisible by world_size bacause this will lead to duplicate detecton results
    """

    def __init__(self, dataset, num_replicas=None, rank=None):
        """
        Arguments:
             - dataset (:obj:`dataset`): instance of dataset object
        """
        if num_replicas is None:
            num_replicas = env.world_size
        if rank is None:
            rank = env.rank

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = len(range(rank, len(self.dataset), num_replicas))
        self.total_size = len(self.dataset)

    def __iter__(self):
        indices = torch.arange(len(self.dataset))
        indices = indices[self.rank::self.num_replicas]
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
