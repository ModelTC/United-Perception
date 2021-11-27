# Standard Library
import math
from collections import defaultdict

# Import from third library
import numpy as np
import torch
from torch.utils.data.sampler import Sampler

from eod.utils.env.dist_helper import env, get_rank, get_world_size
from eod.utils.general.log_helper import default_logger as logger
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


@SAMPLER_REGISTRY.register('repeat_factor')
class DistributedRepeatFactorReSampler(Sampler):
    """ Suitable for long-tail distribution datasets.
    Refer to `LVIS <https://arxiv.org/abs/1908.03195>`_ paper
    """
    def __init__(self, dataset, t=0.001, ri_mode='random_round', pn=0.5,
                 ri_if_empty=1, num_replicas=None, static_size=True, rank=None):
        """
        Arguments:
            - dataset (:obj:`Dataset`): dataset used for sampling.
            - t (:obj:`float`):  thresh- old that intuitively controls the point at which oversampling kicks in
            - ri_mode (:obj:`str`): choices={floor, round, random_round, ceil, c_ceil_r_f_floor}, method to compute
              repeat factor for one image
            - pn (:obj:`float`): power number
            - num_replicas (int): number of processes participating in distributed training, optional.
            - rank (int): rank of the current process within num_replicas, optional.
        """
        if num_replicas is None:
            num_replicas = get_world_size()
        if rank is None:
            rank = get_rank()

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.original_num_samples = self.num_samples
        self.t = t
        self.ri_mode = ri_mode
        self.ri_if_empty = int(ri_if_empty)
        self.pn = pn
        self.static_size = static_size
        self._prepare()
        logger.info('init re-sampler, ri mode: {}'.format(self.ri_mode))

    def _prepare(self):
        # prepare re-sampling factor for category
        rc = defaultdict(int)
        img_num_per_class = defaultdict(int)
        for cls, img_num in sorted(self.dataset.num_images_per_class.items()):
            f = img_num / len(self.dataset)
            img_num_per_class[cls] = img_num
            rc[cls] = max(1, math.pow(self.t / f, self.pn))
            logger.info('class id {}, image count {}, rc {}'.format(cls, img_num, rc[cls]))
        self.rc = rc

    def _compute_ri(self, img_index):
        classes = self.dataset.get_image_classes(img_index)
        ris = [self.rc[cls] for cls in classes]
        if len(ris) == 0:
            return self.ri_if_empty
        if self.ri_mode == 'floor':
            ri = int(max(ris))
        elif self.ri_mode == 'round':
            ri = round(max(ris))
        elif self.ri_mode == 'random_round':
            ri_max = max(ris)
            p = ri_max - int(ri_max)
            if np.random.rand() < p:
                ri = math.ceil(ri_max)
            else:
                ri = int(ri_max)
        elif self.ri_mode == 'ceil':
            ri = math.ceil(max(ris))
        elif self.ri_mode == 'c_ceil_r_f_floor':
            max_ind = np.argmax(ris)
            assert hasattr(self.dataset, 'lvis'), 'Only lvis dataset supportted for c_ceil_r_f_floor mode'
            img_id = self.dataset.img_ids[img_index]
            meta_annos = self.dataset.lvis.img_ann_map[img_id]
            f = self.dataset.lvis.cats[meta_annos[max_ind]['category_id']]['frequency']
            assert f in ['f', 'c', 'r']
            if f in ['r', 'f']:
                ri = int(max(ris))
            else:
                ri = math.ceil(max(ris))
        else:
            raise NotImplementedError
        return ri

    def _get_new_indices(self):
        indices = []
        for idx in range(len(self.dataset)):
            ri = self._compute_ri(idx)
            indices += [idx] * ri

        logger.info('dataset size {}, indexes size {}'.format(len(self.dataset), len(indices)))
        return indices

    def __iter__(self):
        # deterministically shuffle based on epoch

        # generate a perm based using class-aware balance for this epoch
        indices = self._get_new_indices()

        # override num_sample total size
        self.num_samples = int(math.ceil(len(indices) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

        indices = np.random.RandomState(seed=self.epoch).permutation(np.array(indices))
        indices = list(indices)

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset:offset + self.num_samples]
        assert len(indices) == self.num_samples

        # convert to int because this array will be converted to torch.tensor,
        # but torch.as_tensor dosen't support numpy.int64
        # a = torch.tensor(np.float64(1))  # works
        # b = torch.tensor(np.int64(1))  # fails
        indices = list(map(lambda x: int(x), indices))
        return iter(indices)

    def __len__(self):
        return self.original_num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
