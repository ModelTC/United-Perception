from __future__ import division

# Standard Library
import collections
import sys

# Import from third library
import numpy as np
import torch
import math
from up.utils.general.petrel_helper import PetrelHelper
from PIL import Image
try:
    import accimage
except ImportError:
    accimage = None


if sys.version_info < (3, 3):
    Sequence = collections.Sequence
    Iterable = collections.Iterable
else:
    Sequence = collections.abc.Sequence
    Iterable = collections.abc.Iterable


def is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


def is_tensor_image(img):
    return torch.is_tensor(img)


def _is_numpy(img):
    return isinstance(img, np.ndarray)


def is_numpy_image(img):
    return _is_numpy(img) and img.ndim in {2, 3}


def get_image_size(img):
    """ return image size in (h, w) format
    """

    if is_pil_image(img):
        w, h = img.size
    elif is_tensor_image(img):
        h, w = img.shape[-2:]
    elif is_numpy_image(img):
        h, w = img.shape[:2]
    else:
        raise TypeError("Unexpected type {}".format(type(img)))
    return h, w


def coin_tossing(prob):
    """
    """
    return torch.rand(1)[0].item() < prob


def count_dataset_size(meta_files):
    from itertools import (takewhile, repeat)
    buffer = 1024 * 1024 * 8
    dataset_sizes = []
    for meta_file in meta_files:
        meta_size = 0
        with PetrelHelper.open(meta_file) as f:
            try:
                buf_gen = takewhile(lambda x: x, (f.read(buffer) for _ in repeat(None)))
                size = sum(buf.count('\n') for buf in buf_gen)
            except Exception as e:  # noqa
                size = 0
                for _ in f:
                    size += 1
            meta_size += size
        dataset_sizes.append(meta_size)
    return dataset_sizes


def get_rank_indices(dataset_sizes, group=1, world_size=1, mini_epoch=1, rank=0, mini_epoch_idx=0, seed=0, random=True):
    dataset_size = np.array(dataset_sizes).sum()
    if random:
        indices = get_group_random_indices(dataset_size, group=group, seed=seed)
    else:
        return get_test_rank_indices(dataset_size, world_size, rank)
    rank_num_samples = int(math.ceil(dataset_size * 1.0 / (world_size * mini_epoch)))
    total_size = rank_num_samples * world_size * mini_epoch
    indices += indices[:(total_size - len(indices))]
    offset = rank_num_samples * rank
    mini_epoch_offset_begin = mini_epoch_idx * rank_num_samples * world_size
    mini_epoch_offset_end = (mini_epoch_idx + 1) * rank_num_samples * world_size
    rank_indices = indices[mini_epoch_offset_begin:mini_epoch_offset_end][offset:offset + rank_num_samples]
    assert len(rank_indices) == rank_num_samples
    return rank_indices, rank_num_samples


def get_test_rank_indices(dataset_size, world_size=1, rank=0):
    num_samples = len(range(rank, dataset_size, world_size))
    indices = list(range(dataset_size))
    indices = indices[rank::world_size]
    assert len(indices) == num_samples
    return indices, num_samples


def get_group_random_indices(dataset_size, group=1, seed=0, magic_num=10000):
    indices = []
    temp_indices = []
    mini_dataset_size = int(math.ceil(dataset_size * 1.0 / group))
    group = math.ceil(dataset_size * 1.0 / mini_dataset_size)
    last_size = dataset_size - mini_dataset_size * (group - 1)
    for i in range(group):
        if i <= group - 2:
            cur_size = mini_dataset_size
        else:
            cur_size = last_size
        np.random.seed(seed + i + magic_num)
        _indices = np.random.permutation(cur_size).astype(np.int32)
        _indices += i * mini_dataset_size
        temp_indices.append(_indices.tolist())
    np.random.seed(seed + magic_num)
    for i in np.random.permutation(group):
        indices.extend(temp_indices[i])
    return indices
