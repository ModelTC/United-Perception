from __future__ import division

# Standard Library
import collections
import sys
import os
import json

# Import from third library
import numpy as np
import torch
import math
from up.utils.general.petrel_helper import PetrelHelper
from up.utils.env.dist_helper import env, barrier
from up.utils.general.log_helper import default_logger as logger
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


def search_folder_and_save(folder_path, save_path):
    # dfs search image folder
    def _dfs_images_searcher(source_path, writer):
        logger.info('search in ' + source_path)
        image_ext = ['jpg', 'JPG', 'PNG', 'png', 'jpeg', 'JPEG', 'bmp']
        for item in os.listdir(source_path):
            item_path = os.path.join(source_path, item)
            if os.path.isfile(item_path):
                name_ext = item.split('.')
                if len(name_ext) < 2:
                    continue
                if name_ext[-1] in image_ext:
                    writer.write(item_path + '\n')
                    writer.flush()
                else:
                    continue
            elif os.path.isdir(item_path):
                _dfs_images_searcher(item_path, writer)
            else:
                return
    writer = open(save_path, 'w')
    _dfs_images_searcher(folder_path, writer)
    writer.close()


def fake_metas_func(meta_info, task, idx=0, write_path=None):
    meta_templates = {'cls': {'filename': '', 'label': -1, 'image_source': idx},
                      'det': {'filename': '', 'image_height': -1, 'image_width': -1,
                              'instances': [{'is_ignored': False, 'bbox': [0.0, 0.0, 0.0, 0.0], 'label': -1}]}}

    assert task in meta_templates
    metas = []
    with PetrelHelper.open(meta_info) as f:
        for _, line in enumerate(f):
            ip = line.strip().replace('\n', '')
            meta_templates[task].update({'filename': ip})
            metas.append(meta_templates[task].copy())
    if write_path is not None:
        with open(write_path, 'w') as f:
            for im in metas:
                f.write(json.dumps(im, ensure_ascii=False) + '\n')
            f.flush()
    return metas


def _parse_results(res, indicator, writer=None):
    done_imgs = set()

    last_img = ""
    datas_length = len(res)
    # drop the last one json lines becuase it may have mistake
    if datas_length != 0:
        res.pop()
    # drop the last one image becuase it is a uncontrollable factor
    if len(res) != 0:
        last_img = json.loads(res[-1])[indicator]

    for line in res:
        data = json.loads(line)
        if data[indicator] != last_img:
            done_imgs.add(data[indicator])
            if writer is not None:
                writer.write(json.dumps(data, ensure_ascii=False) + '\n')
    if writer is not None:
        writer.flush()
    return done_imgs


def test_resume_init(resume_dir, world_size, rank, rank_set=False, indicator="image_id"):
    # for test resume
    done_imgs = set()
    if resume_dir is None:
        return done_imgs

    if not os.path.exists(resume_dir):
        logger.warning('You use the inference resuming, but the resume dir is not exist!')
        return done_imgs

    resume_files = os.listdir(resume_dir)
    resume_fdict = {}
    for fn in resume_files:
        fn_rank = fn.replace('results.txt.rank', '')
        if fn_rank.isdigit():
            resume_fdict[int(fn_rank)] = fn
    if len(resume_fdict) == 0:
        logger.warning('You use the inference resuming, but the resume files are not exist!')
        return done_imgs

    assert world_size == len(resume_fdict), 'Current world size is not matched with the resumed result files ranks!'

    if not rank_set:
        for fr in resume_fdict:
            file_path = os.path.join(resume_dir, resume_fdict[fr])
            with PetrelHelper.open(file_path) as f:
                datas = f.readlines()
            # wait all process read the data done
            barrier()

            writer = None
            if env.is_master():
                writer = open(file_path, 'w')
            done_imgs = done_imgs.union(_parse_results(datas, indicator, writer))
            if writer is not None:
                writer.close()
            barrier()
    else:
        fr = rank
        file_path = os.path.join(resume_dir, resume_fdict[fr])
        with PetrelHelper.open(file_path) as f:
            datas = f.readlines()

        writer = open(file_path, 'w')
        done_imgs = _parse_results(datas, indicator, writer)
        writer.close()
    return done_imgs
