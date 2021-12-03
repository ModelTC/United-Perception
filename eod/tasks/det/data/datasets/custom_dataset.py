from __future__ import division

# Standard Library
import json
import math

# Import from third library
import cv2
import numpy as np
import torch
import copy
from easydict import EasyDict
import os
import pickle as pk

from eod.utils.general.log_helper import default_logger as logger
from eod.utils.general.registry_factory import DATASET_REGISTRY
from eod.utils.env.dist_helper import env
from eod.data.image_reader import get_cur_image_dir
from eod.data.datasets.base_dataset import BaseDataset
from eod.data.data_utils import get_image_size
from torch.nn.modules.utils import _pair

# TODO: Check GPU usage after move this setting down from upper line
cv2.ocl.setUseOpenCL(False)


__all__ = ['CustomDataset', 'RankCustomDataset']


@DATASET_REGISTRY.register('custom')
class CustomDataset(BaseDataset):
    def __init__(self,
                 meta_file,
                 image_reader,
                 transformer,
                 num_classes,
                 evaluator=None,
                 label_mapping=None,
                 cache=None,
                 clip_box=True):
        super(CustomDataset, self).__init__(
            meta_file, image_reader, transformer, evaluator=evaluator)

        self._num_classes = num_classes
        self.label_mapping = label_mapping
        self.metas = []
        self.aspect_ratios = []
        self._normal_init()
        self.clip_box = clip_box

        if len(self.aspect_ratios) == 0:
            self.aspect_ratios = [1] * len(self.metas)
        self.cache = cache
        if self.cache is not None:
            cache_dir = self.cache.get('cache_dir', './')
            os.makedirs(cache_dir, exist_ok=True)
            cache_name = self.cache.get('cache_name', 'cache.pkl')
            cache_file = os.path.join(cache_dir, cache_name)
            if not os.path.exists(cache_file):
                self.cache_image = {}
                self.cache_dataset()
                if env.is_master():
                    with open(cache_file, "wb") as f:
                        pk.dump(self.cache_image, f)
            else:
                with open(cache_file, "rb") as f:
                    self.cache_image = pk.load(f)

    def cache_dataset(self):
        from multiprocessing.pool import ThreadPool
        NUM_THREADs = min(8, os.cpu_count())
        pool = ThreadPool(NUM_THREADs)
        pool.map(self.set_cache_images, self.metas)
        pool.close()
        pool.join()

    def set_cache_images(self, data):
        filename = data['filename']
        image_dir = get_cur_image_dir(self.image_reader.image_dir, data.get('image_source', 0))
        filename = os.path.join(image_dir, filename)
        with open(filename, "rb") as f:
            img = np.frombuffer(f.read(), np.uint8)
            if 'tar_size' in self.cache:
                img = cv2.imdecode(img, cv2.IMREAD_COLOR)
                img = self.resize_img(img, self.cache['tar_size'])
                img = cv2.imencode('.jng')[1]
            self.cache_image[filename] = img

    def resize_img(self, img, tar_size):
        interp = cv2.INTER_LINEAR
        h0, w0 = img.shape[:2]  # orig hw
        r = min(1. * tar_size[0] / h0, 1. * tar_size[1] / w0)
        if r != 1:
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
        return img

    def get_cache_image(self, data):
        image_dir = get_cur_image_dir(self.image_reader.image_dir, data.get('image_source', 0))
        filename = os.path.join(image_dir, data['filename'])
        return self.cache_image[filename]

    def _normal_init(self):
        if not isinstance(self.meta_file, list):
            self.meta_file = [self.meta_file]
        for idx, meta_file in enumerate(self.meta_file):
            with open(meta_file, "r") as f:
                for line in f:
                    data = json.loads(line)
                    if self.label_mapping is not None:
                        data = self.set_label_mapping(data, self.label_mapping[idx], 0)
                        data['image_source'] = idx
                    self.metas.append(data)
                    if 'image_height' not in data or 'image_width' not in data:
                        logger.warning('image size is not provided, '
                                       'set aspect grouping to 1.')
                        self.aspect_ratios.append(1.0)
                    else:
                        self.aspect_ratios.append(data['image_height'] / data['image_width'])

    def set_label_mapping(self, data, label_mapping, neg_target):
        instances = data['instances']
        data["neg_target"] = neg_target
        for instance in instances:
            if instance.get("is_ignored", False):  # ingore region we don't need mapping
                continue
            if "label" in instance:
                label = int(instance["label"] - 1)
                assert label < len(label_mapping)
                instance["label"] = label_mapping[label]
            else:
                instance["label"] = -1  # ignore region if label is not provided
        return data

    def dump(self, output):
        image_info = output['image_info']
        bboxes = self.tensor2numpy(output['dt_bboxes'])
        image_ids = output['image_id']
        out_res = []
        for b_ix in range(len(image_info)):
            info = image_info[b_ix]
            height, width = map(int, info[3:5])
            img_id = image_ids[b_ix]
            scores = bboxes[:, 5]
            keep_ix = np.where(bboxes[:, 0] == b_ix)[0]
            keep_ix = sorted(keep_ix, key=lambda ix: scores[ix], reverse=True)
            scale_h, scale_w = _pair(info[2])
            img_bboxes = bboxes[keep_ix].copy()
            # sub pad
            pad_w, pad_h = info[6], info[7]
            img_bboxes[:, [1, 3]] -= pad_w
            img_bboxes[:, [2, 4]] -= pad_h
            # clip
            if self.clip_box:
                np.clip(img_bboxes[:, [1, 3]], 0, info[1], out=img_bboxes[:, [1, 3]])
                np.clip(img_bboxes[:, [2, 4]], 0, info[0], out=img_bboxes[:, [2, 4]])
            img_bboxes[:, 1] /= scale_w
            img_bboxes[:, 2] /= scale_h
            img_bboxes[:, 3] /= scale_w
            img_bboxes[:, 4] /= scale_h

            for i in range(len(img_bboxes)):
                box_score, cls = img_bboxes[i][5:7]
                bbox = img_bboxes[i].copy()
                score = float(box_score)
                res = {
                    'height': height,
                    'width': width,
                    'image_id': img_id,
                    'bbox': bbox[1: 1 + 4].tolist(),
                    'score': score,
                    'label': int(cls)
                }
                out_res.append(res)
        return out_res

    def __len__(self):
        return len(self.metas)

    def get_input(self, idx):
        """parse annotation into input dict
        """
        data = self.metas[idx]
        data = copy.deepcopy(data)
        img_id = filename = data['filename']
        gt_bboxes = []
        ig_bboxes = []
        for instance in data.get('instances', []):
            if instance['is_ignored']:
                ig_bboxes.append(instance['bbox'])
            else:
                gt_bboxes.append(instance['bbox'] + [instance['label']])

        if len(ig_bboxes) == 0:
            ig_bboxes = self._fake_zero_data(1, 4)
        if len(gt_bboxes) == 0:
            gt_bboxes = self._fake_zero_data(1, 5)

        gt_bboxes = torch.as_tensor(gt_bboxes, dtype=torch.float32)
        ig_bboxes = torch.as_tensor(ig_bboxes, dtype=torch.float32)
        cache = False
        try:
            if self.cache is not None:
                img = self.get_cache_image(data)
                img = cv2.imdecode(img, cv2.IMREAD_COLOR)
                if self.image_reader.color_mode != 'BGR':
                    cvt_color = getattr(cv2, 'COLOR_BGR2{}'.format(self.image_reader.color_mode))
                    img = cv2.cvtColor(img, cvt_color)
                cache = True
            else:
                img = self.image_reader(filename, data.get('image_source', 0))
        except:  # noqa 
            img = self.image_reader(filename, data.get('image_source', 0))
        input = EasyDict({
            'image': img,
            'gt_bboxes': gt_bboxes,
            'gt_ignores': ig_bboxes,
            'flipped': False,
            'filename': filename,
            'image_id': img_id,
            'dataset_idx': idx,
            'neg_target': data.get('neg_target', 0),
            'cache': cache
        })
        return input

    def __getitem__(self, idx):
        """
        Get a single image data: from dataset

        Arguments:
            - idx (:obj:`int`): index of image, 0 <= idx < len(self)

        Returns:
            - input (:obj:`dict`)

        Output example::

            {
                # (FloatTensor): [1, 3, h, w], RGB format
                'image': ..,
                # (list): [resized_h, resized_w, scale_factor, origin_h, origin_w]
                'image_info': ..,
                # (FloatTensor): [N, 5] (x1, y1, x2, y2, label)
                'gt_bboxes': ..,
                # (FloatTensor): [N, 4] (x1, y1, x2, y2)
                'ig_bboxes': ..,
                # (str): image name
                'filename': ..
            }
        """
        input = self.get_input(idx)
        image_h, image_w = get_image_size(input.image)
        input = self.transformer(input)
        scale_factor = input.get('scale_factor', 1)

        new_image_h, new_image_w = get_image_size(input.image)
        pad_w, pad_h = input.get('dw', 0), input.get('dh', 0)
        input.image_info = [new_image_h, new_image_w, scale_factor, image_h, image_w, input.flipped, pad_w, pad_h]
        return input


@DATASET_REGISTRY.register('rank_custom')
class RankCustomDataset(CustomDataset):
    def __init__(self,
                 meta_file,
                 image_reader,
                 transformer,
                 num_classes,
                 evaluator=None,
                 label_mapping=None,
                 reload_cfg={}):
        self.mini_epoch = reload_cfg.get('mini_epoch', 1)
        self.seed = reload_cfg.get('seed', 0)
        self.mini_epoch_idx = reload_cfg.get('mini_epoch_idx', 0)
        self.group = reload_cfg.get('group', 1)
        self.world_size = env.world_size
        self.rank = env.rank
        super(RankCustomDataset, self).__init__(
            meta_file,
            image_reader,
            transformer,
            num_classes,
            evaluator=evaluator,
            label_mapping=label_mapping)

    def count_dataset_size(self, meta_files):
        from itertools import (takewhile, repeat)
        buffer = 1024 * 1024 * 8
        dataset_size_sum = 0
        for meta_file in meta_files:
            # only support lustre read
            with open(meta_file, "r") as f:
                buf_gen = takewhile(lambda x: x, (f.read(buffer) for _ in repeat(None)))
                size = sum(buf.count('\n') for buf in buf_gen)
                dataset_size_sum += size
        return dataset_size_sum

    def get_rank_indices(self, meta_files):
        dataset_size = self.count_dataset_size(meta_files)
        indices = self.get_group_random_indices(dataset_size, group=self.group)
        rank_num_samples = int(math.ceil(dataset_size * 1.0 / (self.world_size * self.mini_epoch)))
        total_size = rank_num_samples * self.world_size * self.mini_epoch
        indices += indices[:(total_size - len(indices))]
        offset = rank_num_samples * self.rank
        mini_epoch_offset_begin = self.mini_epoch_idx * rank_num_samples * self.world_size
        mini_epoch_offset_end = (self.mini_epoch_idx + 1) * rank_num_samples * self.world_size
        rank_indices = indices[mini_epoch_offset_begin:mini_epoch_offset_end][offset:offset + rank_num_samples]
        assert len(rank_indices) == rank_num_samples
        return rank_indices, rank_num_samples

    def get_group_random_indices(self, dataset_size, group=1):
        magic_num = 10000
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
            np.random.seed(self.seed + i + magic_num)
            _indices = np.random.permutation(cur_size).astype(np.int32)
            _indices += i * mini_dataset_size
            temp_indices.append(_indices.tolist())
        np.random.seed(self.seed + magic_num)
        for i in np.random.permutation(group):
            indices.extend(temp_indices[i])
        return indices

    def _normal_init(self):
        if not isinstance(self.meta_file, list):
            self.meta_file = [self.meta_file]
        rank_indices, rank_num_samples = self.get_rank_indices(self.meta_file)
        rank_indices = set(rank_indices)
        _index = 0
        for idx, meta_file in enumerate(self.meta_file):
            with open(meta_file, "r") as f:
                for line in f:
                    if _index in rank_indices:
                        data = json.loads(line)
                        if self.label_mapping is not None:
                            data = self.set_label_mapping(data, self.label_mapping[idx], 0)
                            data['image_source'] = idx
                        self.metas.append(data)
                        if 'image_height' not in data or 'image_width' not in data:
                            logger.warning('image size is not provided, '
                                           'set aspect grouping to 1.')
                            self.aspect_ratios.append(1.0)
                        else:
                            self.aspect_ratios.append(data['image_height'] / data['image_width'])
                    _index += 1
        if len(rank_indices) != rank_num_samples:
            self.metas += self.metas[:(rank_num_samples - len(rank_indices))]
            self.aspect_ratios += self.aspect_ratios[:(rank_num_samples - len(rank_indices))]
