import json
import numpy as np
import math
from up.data.datasets.base_dataset import BaseDataset
from up.utils.general.registry_factory import DATASET_REGISTRY
from easydict import EasyDict
from PIL import Image
from up.utils.general.petrel_helper import PetrelHelper
from up.data.data_utils import is_numpy_image
from up.utils.env.dist_helper import env
from up.utils.general.registry import Registry

__all__ = ['ClsDataset']

CLS_PARSER_REGISTRY = Registry()


class BaseParser(object):
    def __init__(self, extra_info={}):
        self.extra_info = extra_info

    def parse(self):
        raise NotImplementedError


@CLS_PARSER_REGISTRY.register('imagenet')
class ImageNetParser(BaseParser):
    def parse(self, meta_file, idx, metas, start_index=0, rank_indices=None):
        with PetrelHelper.open(meta_file) as f:
            for index, line in enumerate(f):
                if rank_indices is not None:
                    if (start_index + index) not in rank_indices:
                        continue
                cls_res = {}
                filename, label = line.strip().split()
                cls_res['filename'] = filename
                cls_res['label'] = int(label)
                cls_res['image_source'] = idx
                metas.append(cls_res)
        return metas


@CLS_PARSER_REGISTRY.register('custom_cls')
class CustomClsParser(BaseParser):
    def parse(self, meta_file, idx, metas, start_index=0, rank_indices=None):
        with PetrelHelper.open(meta_file) as f:
            for index, line in enumerate(f):
                if rank_indices is not None:
                    if (start_index + index) not in rank_indices:
                        continue
                cls_res = {}
                res = json.loads(line.strip())
                filename = res['filename']
                if isinstance(res['label'], list):   # support multilabel cls
                    cls_res['label'] = [int(i) for i in res['label']]
                else:
                    cls_res['label'] = int(res['label'])
                cls_res['filename'] = filename
                cls_res['image_source'] = idx
                metas.append(cls_res)
        return metas


@CLS_PARSER_REGISTRY.register("custom_det")
class CustomDetParser(BaseParser):
    def parse(self, meta_file, idx, metas, start_index=0, rank_indices=None):
        min_size = self.extra_info.get("min_size", 40)
        with PetrelHelper.open(meta_file) as f:
            for index, line in enumerate(f):
                if rank_indices is not None:
                    if (start_index + index) not in rank_indices:
                        continue
                data = json.loads(line)
                for instance in data.get("instances", []):
                    if instance.get('is_ignored', False):
                        continue
                    if "label" in instance:
                        # det cls label begin from 1, cls label begin from 0
                        if isinstance(instance["label"], list):  # support multilabel cls
                            label = [int(i) - 1 for i in instance['label']]
                        else:
                            label = int(instance["label"] - 1)
                    else:
                        continue
                    if "bbox" in instance:
                        w = instance['bbox'][2] - instance['bbox'][0]
                        h = instance['bbox'][3] - instance['bbox'][1]
                        if w * h < min_size * min_size:
                            continue
                    else:
                        continue
                    cls_res = {}
                    cls_res['filename'] = data['filename']
                    cls_res['label'] = label
                    cls_res['image_source'] = idx
                    cls_res['bbox'] = instance['bbox']
                    metas.append(cls_res)
        return metas


@DATASET_REGISTRY.register('cls')
class ClsDataset(BaseDataset):
    def __init__(self,
                 meta_file,
                 image_reader,
                 transformer=None,
                 evaluator=None,
                 meta_type='imagenet',
                 parser_info={},
                 image_type='pil'):
        super(ClsDataset, self).__init__(meta_file,
                                         image_reader,
                                         transformer,
                                         evaluator)
        self.meta_type = meta_type
        self.image_type = image_type
        self._list_check()
        self.meta_parser = [CLS_PARSER_REGISTRY[m_type](**parser_info) for m_type in self.meta_type]
        self.parse_metas()

    def _list_check(self):
        if not isinstance(self.meta_file, list):
            self.meta_file = [self.meta_file]
        if not isinstance(self.meta_type, list):
            self.meta_type = [self.meta_type]

    def parse_metas(self):
        self.metas = []
        for idx, meta_file in enumerate(self.meta_file):
            self.meta_parser[idx].parse(meta_file, idx, self.metas)

    def __len__(self):
        return len(self.metas)

    def _load_meta(self, idx):
        return self.metas[idx]

    def get_input(self, idx):
        meta = self._load_meta(idx)
        img = self.image_reader(meta['filename'], meta.get('image_source', 0))
        crop = False
        if 'bbox' in meta:
            xmin, ymin, xmax, ymax = meta['bbox'][:4]
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
            crop = True
        if self.image_type == 'pil':
            if is_numpy_image(img):
                img = Image.fromarray(img)
            if crop:
                img = img.crop((xmin, ymin, xmax, ymax))
        else:
            if crop:
                img = img[ymin:ymax, xmin:xmax]
        label = meta['label']
        input = EasyDict({
            'image': img,
            'gt': label,
            'filename': meta['filename'],
            'data_type': 1
        })
        return input

    def __getitem__(self, idx):
        input = self.get_input(idx)
        if self.transformer is not None:
            input = self.transformer(input)
        return input

    def dump(self, output):
        prediction = self.tensor2numpy(output['preds'])
        label = self.tensor2numpy(output['gt'])
        multicls = False
        scores = output['scores']
        filenames = output['filenames']
        if isinstance(prediction, list):
            # for multicls
            # prediction n * single cls
            # label [1, 2, 3.. n]
            # score n* single cls
            multicls = True
            prediction = np.array(prediction)
            score = [self.tensor2numpy(i) for i in scores]
        else:
            score = self.tensor2numpy(output['scores'])
        out_res = []
        for b_idx in range(label.shape[0]):
            if multicls:
                _prediction = [int(item) for item in prediction[:, b_idx]]
                _label = [int(item) for item in label[b_idx][:]]
                _score = []
                for idx, item in enumerate(score):
                    _score.append([float('%.8f' % s) for s in item[b_idx]])
            else:
                _prediction = int(prediction[b_idx])
                _label = int(label[b_idx])
                _score = [float('%.8f' % s) for s in score[b_idx]]
                _filename = filenames[b_idx]
            res = {
                'prediction': _prediction,
                'label': _label,
                'score': _score,
                'filename': _filename
            }
            out_res.append(res)
        return out_res


@DATASET_REGISTRY.register('rank_cls')
class RankClsDataset(ClsDataset):
    def __init__(self,
                 meta_file,
                 image_reader,
                 transformer=None,
                 evaluator=None,
                 meta_type='imagenet',
                 parser_info={},
                 image_type='pil',
                 reload_cfg={}):
        self.mini_epoch = reload_cfg.get('mini_epoch', 1)
        self.seed = reload_cfg.get('seed', 0)
        self.mini_epoch_idx = reload_cfg.get('mini_epoch_idx', 0)
        self.group = reload_cfg.get('group', 1)
        self.world_size = env.world_size
        self.rank = env.rank
        super(ClsDataset, self).__init__(meta_file,
                                         image_reader,
                                         transformer,
                                         evaluator,
                                         meta_type,
                                         parser_info,
                                         image_type)

    def count_dataset_size(self, meta_files):
        from itertools import (takewhile, repeat)
        buffer = 1024 * 1024 * 8
        self.dataset_sizes = []
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
            self.dataset_sizes.append(meta_size)
        return np.array(self.dataset_sizes).sum()

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

    def parse_metas(self):
        rank_indices, rank_num_samples = self.get_rank_indices(self.meta_file)
        rank_indices = set(rank_indices)
        self.metas = []
        start_indexs = [0]
        # acc sum
        for idx in range(1, len(self.dataset_sizes)):
            start_indexs.append(start_indexs[idx - 1] + self.dataset_sizes[idx - 1])
        for idx, meta_file in enumerate(self.meta_file):
            self.meta_parser[idx].parse(meta_file, idx, self.metas, start_indexs[idx], rank_indices)
        if len(rank_indices) != rank_num_samples:
            self.metas += self.metas[:(rank_num_samples - len(rank_indices))]
