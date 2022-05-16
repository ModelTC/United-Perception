import json
import torch
import numpy as np
from up.data.datasets.base_dataset import BaseDataset
from up.utils.general.registry_factory import DATASET_REGISTRY
from easydict import EasyDict
from PIL import Image
from up.utils.general.petrel_helper import PetrelHelper
from up.data.data_utils import is_numpy_image
from up.utils.env.dist_helper import env
from up.utils.general.registry import Registry
from up.data.data_utils import count_dataset_size, get_rank_indices

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
                filename, label = line.strip().rsplit(' ', 1)
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
                 image_type='pil',
                 fraction=1.0,
                 save_score=False):
        super(ClsDataset, self).__init__(meta_file,
                                         image_reader,
                                         transformer,
                                         evaluator)
        if evaluator:
            self.topk = evaluator.get('kwargs', {}).get('topk', [1, 5])
        else:
            self.topk = [1, 5]
        self.save_score = save_score
        self.meta_type = meta_type
        self.image_type = image_type
        self.fraction = fraction
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
        if self.fraction != 1.:
            rand_idx = np.random.choice(np.arange(len(self.metas)), int(len(self.metas) * self.fraction), replace=False)
            self.metas = np.array(self.metas)[rand_idx].tolist()

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
        maxk = max(self.topk)
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
            _topk_idx, _topk_score = [], []
            if multicls:
                _prediction = [int(item) for item in prediction[:, b_idx]]
                _label = [int(item) for item in label[b_idx][:]]
                _score = []
                _filename = filenames[b_idx]
                for idx, item in enumerate(score):
                    _score.append([float('%.8f' % s) for s in item[b_idx]])
                    maxk_temp = min(maxk, len(item[0]))
                    _topk_score_temp, _topk_idx_temp = torch.tensor(_score[-1]).topk(maxk_temp, 0, True, True)
                    _topk_score.append(_topk_score_temp.tolist())
                    _topk_idx.append(_topk_idx_temp.tolist())
            else:
                _prediction = int(prediction[b_idx])
                _label = int(label[b_idx])
                _score = [float('%.8f' % s) for s in score[b_idx]]
                maxk = min(maxk, len(_score))
                _topk_score, _topk_idx = torch.tensor(_score).topk(maxk, 0, True, True)
                _topk_score = _topk_score.tolist()
                _topk_idx = _topk_idx.tolist()
                _filename = filenames[b_idx]
            res = {
                'prediction': _prediction,
                'label': _label,
                'topk_idx': _topk_idx,
                'score': _topk_score,
                'filename': _filename
            }
            if self.save_score:
                res['score'] = _score
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
                 reload_cfg={},
                 random=True):
        self.mini_epoch = reload_cfg.get('mini_epoch', 1)
        self.seed = reload_cfg.get('seed', 0)
        self.mini_epoch_idx = reload_cfg.get('mini_epoch_idx', 0)
        self.group = reload_cfg.get('group', 1)
        self.world_size = env.world_size
        self.rank = env.rank
        self.random = random
        super(RankClsDataset, self).__init__(meta_file,
                                             image_reader,
                                             transformer,
                                             evaluator,
                                             meta_type,
                                             parser_info,
                                             image_type)

    def parse_metas(self):
        dataset_sizes = count_dataset_size(self.meta_file)
        rank_indices, rank_num_samples = get_rank_indices(dataset_sizes,
                                                          self.group,
                                                          self.world_size,
                                                          self.mini_epoch,
                                                          self.rank,
                                                          self.mini_epoch_idx,
                                                          self.seed,
                                                          self.random)
        rank_indices = set(rank_indices)
        self.metas = []
        start_indexs = [0]
        # acc sum
        for idx in range(1, len(dataset_sizes)):
            start_indexs.append(start_indexs[idx - 1] + dataset_sizes[idx - 1])
        for idx, meta_file in enumerate(self.meta_file):
            self.meta_parser[idx].parse(meta_file, idx, self.metas, start_indexs[idx], rank_indices)
        if len(rank_indices) != rank_num_samples:
            self.metas += self.metas[:(rank_num_samples - len(rank_indices))]
