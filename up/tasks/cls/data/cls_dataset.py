import os
import json
import torch
import numpy as np
from up.data.datasets.base_dataset import BaseDataset
from up.utils.general.registry_factory import DATASET_REGISTRY
from easydict import EasyDict
from collections import Counter, defaultdict
from PIL import Image
from up.utils.general.log_helper import default_logger as logger
from up.utils.general.petrel_helper import PetrelHelper
from up.data.data_utils import is_numpy_image
from up.utils.env.dist_helper import env, barrier
from up.utils.general.registry import Registry
from up.data.data_utils import count_dataset_size, get_rank_indices
from up.data.data_utils import test_resume_init, search_folder_and_save, fake_metas_func

__all__ = ['ClsDataset']

CLS_PARSER_REGISTRY = Registry()


class BaseParser(object):
    def __init__(self, extra_info={}):
        self.extra_info = extra_info

    def parse(self):
        raise NotImplementedError


@CLS_PARSER_REGISTRY.register('inference_only')
class InferenceOnlyParser(BaseParser):
    def parse(self, meta_file, idx, start_index=0, rank_indices=None):
        metas = fake_metas_func(meta_file, 'cls', idx)

        if rank_indices is not None:
            rank_metas = []
            for index, m in enumerate(metas):
                if (start_index + index) not in rank_indices:
                    continue
                rank_metas.append(m)
            metas = rank_metas
        return metas


@CLS_PARSER_REGISTRY.register('imagenet')
class ImageNetParser(BaseParser):
    def parse(self, meta_file, idx, start_index=0, rank_indices=None):
        metas = []
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
    def parse(self, meta_file, idx, start_index=0, rank_indices=None):
        metas = []
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
    def parse(self, meta_file, idx, start_index=0, rank_indices=None):
        min_size = self.extra_info.get("min_size", 40)
        metas = []
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
                 save_score=False,
                 multilabel=False,
                 label_mapping=None,
                 inference_cfg=None):
        super(ClsDataset, self).__init__(meta_file,
                                         image_reader,
                                         transformer,
                                         evaluator)
        if evaluator:
            self.topk = evaluator.get('kwargs', {}).get('topk', [1, 5])
        else:
            self.topk = [1, 5]
        self.label_mapping = label_mapping
        self.save_score = save_score
        self.meta_type = meta_type
        self.image_type = image_type
        self.fraction = fraction
        self.multilabel = multilabel
        self._list_check()
        self.done_imgs = set()
        self.inference_cfg = inference_cfg
        if self.inference_cfg is not None:
            self._inference_init()
        self.meta_parser = [CLS_PARSER_REGISTRY[m_type](**parser_info) for m_type in self.meta_type]
        self.parse_metas()
        if self.label_mapping is not None:
            assert len(self.label_mapping) == len(self.meta_file)

        self.labels = np.array([m['label'] for m in self.metas])
        self.classes, self.class_nums = np.unique(self.labels, return_counts=True)
        self.img_ids = np.arange(len(self.metas))

    def _list_check(self):
        if not isinstance(self.meta_file, list):
            self.meta_file = [self.meta_file]
        if not isinstance(self.meta_type, list):
            self.meta_type = [self.meta_type]

    def set_label_mapping(self, metas, idx):
        for meta in metas:
            meta['label'] += self.label_mapping[idx]

    @property
    def num_list(self):
        return len(self.meta_file)

    @property
    def images_per_list(self):
        _images_per_list = defaultdict(list)
        self._num_images_per_list = Counter()
        for img_idx, img_anns in enumerate(self.metas):
            image_source = img_anns.get('image_source', 0)
            _images_per_list[image_source].append(img_idx)
            self._num_images_per_list += Counter([image_source])
        return _images_per_list

    @property
    def num_images_per_list(self):
        if not hasattr(self, '_num_images_per_list'):
            self._num_images_per_list = Counter()
            for _, img_anns in enumerate(self.metas):
                image_source = img_anns.get('image_source', 0)
                self._num_images_per_list += Counter([image_source])
        return self._num_images_per_list

    def _inference_init(self):
        test_resume = self.inference_cfg.get('test_resume', None)
        if test_resume is not None:
            resume_dir = test_resume.get('resume_dir', 'results_dir')
            rank_set = test_resume.get('rank_set', False)
            indicator = test_resume.get('indicator', 'filename')
            self.test_resume = True
            self.done_imgs = test_resume_init(resume_dir, env.world_size, env.rank, rank_set, indicator)
        else:
            self.test_resume = False
            self.done_imgs = set()

        infer_from = self.inference_cfg.get('inference_from', None)
        if infer_from is not None:
            logger.info('You inference results from a imagepaths file or images folder')
            if os.path.isdir(infer_from):
                inference_files = self.inference_cfg.get('folder_images_save_path', 'imagepaths.txt')
                if env.is_master():
                    search_folder_and_save(infer_from, inference_files)
                barrier()
            elif os.path.isfile(infer_from):
                inference_files = infer_from
            else:
                raise NotImplementedError
            self.meta_file = [inference_files]

    def parse_metas(self):
        self.metas = []
        for idx, meta_file in enumerate(self.meta_file):
            if self.multilabel:
                metas = self.meta_parser[idx].parse(meta_file, idx, self.multilabel)
            else:
                metas = self.meta_parser[idx].parse(meta_file, idx)
            if len(self.done_imgs) != 0:
                filtered_metas = []
                for m in metas:
                    if m['filename'] not in self.done_imgs:
                        filtered_metas.append(m)
                metas = filtered_metas
            if self.label_mapping is not None:
                self.set_label_mapping(metas, idx)
            self.metas.extend(metas)
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
        label = self.tensor2numpy(output['gt'])
        # if label is a multilabel, use logits
        if len(label[0].shape) > 1:
            prediction = self.tensor2numpy(output['logits'])
        else:
            prediction = self.tensor2numpy(output['preds'])
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
            if self.multilabel and label.shape[0] != prediction.shape[0]:
                label = np.transpose(label, (1, 0, 2))
            score = [self.tensor2numpy(i) for i in scores]
        else:
            score = self.tensor2numpy(output['scores'])
        out_res = []
        for b_idx in range(label.shape[0]):
            _topk_idx, _topk_score = [], []
            if multicls:
                if len(prediction[0].shape) > 1:
                    _prediction = [item for item in prediction[b_idx]]
                else:
                    _prediction = [int(item) for item in prediction[:, b_idx]]
                if len(label[0].shape) > 1:
                    _label = [item for item in label[b_idx]]
                else:
                    _label = [int(item) for item in label[b_idx][:]]
                _filename = [filename for filename in filenames]
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
                'topk_score': _topk_score,
                'filename': _filename
            }
            if self.save_score:
                res['score'] = _score
            out_res.append(res)
        return out_res

    @property
    def images_per_class(self):
        image_list_per_class = defaultdict(list)
        for img_idx, meta in enumerate(self.metas):
            if isinstance(meta['label'], list):
                label = meta['label'][0]
            else:
                label = meta['label']
            image_list_per_class[label + 1].append(img_idx)
        return image_list_per_class

    @property
    def num_images_per_class(self):
        self._num_images_per_class = defaultdict(int)
        for meta in self.metas:
            if isinstance(meta['label'], list):
                label = meta['label'][0]
            else:
                label = meta['label']
            self._num_images_per_class[label + 1] += 1
        self.num_classes = len(self._num_images_per_class) + 1
        return self._num_images_per_class


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
                 random=True,
                 inference_cfg=None):
        self.mini_epoch = reload_cfg.get('mini_epoch', 1)
        self.seed = reload_cfg.get('seed', 0)
        self.mini_epoch_idx = reload_cfg.get('mini_epoch_idx', 0)
        self.group = reload_cfg.get('group', 1)
        self.world_size = env.world_size
        self.rank = env.rank
        self.random = random
        if (inference_cfg is not None) and (inference_cfg.get('test_resume', None) is not None):
            inference_cfg['test_resume'].update({'rank_set': True})
        super(RankClsDataset, self).__init__(meta_file,
                                             image_reader,
                                             transformer,
                                             evaluator,
                                             meta_type,
                                             parser_info,
                                             image_type,
                                             inference_cfg=inference_cfg)

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
            self.metas.extend(self.meta_parser[idx].parse(meta_file, idx, start_indexs[idx], rank_indices))
        if len(self.done_imgs) != 0:
            filtered_metas = []
            for m in self.metas:
                if m['filename'] not in self.done_imgs:
                    filtered_metas.append(m)
            self.metas = filtered_metas
        if len(rank_indices) != rank_num_samples:
            self.metas += self.metas[:(rank_num_samples - len(rank_indices))]
