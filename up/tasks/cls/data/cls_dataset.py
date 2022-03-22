from audioop import mul
import json
import numpy as np
from up.data.datasets.base_dataset import BaseDataset
from up.utils.general.registry_factory import DATASET_REGISTRY
from easydict import EasyDict
from PIL import Image
from up.utils.general.petrel_helper import PetrelHelper
from up.data.data_utils import is_numpy_image
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
    def parse(self, meta_file, idx, metas):
        with PetrelHelper.open(meta_file) as f:
            for line in f:
                cls_res = {}
                filename, label = line.strip().split()
                cls_res['filename'] = filename
                cls_res['label'] = int(label)
                cls_res['image_source'] = idx
                metas.append(cls_res)
        return metas


@CLS_PARSER_REGISTRY.register('custom_cls')
class CustomClsParser(BaseParser):
    def parse(self, meta_file, idx, metas):
        with PetrelHelper.open(meta_file) as f:
            for line in f:
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
    def parse(self, meta_file, idx, metas):
        min_size = self.extra_info.get("min_size", 40)
        with PetrelHelper.open(meta_file) as f:
            for line in f:
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
        score = self.tensor2numpy(output['scores'])
        multicls = False
        if isinstance(prediction, list):
            multicls = True
            prediction = np.array(prediction)
            score = np.array(score)
        if multicls:
            bs = prediction[0].shape[0]
        else:
            bs = prediction.shape[0]
        out_res = []
        for b_idx in range(bs):
            if multicls:
                _prediction = [int(item) for item in prediction[:, b_idx]]
                _label = [int(item) for item in label[b_idx][:]]
                _score = []
                for item in score[:, b_idx]:
                    _score.append([float('%.8f' % s) for s in item])
            else:
                _prediction = int(prediction[b_idx])
                _label = int(label[b_idx])
                _score = [float('%.8f' % s) for s in score[b_idx]]
            res = {
                'prediction': _prediction,
                'label': _label,
                'score': _score,
            }
            out_res.append(res)
        return out_res
