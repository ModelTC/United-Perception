import json
import numpy as np
import os
from easydict import EasyDict

from up.data.datasets.base_dataset import BaseDataset
from up.utils.general.registry_factory import DATASET_REGISTRY
from up.data.image_reader import build_image_reader
from up.utils.general.registry import Registry
from up.utils.general.petrel_helper import PetrelHelper
from up.utils.general.log_helper import default_logger as logger
from up.utils.env.dist_helper import env

from .seg_evaluator import intersectionAndUnion

__all__ = ['SegDataset']

SEG_PARSER_REGISTRY = Registry()


class BaseParser(object):
    def __init__(self, extra_info={}):
        self.extra_info = extra_info

    def parse(self):
        raise NotImplementedError


@SEG_PARSER_REGISTRY.register('cityscapes')
class CityScapesParser(BaseParser):
    def parse(self, meta_file, idx, metas):
        with PetrelHelper.open(meta_file) as fin:
            for line in fin:
                cls_res = {}
                spts = line.strip().split()
                cls_res['filename'] = spts[0]
                cls_res['seg_label_filename'] = spts[1]
                cls_res['image_source'] = idx
                metas.append(cls_res)
        return metas


@DATASET_REGISTRY.register('seg')
class SegDataset(BaseDataset):
    def __init__(self,
                 meta_file,
                 image_reader,
                 transformer=None,
                 evaluator=None,
                 seg_type='cityscapes',
                 parser_info={},
                 seg_label_reader=None,
                 output_pred=False,
                 ignore_label=255,
                 num_classes=19):
        super(SegDataset, self).__init__(meta_file,
                                         image_reader,
                                         transformer,
                                         evaluator)
        self.seg_type = seg_type
        assert seg_label_reader is not None
        self.seg_label_reader = build_image_reader(seg_label_reader)
        self._list_check()
        self.meta_parser = [SEG_PARSER_REGISTRY[m_type](**parser_info) for m_type in self.seg_type]
        self.parse_metas()
        self.output_pred = output_pred
        self.ignore_label = ignore_label
        self.num_classes = num_classes

    def parse_metas(self):
        self.metas = []
        for idx, meta_file in enumerate(self.meta_file):
            self.meta_parser[idx].parse(meta_file, idx, self.metas)

    def _list_check(self):
        if not isinstance(self.meta_file, list):
            self.meta_file = [self.meta_file]
        if not isinstance(self.seg_type, list):
            self.seg_type = [self.seg_type]

    def __len__(self):
        return len(self.metas)

    def _load_meta(self, idx):
        return self.metas[idx]

    def get_input(self, idx):
        meta = self._load_meta(idx)
        img = self.image_reader(meta['filename'], meta.get('image_source', 0))
        seg_label = self.seg_label_reader(meta['seg_label_filename'], meta.get('image_source', 0))
        input = EasyDict({
            'image': img,
            'gt_semantic_seg': seg_label
        })
        return input

    def __getitem__(self, idx):
        input = self.get_input(idx)
        if self.transformer is not None:
            input = self.transformer(input)
        input.image_info = input['image'].size()
        return input

    def dump(self, output):
        pred = output['blob_pred'].max(1)[1]
        pred = self.tensor2numpy(pred)
        if 'gt_semantic_seg' in output and output['gt_semantic_seg'] is not None:
            seg_label = self.tensor2numpy(output['gt_semantic_seg'])
        else:
            seg_label = np.zeros((pred.shape))
        out_res = []
        for _idx in range(pred.shape[0]):
            if 'gt_semantic_seg' in output and output['gt_semantic_seg'] is not None:
                inter, union, target = intersectionAndUnion(pred[_idx],
                                                            seg_label[_idx],
                                                            self.num_classes,
                                                            self.ignore_label)
                res = {
                    'inter': inter,
                    'union': union,
                    'target': target
                }
            else:
                res = {}
            if self.output_pred:
                res['pred'] = pred[_idx]
            out_res.append(res)
        return out_res


@DATASET_REGISTRY.register('custom_seg')
class CustomSeg(BaseDataset):
    def __init__(self,
                 meta_file,
                 image_reader,
                 transformer,
                 num_classes,
                 evaluator=None,
                 seg_label_reader=None,
                 output_pred=False,
                 ignore_label=255,
                 class_names=None,
                 mask_dump_dir=None):
        # mask reader
        if isinstance(seg_label_reader, str) or seg_label_reader is None:
            mc = image_reader['kwargs'].get('memcached', True)
            seg_label_reader = {
                'type': 'ceph_opencv',
                'kwargs': {
                    'image_dir': seg_label_reader if seg_label_reader else image_reader['kwargs']['image_dir'],
                    'color_mode': 'GRAY',
                    'memcached': mc,
                }
            }
        self.seg_label_reader = build_image_reader(seg_label_reader)

        super(CustomSeg, self).__init__(meta_file,
                                        image_reader,
                                        transformer,
                                        evaluator=evaluator,
                                        class_names=class_names)

        self._normal_init(meta_file)
        self.num_classes = num_classes
        self.output_pred = output_pred

        self.mask_dump_dir = mask_dump_dir
        self.ignore_label = ignore_label

        if (env.is_master() or env.world_size == 1) and mask_dump_dir:
            os.makedirs(mask_dump_dir, exist_ok=True)

        if len(self.aspect_ratios) == 0:
            self.aspect_ratios = [1] * len(self.metas)

    def _normal_init(self, meta_files):
        if isinstance(meta_files, str):
            meta_files = [meta_files]

        self.metas = []
        self.aspect_ratios = []

        for idx, meta_file in enumerate(meta_files):
            with PetrelHelper.open(meta_file) as fin:
                for line in fin:
                    meta = json.loads(line)
                    meta['image_source'] = idx
                    self.metas.append(meta)

                    if 'image_height' not in meta or 'image_width' not in meta:
                        logger.warning('image size is not provided, ' 'set aspect grouping to 1.')
                        self.aspect_ratios.append(1.0)
                    else:
                        self.aspect_ratios.append(meta['image_height'] / meta['image_width'])

    def _load_meta(self, idx):
        return self.metas[idx]

    def __len__(self):
        return len(self.metas)

    def get_input(self, idx):
        meta = self._load_meta(idx)
        img_id = filename = meta['filename']
        mask_filename = meta['mask_filename']
        img = self.image_reader(meta['filename'], meta.get('image_source', 0))
        gt_semantic_seg = self.seg_label_reader(meta['mask_filename'], meta.get('image_source', 0))
        input = EasyDict({
            'filename': filename,
            'image_id': img_id,
            'image': img,
            'mask_filename': mask_filename,
            'gt_semantic_seg': gt_semantic_seg,
            'flipped': False,
            'data_type': 0,
        })
        return input

    def __getitem__(self, idx):
        # idx=0
        input = self.get_input(idx)
        # image_h, image_w = get_image_size(input.image)
        if self.transformer is not None:
            input = self.transformer(input)
        image_info = input['image'].size()
        # scale_factor = input.get('scale_factor', 1)
        # new_image_h, new_image_w = get_image_size(input.image)
        # pad_w, pad_h = 0, 0
        # image_info = [
        #     new_image_h, new_image_w, scale_factor, image_h, image_w, input.flipped, pad_w, pad_h, self.ignore_label
        # ]
        input.image_info = image_info
        return input

    def dump(self, output):
        pred = output['blob_pred'].max(1)[1]
        pred = self.tensor2numpy(pred)
        if 'gt_semantic_seg' in output and output['gt_semantic_seg'] is not None:
            seg_label = self.tensor2numpy(output['gt_semantic_seg'])
        else:
            seg_label = np.zeros((pred.shape))
        out_res = []
        for _idx in range(pred.shape[0]):
            if 'gt_semantic_seg' in output and output['gt_semantic_seg'] is not None:
                inter, union, target = intersectionAndUnion(pred[_idx],
                                                            seg_label[_idx],
                                                            self.num_classes,
                                                            self.ignore_label)
                res = {
                    'inter': inter,
                    'union': union,
                    'target': target
                }
            else:
                res = {}
            if self.output_pred:
                res['pred'] = pred[_idx]
            out_res.append(res)
        return out_res
