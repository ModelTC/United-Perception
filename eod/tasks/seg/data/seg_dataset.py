
from eod.data.datasets.base_dataset import BaseDataset
from eod.utils.general.registry_factory import DATASET_REGISTRY
from easydict import EasyDict
from eod.data.image_reader import build_image_reader
from eod.utils.general.registry import Registry

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
        with open(meta_file, "r") as f:
            for line in f.readlines():
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
                 seg_label_reader=None):
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
            'gt_seg': seg_label
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
        seg_label = self.tensor2numpy(output['gt_seg'])
        out_res = []
        for _idx in range(pred.shape[0]):
            res = {
                'pred': pred[_idx],
                'gt_seg': seg_label[_idx]
            }
            out_res.append(res)
        return out_res
