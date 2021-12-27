from __future__ import division

# Standard Library

# Import from third library
from lvis import LVIS

# Import from eod
from eod.utils.general.registry_factory import DATASET_REGISTRY

# Import from local
from .coco_dataset import CocoDataset

__all__ = ['LvisDataset']


@DATASET_REGISTRY.register('lvis')
class LvisDataset(CocoDataset):
    """Lvis Dataset
    """
    _loader = LVIS

    def __init__(self,
                 meta_file,
                 image_reader,
                 transformer,
                 source='lvis',
                 has_mask=False,
                 has_grid=False,
                 use_ignore=False,
                 evaluator=None,
                 cache=None):
        super(LvisDataset, self).__init__(meta_file=meta_file,
                                          image_reader=image_reader,
                                          transformer=transformer,
                                          evaluator=evaluator,
                                          source=source,
                                          has_mask=has_mask,
                                          has_grid=has_grid,
                                          use_ignore=use_ignore,
                                          cache=cache)
        # hack the difference between LVIS and COCO API
        self.coco.imgToAnns = self.coco.img_ann_map
        self.coco.catToImgs = self.coco.cat_img_map
        self.lvis = self.coco

    def get_filename(self, meta_img):
        return meta_img['file_name'][13:] if meta_img['file_name'].startswith('COCO_val2014_') \
            else meta_img['file_name']


@DATASET_REGISTRY.register('lvisv1')
class LvisV1Dataset(LvisDataset):
    """Lvis Dataset
    """
    _loader = LVIS

    def __init__(self,
                 meta_file,
                 image_reader,
                 transformer,
                 source='lvis',
                 has_mask=False,
                 has_grid=False,
                 use_ignore=False,
                 evaluator=None,
                 min_size=1,
                 cache=None):
        super(LvisV1Dataset, self).__init__(meta_file=meta_file,
                                            image_reader=image_reader,
                                            transformer=transformer,
                                            evaluator=evaluator,
                                            source=source,
                                            has_mask=has_mask,
                                            has_grid=has_grid,
                                            use_ignore=use_ignore,
                                            cache=cache)
        self._filter_gts(min_size)

    def _filter_gts(self, min_size):
        # for mask rcnn gt box filtering
        for img_id in self.img_ids:
            meta_annos = self.coco.imgToAnns[img_id]
            if len(meta_annos) == 1:
                ann = meta_annos[0]
                if ann['bbox'][2] <= min_size or ann['bbox'][3] <= min_size:
                    self.img_ids.remove(img_id)

    def get_filename(self, meta_img):
        coco_url = meta_img['coco_url']
        filename = coco_url.replace('http://images.cocodataset.org/', '')
        return filename
