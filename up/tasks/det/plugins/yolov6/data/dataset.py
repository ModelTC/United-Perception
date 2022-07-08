import os
import cv2
import copy
import numpy as np
from PIL import Image
from easydict import EasyDict

import torch

from up.utils.general.global_flag import ALIGNED_FLAG
from up.tasks.det.data.datasets.coco_dataset import CocoDataset
from up.utils.general.registry_factory import DATASET_REGISTRY
from up.tasks.det.data.datasets.det_transforms import boxes2polygons


@DATASET_REGISTRY.register('yolov6_val')
class Yolov6Dataset(CocoDataset):
    def __init__(self,
                 meta_file,
                 image_reader,
                 transformer,
                 source='coco',
                 class_names=None,
                 has_keypoint=False,
                 has_mask=False,
                 has_grid=False,
                 box2mask=False,
                 has_semantic_seg=False,
                 semantic_seg_prefix=None,
                 unknown_erasing=False,
                 use_ignore=False,
                 evaluator=None,
                 cache=None,
                 clip_box=True,
                 batch_size=32,
                 img_size=640,
                 stride=32):
        super(Yolov6Dataset, self).__init__(
            meta_file,
            image_reader,
            transformer,
            source,
            class_names,
            has_keypoint,
            has_mask,
            has_grid,
            box2mask,
            has_semantic_seg,
            semantic_seg_prefix,
            unknown_erasing,
            use_ignore,
            evaluator,
            cache,
            clip_box,
        )
        ar = np.array(self.aspect_ratios, dtype=np.float64)
        ind_sort = ar.argsort()
        self.ar_sort = ar[ind_sort]
        self.img_ids = [self.img_ids[i] for i in ind_sort]

        batch_ind = np.floor(np.arange(len(self.img_ids)) / batch_size).astype(np.int)
        batch_num = batch_ind[-1] + 1

        self.scales = []
        for i in range(batch_num):
            scales_i = [[1, 1]]
            ari = self.ar_sort[batch_ind == i]
            ar_min, ar_max = ari.min(), ari.max()
            if ar_max < 1:
                scales_i[0] = [ar_max, 1]
            elif ar_min > 1:
                scales_i[0] = [1, 1 / ar_min]
            self.scales.extend(scales_i * batch_size)

        self.shapes = np.ceil(np.array(self.scales) * img_size / stride + 0.5).astype(np.int) * stride

    def get_input(self, idx):
        """ parse annotation into input dict
        """
        img_id = self.img_ids[idx]
        tgt_shape = self.shapes[idx]
        meta_img = self.coco.imgs[img_id]
        meta_annos = self.coco.imgToAnns[img_id]
        # filename = os.path.join(self.image_dir, meta_img['file_name'])
        filename = self.get_filename(meta_img)

        gt_bboxes, ig_bboxes = [], []
        gt_keyps = [] if self.has_keypoint else None
        gt_masks = [] if self.has_mask else None
        gt_semantic_seg = np.zeros((meta_img['height'], meta_img['width']),
                                   dtype=np.uint8) if self.has_semantic_seg else None
        if self.has_semantic_seg:
            gt_semantic_seg = np.array(
                Image.open(os.path.join(self.semantic_seg_prefix, os.path.basename(filename).replace('jpg', 'png'))))

        for ann in meta_annos:
            # to keep compatible with lvis dataset
            if ann.get('iscrowd', False):
                if self.use_ignore:
                    # avoid inplace changing the original data
                    bbox = copy.copy(ann['bbox'])
                    bbox[2] += bbox[0] - ALIGNED_FLAG.offset
                    bbox[3] += bbox[1] - ALIGNED_FLAG.offset
                    ig_bboxes.append(bbox)
                continue
            label = self.category_to_class[ann['category_id']]
            bbox = ann['bbox'] + [label]
            bbox[2] += bbox[0] - ALIGNED_FLAG.offset
            bbox[3] += bbox[1] - ALIGNED_FLAG.offset
            gt_bboxes.append(bbox)

            if self.has_mask:
                if self.box2mask:
                    polygons = [boxes2polygons([bbox], sample=4 * 4).reshape(-1)]
                    gt_masks.append(polygons)
                else:
                    polygons = [np.array(poly, dtype=np.float32) for poly in ann['segmentation']]
                    gt_masks.append(polygons)

        if len(ig_bboxes) == 0:
            ig_bboxes = self._fake_zero_data(1, 4)
        if len(gt_bboxes) == 0:
            gt_bboxes = self._fake_zero_data(1, 5)

        ig_bboxes = torch.as_tensor(ig_bboxes, dtype=torch.float32) if self.use_ignore else None
        gt_bboxes = torch.as_tensor(gt_bboxes, dtype=torch.float32)
        gt_keyps = torch.stack(gt_keyps) if gt_keyps else None
        gt_masks = gt_masks if gt_masks else None  # cannot be a tensor
        tgt_shape = torch.as_tensor(tgt_shape, dtype=torch.float32)
        try:
            if self.cache is not None:
                img = self.get_cache_image(meta_img)
                img = cv2.imdecode(img, cv2.IMREAD_COLOR)
                if self.image_reader.color_mode != 'BGR':
                    cvt_color = getattr(cv2, 'COLOR_BGR2{}'.format(self.image_reader.color_mode))
                    img = cv2.cvtColor(img, cvt_color)
            else:
                img = self.image_reader(filename)
        except:  # noqa
            img = self.image_reader(filename)

        input = EasyDict({
            'image': img,
            'gt_bboxes': gt_bboxes,
            'gt_ignores': ig_bboxes,
            'gt_keyps': gt_keyps,
            'gt_masks': gt_masks,
            'gt_semantic_seg': gt_semantic_seg,
            'flipped': False,
            'filename': filename,
            'image_id': img_id,
            'dataset_idx': idx,
            'tgt_shape': tgt_shape,
        })
        return input
