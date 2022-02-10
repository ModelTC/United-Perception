from __future__ import division

# Standard Library
import copy
import os
from collections import defaultdict

# Import from third library
import numpy as np
import torch
from easydict import EasyDict
from PIL import Image
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from torch.nn.modules.utils import _pair
import pickle as pk
from eod.utils.env.dist_helper import env
import cv2


from eod.utils.general.context import no_print
from eod.utils.general.registry_factory import DATASET_REGISTRY
from eod.utils.general.global_flag import ALIGNED_FLAG
from eod.data.datasets.base_dataset import BaseDataset
from eod.data.data_utils import get_image_size
from eod.tasks.det.data.datasets.det_transforms import boxes2polygons


__all__ = ['CocoDataset']


CLASS_NAMES = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]

CLASS_PS_NAMES = CLASS_NAMES + [
    'banner', 'blanket', 'bridge', 'cardboard',
    'counter', 'curtain', 'door-stuff', 'floor-wood', 'flower', 'fruit',
    'gravel', 'house', 'light', 'mirror-stuff', 'net', 'pillow', 'platform',
    'playingfield', 'railroad', 'river', 'road', 'roof', 'sand', 'sea', 'shelf',
    'snow', 'stairs', 'tent', 'towel', 'wall-brick', 'wall-stone', 'wall-tile',
    'wall-wood', 'water-other', 'window-blind', 'window-other', 'tree-merged',
    'fence-merged', 'ceiling-merged', 'sky-other-merged', 'cabinet-merged',
    'table-merged', 'floor-other-merged', 'pavement-merged', 'mountain-merged',
    'grass-merged', 'dirt-merged', 'paper-merged', 'food-other-merged',
    'building-other-merged', 'rock-merged', 'wall-other-merged', 'rug-merged'
]


@DATASET_REGISTRY.register('coco')
class CocoDataset(BaseDataset):
    """COCO Dataset
    """
    _loader = COCO

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
                 clip_box=True):
        with no_print():
            self.coco = self._loader(meta_file)

        category_ids = self.coco.cats.keys()

        self.classes = ['__background__'] + [
            self.coco.cats[c]['name'] for c in sorted(category_ids)
        ]
        # map coco discrete category ids to contiguous ids
        self.category_to_class = {
            c: i + 1
            for i, c in enumerate(sorted(category_ids))
        }
        self.class_to_category = {
            i + 1: c
            for i, c in enumerate(sorted(category_ids))
        }

        super(CocoDataset, self).__init__(meta_file, image_reader, transformer,
                                          evaluator=evaluator, class_names=self.classes)
        self.has_keypoint = has_keypoint
        self.has_mask = has_mask
        self.has_grid = has_grid
        self.box2mask = box2mask
        self.use_ignore = use_ignore
        self.has_semantic_seg = has_semantic_seg
        self.semantic_seg_prefix = semantic_seg_prefix
        self.unknown_erasing = unknown_erasing
        self.clip_box = clip_box

        if self.box2mask:
            assert self.has_mask

        for img in self.coco.imgs.values():
            img['aspect_ratio'] = float(img['height']) / img['width']
        self.img_ids = list(set([_['image_id'] for _ in self.coco.anns.values()]))

        if len(self.img_ids) == 0:
            # len of img ids is 0 (might because this is test info), load from images'
            self.img_ids = list(self.coco.imgs.keys())

        self.img_ids = sorted(self.img_ids)
        self.aspect_ratios = [self.coco.imgs[ix]['aspect_ratio'] for ix in self.img_ids]

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
        pool.map(self.set_cache_images, self.img_ids)
        pool.close()
        pool.join()

    def set_cache_images(self, img_id):
        meta_img = self.coco.imgs[img_id]
        filename = self.get_filename(meta_img)
        with open(os.path.join(self.image_reader.image_directory(), filename), "rb") as f:
            img = np.frombuffer(f.read(), np.uint8)
            self.cache_image[filename] = img

    def get_cache_image(self, meta_img):
        filename = self.get_filename(meta_img)
        return self.cache_image[filename]

    def __len__(self):
        return len(self.img_ids)

    def get_filename(self, meta_img):
        return meta_img['file_name']

    def get_image_classes(self, img_index):
        """For Repeat Factor Sampler
        """
        img_id = self.img_ids[img_index]
        img_anns = self.coco.imgToAnns[img_id]
        return [
            self.category_to_class[ann['category_id']]
            for ann in img_anns if not ann.get('iscrowd', False)
        ]

    @property
    def images_per_category(self):
        """The results is volatile since it may consume too much memory
        """
        coco_img_id_to_img_index = {
            img_id: img_index
            for img_index, img_id in enumerate(self.img_ids)
        }
        to_img_index = lambda img_ids: set([coco_img_id_to_img_index[img_id] for img_id in img_ids])
        return {
            cat: list(to_img_index(img_ids))
            for cat, img_ids in self.coco.catToImgs.items()
        }

    @property
    def num_images_per_category(self):
        if hasattr(self, '_num_images_per_category'):
            return self._num_images_per_category
        self._num_images_per_category = {
            cat: len(set(img_list))
            for cat, img_list in self.coco.catToImgs.items()
        }
        return self._num_images_per_category

    @property
    def num_instances_per_category(self):
        if hasattr(self, '_num_instances_per_category'):
            return self._num_instances_per_category
        self._num_instances_per_category = defaultdict(int)
        for ann in self.coco.anns.values():
            self._num_instances_per_category[ann['category_id']] += 1
        return self._num_images_per_category

    @property
    def images_per_class(self):
        return {
            self.category_to_class[cat]: images
            for cat, images in self.images_per_category.items()
        }

    @property
    def num_images_per_class(self):
        """ For Class Aware Balanced Sampler and Repeat Factor Sampler
        """
        return {
            self.category_to_class[cat]: num
            for cat, num in self.num_images_per_category.items()
        }

    @property
    def num_instances_per_class(self):
        return {
            self.category_to_class[cat]: num
            for cat, num in self.num_instances_per_category.items()
        }

    def get_input(self, idx):
        """ parse annotation into input dict
        """
        img_id = self.img_ids[idx]
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
            'dataset_idx': idx
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
                # (FloatTensor): [N, num_keyps, 3] (x, y, flag)
                'gt_keyps': <tensor>,
                # (list of list of ndarray): [N] [polygons]
                'gt_masks': [],
                #  (list of tuple): [(1, 2), (3, 4), ...], for visualization
                'keyp_pairs': [],
                # (str): image name
                'filename': ..
            }
        """
        input = self.get_input(idx)
        image_h, image_w = get_image_size(input.image)
        input = self.transformer(input)

        if self.has_grid:
            gt_grids = self._generate_grids(input.gt_bboxes)
            input.gt_grids = torch.as_tensor(gt_grids, dtype=torch.float32)

        scale_factor = input.get('scale_factor', 1)
        new_image_h, new_image_w = get_image_size(input.image)
        pad_w, pad_h = input.get('dw', 0), input.get('dh', 0)
        image_info = [new_image_h, new_image_w, scale_factor, image_h, image_w, input.flipped, pad_w, pad_h]
        input.image_info = image_info
        return input

    def _fake_zero_data(self, *size):
        return torch.zeros(size)

    def dump(self, output):
        """
        Write predicted results into files.

        .. note::

            Masks are binaried by threshold 0.5 and converted to rle string

        Arguments:
            - writer: output stream to write results
            - image_info (FloatTensor): [B, 5] (resized_h, resized_w, scale_factor, origin_h, origin_w)
            - bboxes (FloatTensor): [N, 7] (batch_idx, x1, y1, x2, y2, score, cls)
            - keypoints (FloatTensor): [N, K, 3], (x, y, score)
            - masks (list FloatTensor): [N, h, w]
        """
        # filenames = output['filenames']
        # image_info = self.tensor2numpy(output['image_info'])
        image_info = output['image_info']
        bboxes = self.tensor2numpy(output['dt_bboxes'])
        keypoints = self.tensor2numpy(output.get('dt_keyps', None))
        masks = self.tensor2numpy(output.get('dt_masks', None))
        image_ids = output['image_id']
        if len(bboxes) == 0:
            return

        dump_results = []
        for b_ix in range(len(image_info)):
            info = image_info[b_ix]
            img_h, img_w = map(int, info[3: 5])
            img_id = image_ids[b_ix]

            scores = bboxes[:, 5]
            keep_ix = np.where(bboxes[:, 0] == b_ix)[0]
            keep_ix = sorted(keep_ix, key=lambda ix: scores[ix], reverse=True)

            scale_h, scale_w = _pair(info[2])
            img_bboxes = bboxes[keep_ix]
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

            x = img_bboxes[:, 1][:, None]
            y = img_bboxes[:, 2][:, None]
            w = (img_bboxes[:, 3] - img_bboxes[:, 1] + ALIGNED_FLAG.offset)[:, None]
            h = (img_bboxes[:, 4] - img_bboxes[:, 2] + ALIGNED_FLAG.offset)[:, None]
            res_bboxes = np.concatenate([x, y, w, h], axis=1)

            if keypoints is not None:
                img_keyps = keypoints[keep_ix]
                img_keyps[..., 0] /= scale_w
                img_keyps[..., 1] /= scale_h
                img_keyps = img_keyps.reshape(img_keyps.shape[0], -1)

            for idx in range(len(img_bboxes)):
                res = {'image_id': img_id}
                box_score, cls = img_bboxes[idx][5:7]
                res['bbox'] = res_bboxes[idx].tolist()
                res['score'] = float(box_score)
                res['category_id'] = self.class_to_category.get(int(cls), int(cls))

                if keypoints is not None:
                    res['keypoints'] = img_keyps[idx].tolist()

                if masks is not None:
                    # We have resized mask to origin image size in mask_head fo fast inference,
                    mask = np.asfortranarray(masks[keep_ix[idx]], dtype=np.uint8)
                    rle = maskUtils.encode(mask)
                    if isinstance(rle['counts'], bytes):
                        rle['counts'] = str(rle['counts'], encoding='utf-8')
                    res['segmentation'] = rle

                dump_results.append(res)
        return dump_results
