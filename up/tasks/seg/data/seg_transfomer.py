import numpy as np
import cv2
import collections
import numbers
import random
import math

from up.data.datasets.transforms import Augmentation
from up.utils.general.registry_factory import AUGMENTATION_REGISTRY


@AUGMENTATION_REGISTRY.register('seg_resize')
class SegResize(Augmentation):
    def __init__(self, size, **kwargs):
        super(Augmentation, self).__init__()
        assert (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = tuple(size)

    def augment(self, data):
        data['image'] = cv2.resize(data['image'], dsize=self.size, interpolation=cv2.INTER_LINEAR)
        data['gt_seg'] = cv2.resize(data['gt_seg'], dsize=self.size, interpolation=cv2.INTER_NEAREST)
        return data


@AUGMENTATION_REGISTRY.register('seg_rand_resize')
class SegRandResize(Augmentation):
    """
    Randomly resize image & label with scale factor in [scale_min, scale_max]
    """
    def __init__(self, scale, aspect_ratio=None):
        super(SegRandResize, self).__init__()
        assert (isinstance(scale, collections.Iterable) and len(scale) == 2)
        if isinstance(scale, collections.Iterable) and len(scale) == 2 \
                and isinstance(scale[0], numbers.Number) and isinstance(scale[1], numbers.Number):
            self.scale = scale
        else:
            raise (RuntimeError("segtransforms.RandScale() scale param error.\n"))
        if aspect_ratio is None:
            self.aspect_ratio = aspect_ratio
        elif isinstance(aspect_ratio, collections.Iterable) and len(aspect_ratio) == 2 \
                and isinstance(aspect_ratio[0], numbers.Number) and isinstance(aspect_ratio[1], numbers.Number) \
                and 0 < aspect_ratio[0] < aspect_ratio[1]:
            self.aspect_ratio = aspect_ratio
        else:
            raise (RuntimeError("segtransforms.RandScale() aspect_ratio param error.\n"))

    def augment(self, data):
        image = data['image']
        label = data['gt_seg']
        if random.random() < 0.5:
            temp_scale = self.scale[0] + (1. - self.scale[0]) * random.random()
        else:
            temp_scale = 1. + (self.scale[1] - 1.) * random.random()
        temp_aspect_ratio = 1.0
        if self.aspect_ratio is not None:
            temp_aspect_ratio = self.aspect_ratio[0] + (self.aspect_ratio[1] - self.aspect_ratio[0]) * random.random()
            temp_aspect_ratio = math.sqrt(temp_aspect_ratio)
        scale_factor_w = temp_scale * temp_aspect_ratio
        scale_factor_h = temp_scale / temp_aspect_ratio
        h, w, _ = image.shape
        new_w = int(w * scale_factor_w)
        new_h = int(h * scale_factor_h)
        data['image'] = cv2.resize(image, dsize=(new_w, new_h), interpolation=cv2.INTER_LINEAR)
        data['gt_seg'] = cv2.resize(label, dsize=(new_w, new_h), interpolation=cv2.INTER_NEAREST)
        return data


@AUGMENTATION_REGISTRY.register('seg_crop')
class SegCrop(Augmentation):
    """Crops the given tensor.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
        int instead of sequence like (h, w), a square crop (size, size) is made.
    """
    def __init__(self, size, crop_type='center', ignore_label=255):
        super(SegCrop, self).__init__()
        if isinstance(size, int):
            self.crop_h = size
            self.crop_w = size
        elif isinstance(size, collections.Iterable) and len(size) == 2 \
                and isinstance(size[0], int) and isinstance(size[1], int) \
                and size[0] > 0 and size[1] > 0:
            self.crop_h = size[0]
            self.crop_w = size[1]
        else:
            raise (RuntimeError("crop size error.\n"))
        if crop_type == 'center' or crop_type == 'rand':
            self.crop_type = crop_type
        else:
            raise (RuntimeError("crop type error: rand | center\n"))
        if isinstance(ignore_label, int):
            self.ignore_label = ignore_label
        else:
            raise (RuntimeError("ignore_label should be an integer number\n"))

    def augment(self, data):
        image = data['image']
        label = data['gt_seg']
        h, w, _ = image.shape
        pad_h = max(self.crop_h - h, 0)
        pad_w = max(self.crop_w - w, 0)
        if pad_h > 0 or pad_w > 0:
            image = cv2.copyMakeBorder(image, 0, pad_h, 0,
                                       pad_w, cv2.BORDER_CONSTANT,
                                       value=(0.0, 0.0, 0.0))
            label = cv2.copyMakeBorder(label, 0, pad_h, 0,
                                       pad_w, cv2.BORDER_CONSTANT,
                                       value=(self.ignore_label))
        h, w, _ = image.shape
        if self.crop_type == 'rand':
            h_off = random.randint(0, h - self.crop_h)
            w_off = random.randint(0, w - self.crop_w)
        else:
            h_off = (h - self.crop_h) // 2
            w_off = (w - self.crop_w) // 2

        data['image'] = np.asarray(image[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)
        data['gt_seg'] = np.asarray(label[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)
        return data


@AUGMENTATION_REGISTRY.register('seg_random_flip')
class SegRandomHorizontalFlip(Augmentation):
    def augment(self, data):
        image = data['image']
        label = data['gt_seg']
        flip = np.random.choice(2) * 2 - 1
        data['image'] = image[:, ::flip, :]
        data['gt_seg'] = label[:, ::flip]
        return data


@AUGMENTATION_REGISTRY.register('seg_rand_rotate')
class RandRotate(Augmentation):
    def augment(self, data):
        image = data['image']
        label = data['gt_seg']
        angle = random.random() * 20 - 10
        h, w = image.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        data['image'] = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR)
        data['gt_seg'] = cv2.warpAffine(label, rotation_matrix, (w, h), flags=cv2.INTER_NEAREST)
        return data


@AUGMENTATION_REGISTRY.register('seg_rand_blur')
class RandomGaussianBlur(Augmentation):
    def augment(self, data):
        gauss_size = random.choice([1, 3, 5, 7])
        if gauss_size > 1:
            # do the gaussian blur
            data['image'] = cv2.GaussianBlur(data['image'], (gauss_size, gauss_size), 0)

        return data
