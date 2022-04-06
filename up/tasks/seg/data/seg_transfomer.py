import numpy as np
import cv2
import collections
import numbers
import random
import math
import copy

from up.data.datasets.transforms import Augmentation
from up.utils.general.registry_factory import AUGMENTATION_REGISTRY


@AUGMENTATION_REGISTRY.register('color_jitter_mmseg')
class RandomColorJitterMMSeg(Augmentation):

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18,
                 color_type='BGR'):
        super(RandomColorJitterMMSeg, self).__init__()
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta
        self.color2hsv = getattr(cv2, 'COLOR_{}2HSV'.format(color_type))
        self.hsv2color = getattr(cv2, 'COLOR_HSV2{}'.format(color_type))

    def convert(self, img, alpha=1, beta=0):
        """Multiple with alpha and add beat with clip."""
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, img):
        """Brightness distortion."""
        if random.randint(0, 2):
            return self.convert(
                img,
                beta=random.uniform(-self.brightness_delta,
                                    self.brightness_delta))
        return img

    def contrast(self, img):
        """Contrast distortion."""
        if random.randint(0, 2):
            return self.convert(
                img,
                alpha=random.uniform(self.contrast_lower, self.contrast_upper))
        return img

    def saturation(self, img):
        """Saturation distortion."""
        if random.randint(0, 2):
            img = cv2.cvtColor(img, self.color2hsv)
            img[:, :, 1] = self.convert(
                img[:, :, 1],
                alpha=random.uniform(self.saturation_lower,
                                     self.saturation_upper))
            img = cv2.cvtColor(img, self.hsv2color)
        return img

    def hue(self, img):
        """Hue distortion."""
        if random.randint(0, 2):
            img = cv2.cvtColor(img, self.color2hsv)
            img[:, :,
                0] = (img[:, :, 0].astype(int)
                      + random.randint(-self.hue_delta, self.hue_delta)) % 180
            img = cv2.cvtColor(img, self.hsv2color)
        return img

    def augment(self, data):
        """
        Arguments:
            img (np.array): Input image.
        Returns:
            img (np.array): Color jittered image.
        """
        output = copy.copy(data)
        img = data.image
        assert isinstance(img, np.ndarray)
        img = np.uint8(img)

        # random brightness
        img = self.brightness(img)
        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = random.randint(0, 2)
        if mode == 1:
            img = self.contrast(img)

        # random saturation
        img = self.saturation(img)
        # random hue
        img = self.hue(img)
        # random contrast
        if mode == 0:
            img = self.contrast(img)

        img = np.asanyarray(img)
        output.image = img
        return output

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness_delta)
        format_string += ', contrast=({0},{1})'.format(self.contrast_lower, self.contrast_upper)
        format_string += ', saturation=({0},{1})'.format(self.saturation_lower, self.saturation_upper)
        format_string += ', hue={0})'.format(self.hue_delta)
        return format_string


@AUGMENTATION_REGISTRY.register('seg_resize')
class SegResize(Augmentation):
    def __init__(self, size, **kwargs):
        super(Augmentation, self).__init__()
        assert (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = tuple(size)

    def augment(self, data):
        data['image'] = cv2.resize(data['image'], dsize=self.size, interpolation=cv2.INTER_LINEAR)
        data['gt_semantic_seg'] = cv2.resize(data['gt_semantic_seg'], dsize=self.size, interpolation=cv2.INTER_NEAREST)
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
        label = data['gt_semantic_seg']
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
        data['gt_semantic_seg'] = cv2.resize(label, dsize=(new_w, new_h), interpolation=cv2.INTER_NEAREST)
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
        label = data['gt_semantic_seg']
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
        data['gt_semantic_seg'] = np.asarray(label[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)
        return data


@AUGMENTATION_REGISTRY.register('seg_random_flip')
class SegRandomHorizontalFlip(Augmentation):
    def augment(self, data):
        image = data['image']
        label = data['gt_semantic_seg']
        flip = np.random.choice(2) * 2 - 1
        data['image'] = image[:, ::flip, :]
        data['gt_semantic_seg'] = label[:, ::flip]
        return data


@AUGMENTATION_REGISTRY.register('seg_rand_rotate')
class RandRotate(Augmentation):
    def augment(self, data):
        image = data['image']
        label = data['gt_semantic_seg']
        angle = random.random() * 20 - 10
        h, w = image.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        data['image'] = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR)
        data['gt_semantic_seg'] = cv2.warpAffine(label, rotation_matrix, (w, h), flags=cv2.INTER_NEAREST)
        return data


@AUGMENTATION_REGISTRY.register('seg_rand_blur')
class RandomGaussianBlur(Augmentation):
    def augment(self, data):
        gauss_size = random.choice([1, 3, 5, 7])
        if gauss_size > 1:
            # do the gaussian blur
            data['image'] = cv2.GaussianBlur(data['image'], (gauss_size, gauss_size), 0)

        return data


@AUGMENTATION_REGISTRY.register('seg_rand_brightness')
class Random_Brightness(Augmentation):
    def __init__(self, shift_value=10):
        super().__init__()
        self.shift_value = shift_value

    def augment(self, data):
        if random.random() < 0.5:
            return data
        image = data['image']
        image = image.astype(np.float32)
        shift = random.randint(-self.shift_value, self.shift_value)
        image[:, :, :] += shift
        image = np.around(image)
        image = np.clip(image, 0, 255).astype(np.uint8)
        data['image'] = image
        return data
