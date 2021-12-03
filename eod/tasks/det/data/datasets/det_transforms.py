from __future__ import division

# Standard Library
import copy
import math
from collections.abc import Sequence
import torchvision.transforms as transforms
import numbers
import sys
from PIL import Image

# Import from third library
import cv2
import numpy as np
import random
import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import (
    adjust_brightness,
    adjust_contrast,
    adjust_hue,
    adjust_saturation
)
from eod.utils.general.registry_factory import AUGMENTATION_REGISTRY, BATCHING_REGISTRY
from eod.utils.general.global_flag import ALIGNED_FLAG
from eod.data.datasets.transforms import (
    Augmentation,
    has_gt_bboxes,
    has_gt_ignores,
    has_gt_keyps,
    has_gt_masks,
    has_gt_semantic_seg,
    check_fake_gt
)
from eod.data.data_utils import (
    coin_tossing,
    get_image_size,
    is_numpy_image,
    is_pil_image
)

# TODO: Check GPU usage after move this setting down from upper line
cv2.ocl.setUseOpenCL(False)


__all__ = [
    'Flip',
    'KeepAspectRatioResize',
    'KeepAspectRatioResizeMax',
    'BatchPad',
    'ImageExpand',
    'RandomColorJitter',
    'FixOutputResize',
    'ImageCrop',
    'ImageStitchExpand'
]


def tensor2numpy(data):
    if check_fake_gt(data.gt_bboxes) and check_fake_gt(data.gt_ignores):
        return np.zeros((0, 5))
    if data.gt_bboxes is None:
        gts = np.zeros((0, 5))
    else:
        gts = data.gt_bboxes.cpu().numpy()
    if data.gt_ignores is not None:
        igs = data.gt_ignores.cpu().numpy()
    else:
        igs = np.zeros((0, 4))
    if igs.shape[0] != 0:
        ig_bboxes = np.hstack([igs, -np.ones(igs.shape[0])[:, np.newaxis]])
        new_gts = np.concatenate((gts, ig_bboxes))
    else:
        new_gts = gts
    return new_gts


def numpy2tensor(boxes_t):
    ig_bboxes_t = boxes_t[boxes_t[:, 4] == -1][:, :4]
    gt_bboxes_t = boxes_t[boxes_t[:, 4] != -1]
    gt_bboxes_t = torch.as_tensor(gt_bboxes_t, dtype=torch.float32)
    ig_bboxes_t = torch.as_tensor(ig_bboxes_t, dtype=torch.float32)
    if len(ig_bboxes_t) == 0:
        ig_bboxes_t = torch.zeros((1, 4))
    if len(gt_bboxes_t) == 0:
        gt_bboxes_t = torch.zeros((1, 5))
    return gt_bboxes_t, ig_bboxes_t


def np_bbox_iof_overlaps(b1, b2):
    area1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
    inter_xmin = np.maximum(b1[:, 0].reshape(-1, 1), b2[:, 0].reshape(1, -1))
    inter_ymin = np.maximum(b1[:, 1].reshape(-1, 1), b2[:, 1].reshape(1, -1))
    inter_xmax = np.minimum(b1[:, 2].reshape(-1, 1), b2[:, 2].reshape(1, -1))
    inter_ymax = np.minimum(b1[:, 3].reshape(-1, 1), b2[:, 3].reshape(1, -1))
    inter_h = np.maximum(inter_xmax - inter_xmin, 0)
    inter_w = np.maximum(inter_ymax - inter_ymin, 0)
    inter_area = inter_h * inter_w
    return inter_area / np.maximum(area1[:, np.newaxis], 1)


def boxes2polygons(boxes, sample=4):
    num_boxes = len(boxes)
    boxes = np.array(boxes).reshape(num_boxes, -1)
    a = (boxes[:, 2] - boxes[:, 0] + 1) / 2
    b = (boxes[:, 3] - boxes[:, 1] + 1) / 2
    x = (boxes[:, 2] + boxes[:, 0]) / 2
    y = (boxes[:, 3] + boxes[:, 1]) / 2

    angle = 2 * math.pi / sample
    polygons = np.zeros([num_boxes, sample, 2], dtype=np.float32)
    for i in range(sample):
        polygons[:, i, 0] = a * math.cos(i * angle) + x
        polygons[:, i, 1] = b * math.sin(i * angle) + y
    return polygons


@AUGMENTATION_REGISTRY.register('flip')
class Flip(Augmentation):
    """
    """

    def __init__(self, flip_p):
        super(Flip, self).__init__()
        self.flip_p = flip_p

    def augment(self, data):
        if not coin_tossing(self.flip_p):
            return data

        output = copy.copy(data)
        output.image = self.flip_image(data.image)
        height, width = get_image_size(data.image)
        # height, width = data.image.shape[:2]
        if has_gt_bboxes(data):
            output.gt_bboxes = self.flip_boxes(data.gt_bboxes, width)
        if has_gt_ignores(data):
            output.gt_ignores = self.flip_boxes(data.gt_ignores, width)
        if has_gt_keyps(data):
            output.gt_keyps = self.flip_keyps(data.gt_keyps, data.keyp_pairs, width)
        if has_gt_masks(data):
            output.gt_masks = self.flip_masks(data.gt_masks, width)
        if has_gt_semantic_seg(data):
            output.gt_semantic_seg = self.flip_semantic_seg(data.gt_semantic_seg)
        output.flipped = True
        return output

    def flip_image(self, img):
        if is_pil_image(img):
            return TF.hflip(img)
        elif is_numpy_image(img):
            return cv2.flip(img, 1)
        else:
            raise TypeError('{} format not supported'.format(type(img)))

    def flip_boxes(self, boxes, width):
        x1 = boxes[:, 0].clone().detach()
        x2 = boxes[:, 2].clone().detach()
        boxes[:, 0] = width - ALIGNED_FLAG.offset - x2
        boxes[:, 2] = width - ALIGNED_FLAG.offset - x1
        return boxes

    def flip_masks(self, polygons, width):
        """Actuall flip polygons"""
        flipped_masks = []
        for polygon in polygons:
            flipped = []
            for poly in polygon:
                p = poly.copy()
                p[0::2] = width - poly[0::2] - ALIGNED_FLAG.offset
                flipped.append(p)
            flipped_masks.append(flipped)
        return flipped_masks

    def flip_keyps(self, keyps, pairs, width):
        if keyps.nelement() == 0:
            return keyps

        N, K = keyps.size()[:2]
        keyps = keyps.view(-1, 3)
        labeled = torch.nonzero(keyps[:, 2] > 0)
        if labeled.size(0) == 0:
            return keyps.view(N, K, 3)

        labeled = labeled.view(1, -1)[0]
        keyps[labeled, 0] = width - ALIGNED_FLAG.offset - keyps[labeled, 0]
        keyps = keyps.view(N, K, 3)
        clone_keyps = keyps.clone().detach()
        for left, right in pairs:
            # keyp_left = keyps[:, left, :].clone()
            # keyps[:, left, :] = keyps[:, right, :]
            # keyps[: right, :] = keyp_left
            keyps[:, left, :] = keyps[:, right, :]
            keyps[:, right, :] = clone_keyps[:, left, :]
        return keyps

    def flip_semantic_seg(self, semantic_seg):
        return self.flip_image(semantic_seg)


class Resize(Augmentation):
    def __init__(self, scales, max_size=sys.maxsize, separate_wh=False):
        """
        Args:
            scales: evaluable str, e.g. range(500,600,1) or given list
            max_size: int
        """
        super(Augmentation, self).__init__()
        if isinstance(scales, str):
            self.scales = list(eval(scales))
        else:
            assert isinstance(scales, Sequence)
            self.scales = scales
        self.max_size = max_size
        self.separate_wh = separate_wh

    def augment(self, data, scale_factor=None):
        if scale_factor is None:
            # img_h, img_w = data.image.shape[:2]
            img_h, img_w = get_image_size(data.image)
            scale_factor = self.get_scale_factor(img_h, img_w)
        output = copy.copy(data)
        output.image = self.resize_image(data.image, scale_factor)
        if self.separate_wh:
            h, w = get_image_size(data.image)
            o_h, o_w = get_image_size(output.image)
            scale_factor = (o_h / h, o_w / w)
        if has_gt_bboxes(data):
            output.gt_bboxes = self.resize_boxes(data.gt_bboxes, scale_factor)
        if has_gt_ignores(data):
            output.gt_ignores = self.resize_boxes(data.gt_ignores, scale_factor)
        if has_gt_keyps(data):
            output.gt_keyps = self.resize_keyps(data.gt_keyps, scale_factor)
        if has_gt_masks(data):
            output.gt_masks = self.resize_masks(data.gt_masks, scale_factor)
        if has_gt_semantic_seg(data):
            output.gt_semantic_seg = self.resize_semantic_seg(data.gt_semantic_seg, scale_factor)
        output.scale_factor = scale_factor
        return output

    def get_scale_factor(self, img_h, img_w):
        """return scale_factor_h, scale_factor_w
        """
        raise NotImplementedError

    def resize_image(self, img, scale_factor):
        ratio_h, ratio_w = _pair(scale_factor)
        if is_pil_image(img):
            origin_w, origin_h = img.size
            target_h, target_w = int(round(ratio_h * origin_h)), int(round(ratio_w * origin_w))
            return TF.resize(img, (target_h, target_w))
        elif is_numpy_image(img):
            return cv2.resize(
                img, None, None,
                fx=ratio_w,
                fy=ratio_h,
                interpolation=cv2.INTER_LINEAR)
        else:
            raise TypeError('{} format not supported'.format(type(img)))

    def resize_boxes(self, boxes, scale_factor):
        ratio_h, ratio_w = _pair(scale_factor)
        if ratio_h == ratio_w:
            boxes[:, :4] *= ratio_h
        else:
            boxes[:, 0] *= ratio_w
            boxes[:, 1] *= ratio_h
            boxes[:, 2] *= ratio_w
            boxes[:, 3] *= ratio_h
        return boxes

    def resize_masks(self, masks, scale_factor):
        """
        Actually resize polygons

        Note:
            Since resizing polygons will cause a shift for masks generation
            here add an offset to eliminate the difference between resizing polygons and resizing masks
        """
        ratio_h, ratio_w = _pair(scale_factor)
        for polys in masks:
            for poly in polys:
                poly[0::2] *= ratio_w
                poly[1::2] *= ratio_h
        return masks

    def resize_keyps(self, keyps, scale_factor):
        ratio_h, ratio_w = _pair(scale_factor)
        N, K = keyps.size()[:2]
        keyps = keyps.view(-1, 3)
        labeled = torch.nonzero(keyps[:, 2] > 0)
        if labeled.size(0) == 0:
            return keyps.view(N, K, 3)

        labeled = labeled.view(1, -1)[0]
        keyps[labeled, 0] *= ratio_w
        keyps[labeled, 1] *= ratio_h
        keyps = keyps.view(N, K, 3)
        return keyps

    def resize_semantic_seg(self, semantic_seg, scale_factor):
        ratio_h, ratio_w = _pair(scale_factor)
        return cv2.resize(
            semantic_seg, None, None,
            fx=ratio_w,
            fy=ratio_h,
            interpolation=cv2.INTER_NEAREST)


@AUGMENTATION_REGISTRY.register('keep_ar_resize')
class KeepAspectRatioResize(Resize):

    def get_scale_factor(self, img_h, img_w):
        """return scale_factor_h, scale_factor_w
        """
        short = min(img_w, img_h)
        large = max(img_w, img_h)
        scale = np.random.choice(self.scales)
        if scale <= 0:
            scale_factor = 1.0
        else:
            scale_factor = min(scale / short, self.max_size / large)
        return scale_factor, scale_factor


@AUGMENTATION_REGISTRY.register('keep_ar_resize_max')
class KeepAspectRatioResizeMax(Resize):
    def __init__(self,
                 max_size=sys.maxsize,
                 separate_wh=False,
                 min_size=1,
                 padding_type=None,
                 padding_val=0,
                 random_size=[],
                 scale_step=32):
        """
        Args:
            scales: evaluable str, e.g. range(500,600,1) or given list
            max_size: int
        """
        super(KeepAspectRatioResizeMax,
              self).__init__([100], max_size, separate_wh)
        self.min_size = min_size
        self.padding_type = padding_type
        self.padding_val = padding_val
        self.random_size = random_size
        self.scale_step = scale_step

    def get_scale_factor(self, img_h, img_w):
        """return scale_factor_h, scale_factor_w
        """
        large = max(img_w, img_h)
        if len(self.random_size) == 0:
            self.cur_size = self.max_size
        else:
            self.cur_size = random.randint(*self.random_size) * self.scale_step
        scale_factor = float(self.cur_size) / large
        return scale_factor, scale_factor

    def padding(self, data):
        image = data['image']
        height, width = image.shape[:2]
        if height == self.cur_size and width == self.cur_size:
            return data
        if image.ndim == 3:
            padding_img = np.full((self.cur_size, self.cur_size, 3),
                                  self.padding_val,
                                  dtype=image.dtype)
        else:
            padding_img = np.full((self.cur_size, self.cur_size),
                                  self.padding_val,
                                  dtype=image.dtype)
        if self.padding_type == 'left_top':
            padding_img[:height, :width] = image
        else:
            raise NotImplementedError
        data['image'] = padding_img
        return data

    def augment(self, data, scale_factor=None):
        if scale_factor is None:
            # img_h, img_w = data.image.shape[:2]
            img_h, img_w = get_image_size(data.image)
            scale_factor = self.get_scale_factor(img_h, img_w)
        output = copy.copy(data)
        output.image = self.resize_image(data.image, scale_factor)
        if self.padding_type is not None:
            output = self.padding(output)
        if self.separate_wh:
            h, w = get_image_size(data.image)
            o_h, o_w = get_image_size(output.image)
            scale_factor = (o_h / h, o_w / w)
        if has_gt_bboxes(data):
            # filter
            f_bboxes = self.resize_boxes(data.gt_bboxes, scale_factor)
            w_mask_b = (f_bboxes[:, 2] - f_bboxes[:, 0]) > self.min_size
            h_mask_b = (f_bboxes[:, 3] - f_bboxes[:, 1]) > self.min_size
            mask_b = w_mask_b & h_mask_b
            f_bboxes = f_bboxes[mask_b]
            output.gt_bboxes = f_bboxes
        if has_gt_ignores(data):
            output.gt_ignores = self.resize_boxes(data.gt_ignores,
                                                  scale_factor)
        if has_gt_keyps(data):
            output.gt_keyps = self.resize_keyps(data.gt_keyps, scale_factor)
        if has_gt_masks(data):
            output.gt_masks = self.resize_masks(data.gt_masks, scale_factor)
        if has_gt_semantic_seg(data):
            output.gt_semantic_seg = self.resize_semantic_seg(
                data.gt_semantic_seg, scale_factor)
        output.scale_factor = scale_factor
        return output


@BATCHING_REGISTRY.register('batch_pad')
class BatchPad(object):
    def __init__(self, alignment=1, pad_value=0):
        self.alignment = alignment
        self.pad_value = pad_value

    def __call__(self, data):
        """
        Args:
            images: list of tensor
        """
        images = data['image']
        max_img_h = max([_.size(-2) for _ in images])
        max_img_w = max([_.size(-1) for _ in images])
        target_h = int(np.ceil(max_img_h / self.alignment) * self.alignment)
        target_w = int(np.ceil(max_img_w / self.alignment) * self.alignment)
        padded_images = []
        for image in images:
            assert image.dim() == 3
            src_h, src_w = image.size()[-2:]
            pad_size = (0, target_w - src_w, 0, target_h - src_h)
            padded_images.append(F.pad(image, pad_size, 'constant', self.pad_value).data)
        data['image'] = torch.stack(padded_images)
        return data


class ImageExpand(Augmentation):
    """ expand image with means
    """

    def __init__(self, means=127.5, expand_ratios=2., expand_prob=0.5):
        self.means = means
        self.expand_ratios = expand_ratios
        self.expand_prob = expand_prob

    def augment(self, data):
        assert not has_gt_keyps(data), "key points is not supported !!!"
        assert not has_gt_masks(data), "masks is not supported !!!"
        if not coin_tossing(self.expand_prob):
            return data
        output = copy.copy(data)
        image = output.image
        new_gts = tensor2numpy(output)
        if check_fake_gt(new_gts):
            return data
        height, width = image.shape[:2]

        scale = random.uniform(1, self.expand_ratios)
        w = int(scale * width)
        h = int(scale * height)

        left = random.randint(0, w - width)
        top = random.randint(0, h - height)

        boxes_t = new_gts.copy()
        boxes_t[:, :2] += (left, top)
        boxes_t[:, 2:4] += (left, top)
        if len(image.shape) == 3:
            expand_image = np.empty((h, w, 3), dtype=image.dtype)
        else:
            expand_image = np.empty((h, w), dtype=image.dtype)
        expand_image[:, :] = self.means
        expand_image[top:top + height, left:left + width] = image
        gt_bboxes, ig_bboxes = numpy2tensor(boxes_t)
        output.gt_bboxes = gt_bboxes
        output.gt_ignores = ig_bboxes
        output.image = expand_image
        return output


@AUGMENTATION_REGISTRY.register('fix_output_resize')
class FixOutputResize(Resize):
    """ fixed output resize
    """

    def get_scale_factor(self, img_h, img_w):
        out_h, out_w = self.scales[0], self.scales[1]
        return out_h / img_h, out_w / img_w


@AUGMENTATION_REGISTRY.register('color_jitter')
class RandomColorJitter(Augmentation):
    """
    Randomly change the brightness, contrast and saturation of an image.

    Arguments:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
        brightness_delta (int or tuple of int (min, max)): How much to jitter brightness by delta value.
            brightness_delta is chosen uniformly from [max(0, 1 - brightness_delta),
            min(0, 1 + brightness_delta)].
        hue_delta (int or tuple of int (min, max)): How much to jitter hue by delta value.
            hue_delta is chosen uniformly from [max(0, 1 - hue_delta),
            min(0, 1 + hue_delta)].
        channel_shuffle (bool): Whether to randomly shuffle the image channels or not.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, brightness_delta=0,
                 hue_delta=0, channel_shuffle=False, prob=0):
        super(RandomColorJitter, self).__init__()
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5), clip_first_on_zero=False)
        self.brightness_delta = self._check_input(brightness_delta, 'brightness_delta', center=0)
        self.hue_delta = self._check_input(hue_delta, 'hue_delta', center=0)
        self.channel_shuffle = channel_shuffle
        self.prob = prob

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    def get_params(self, brightness, contrast, saturation, hue, brightness_delta, hue_delta, channel_shuffle):
        """
        Get a randomized transform to be applied on image.

        Arguments are same as that of __init__.

        rETurns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        img_transforms = []

        if brightness is not None and random.random() < self.prob:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            img_transforms.append(transforms.Lambda(lambda img: adjust_brightness(img, brightness_factor)))

        if contrast is not None and random.random() < self.prob:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            img_transforms.append(transforms.Lambda(lambda img: adjust_contrast(img, contrast_factor)))

        if saturation is not None and random.random() < self.prob:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            img_transforms.append(transforms.Lambda(lambda img: adjust_saturation(img, saturation_factor)))

        if hue is not None and random.random() < self.prob:
            hue_factor = random.uniform(hue[0], hue[1])
            img_transforms.append(transforms.Lambda(lambda img: adjust_hue(img, hue_factor)))

        if brightness_delta is not None and random.random() < self.prob:
            brightness_delta = random.uniform(brightness_delta[0], brightness_delta[1])
            img_transforms.append(transforms.Lambda(lambda img: img + brightness_delta))

        if hue_delta is not None and random.random() < self.prob:
            hue_delta = random.uniform(hue_delta[0], hue_delta[1])

            def augment_hue_delta(img):
                img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                img[..., 0] += hue_delta
                img[..., 0][img[..., 0] > 360] -= 360
                img[..., 0][img[..., 0] < 0] += 360
                img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
                return img
            img_transforms.append(augment_hue_delta)

        if channel_shuffle and random.random() < self.prob:
            img_transforms.append(transforms.Lambda(lambda img: img[:, np.random.permutation(3)]))

        random.shuffle(img_transforms)
        img_transforms = transforms.Compose(img_transforms)

        return img_transforms

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
        img = Image.fromarray(np.uint8(img))
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue,
                                    self.brightness_delta, self.hue_delta,
                                    self.channel_shuffle)
        img = transform(img)
        img = np.asanyarray(img)
        output.image = img
        return output

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string


@AUGMENTATION_REGISTRY.register('crop')
class ImageCrop(Augmentation):
    """ random crop image base gt and crop region (iof)
    """

    def __init__(self, means=127.5, scale=600, crop_prob=0.5):
        self.means = means
        self.img_scale = scale
        self.crop_prob = crop_prob

    def _nobbox_crop(self, width, height, image):
        h = w = min(width, height)
        if width == w:
            left = 0
        else:
            left = random.randrange(width - w)
        if height == h:
            t = 0
        else:
            t = random.randrange(height - h)
        roi = np.array((left, t, left + w, t + h))
        image_t = image[roi[1]:roi[3], roi[0]:roi[2]]
        return image_t

    def _random_crop(self, width, height, image, new_gts):
        for _ in range(100):
            if not coin_tossing(self.crop_prob):
                scale = 1
            else:
                scale = random.uniform(0.5, 1.)
            short_side = min(width, height)
            w = int(scale * short_side)
            h = w

            if width == w:
                left = 0
            else:
                left = random.randrange(width - w)
            if height == h:
                t = 0
            else:
                t = random.randrange(height - h)
            roi = np.array((left, t, left + w, t + h))

            value = np_bbox_iof_overlaps(new_gts, roi[np.newaxis])
            flag = (value >= 1)
            if not flag.any():
                continue

            centers = (new_gts[:, :2] + new_gts[:, 2:4]) / 2
            mask = np.logical_and(roi[:2] < centers, centers < roi[2:]).all(axis=1)
            boxes_t = new_gts[mask].copy()

            b_w_t = (boxes_t[:, 2] - boxes_t[:, 0] + ALIGNED_FLAG.offset) / w * self.img_scale
            b_h_t = (boxes_t[:, 3] - boxes_t[:, 1] + ALIGNED_FLAG.offset) / h * self.img_scale
            mask_b = np.minimum(b_w_t, b_h_t) >= 6.0

            if boxes_t.shape[0] == 0 or mask_b.shape[0] == 0:
                continue

            image_t = image[roi[1]:roi[3], roi[0]:roi[2]]

            boxes_t[:, :2] = np.maximum(boxes_t[:, :2], roi[:2])
            boxes_t[:, :2] -= roi[:2]
            boxes_t[:, 2:4] = np.minimum(boxes_t[:, 2:4], roi[2:])
            boxes_t[:, 2:4] -= roi[:2]
            return image_t, boxes_t
        long_side = max(width, height)
        if len(image.shape) == 3:
            image_t = np.empty((long_side, long_side, 3), dtype=image.dtype)
        else:
            image_t = np.empty((long_side, long_side), dtype=image.dtype)
        image_t[:, :] = self.means
        image_t[0:0 + height, 0:0 + width] = image
        return image_t, new_gts

    def augment(self, data):
        assert not has_gt_keyps(data), "key points is not supported !!!"
        assert not has_gt_masks(data), "masks is not supported !!!"

        output = copy.copy(data)
        image = output.image
        new_gts = tensor2numpy(output)
        height, width = image.shape[:2]
        if check_fake_gt(new_gts):
            output.image = self._nobbox_crop(width, height, image)
            return output
        crop_image, boxes_t = self._random_crop(width, height, image, new_gts)
        gt_bboxes, ig_bboxes = numpy2tensor(boxes_t)
        output.gt_bboxes = gt_bboxes
        output.gt_ignores = ig_bboxes
        output.image = crop_image
        return output


@AUGMENTATION_REGISTRY.register('stitch_expand')
class ImageStitchExpand(Augmentation):
    """ expand image with 4 original image
    """

    def __init__(self, expand_ratios=2., expand_prob=0.5):
        self.expand_ratios = expand_ratios
        self.expand_prob = expand_prob

    def augment(self, data):
        assert not has_gt_keyps(data), "key points is not supported !!!"
        assert not has_gt_masks(data), "masks is not supported !!!"

        if not coin_tossing(self.expand_prob):
            return data
        scale = random.uniform(1, self.expand_ratios)
        output = copy.copy(data)
        image = output.image
        new_gts = tensor2numpy(output)
        if check_fake_gt(new_gts) or (scale - 1) <= 1e-2:
            return data
        im_h, im_w = image.shape[:2]
        w = int(scale * im_w)
        h = int(scale * im_h)

        new_gts_tmps = []
        for j in range(2):
            for i in range(2):
                new_gts_tmp = new_gts.copy()
                new_gts_tmp[:, [0, 2]] += im_w * i
                new_gts_tmp[:, [1, 3]] += im_h * j
                new_gts_tmps.append(new_gts_tmp)
        new_gts_cat = np.concatenate(new_gts_tmps)
        height, width, = 2 * im_h, 2 * im_w
        for _ in range(10):
            ll = 0 if width == w else random.randrange(width - w)
            t = 0 if height == h else random.randrange(height - h)
            roi = np.array((ll, t, ll + w, t + h))
            value = np_bbox_iof_overlaps(new_gts_cat, roi[np.newaxis])
            flag = (value >= 1)
            if not flag.any():
                continue

            centers = (new_gts_cat[:, :2] + new_gts_cat[:, 2:4]) / 2
            mask = np.logical_and(roi[:2] < centers, centers < roi[2:]).all(axis=1)
            boxes_t = new_gts_cat[mask].copy()

            boxes_t[:, :2] = np.maximum(boxes_t[:, :2], roi[:2])
            boxes_t[:, :2] -= roi[:2]
            boxes_t[:, 2:4] = np.minimum(boxes_t[:, 2:4], roi[2:])
            boxes_t[:, 2:4] -= roi[:2]

            image00 = image[roi[1]:im_h, roi[0]:im_w]
            image01 = image[roi[1]:im_h, 0:roi[2] - im_w]
            image10 = image[0:roi[3] - im_h, roi[0]:im_w]
            image11 = image[0:roi[3] - im_h, 0:roi[2] - im_w]

            image_t = np.concatenate((np.concatenate((image00, image01), axis=1),
                                      np.concatenate((image10, image11), axis=1)))  # noqa
            gt_bboxes, ig_bboxes = numpy2tensor(boxes_t)
            output.gt_bboxes = gt_bboxes
            output.gt_ignores = ig_bboxes
            output.image = image_t
            return output
        return output
