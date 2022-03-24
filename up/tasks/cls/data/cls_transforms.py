import torch
import math
import random
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from up.data.datasets.transforms import Augmentation
from up.utils.general.registry_factory import AUGMENTATION_REGISTRY, BATCHING_REGISTRY
from up.tasks.cls.data.rand_augmentation import *
from up.tasks.cls.data.auto_augmentation import *
from up.tasks.cls.data.data_utils import *
import copy

try:
    from torchvision.transforms.functional import InterpolationMode
    _torch_interpolation_to_str = {InterpolationMode.NEAREST: 'nearest',
                                   InterpolationMode.BILINEAR: 'bilinear',
                                   InterpolationMode.BICUBIC: 'bicubic',
                                   InterpolationMode.BOX: 'box',
                                   InterpolationMode.HAMMING: 'hamming',
                                   InterpolationMode.LANCZOS: 'lanczos'}
except: # noqa
    _torch_interpolation_to_str = {0: 'nearest',
                                   1: 'lanczos',
                                   2: 'bilinear',
                                   3: 'bicubic',
                                   4: 'box',
                                   5: 'hamming'}
_str_to_torch_interpolation = {b: a for a, b in _torch_interpolation_to_str.items()}


__all__ = [
    'TorchAugmentation',
    'RandomResizedCrop',
    'RandomHorizontalFlip',
    'PILColorJitter',
    'TorchResize',
    'TorchCenterCrop',
    'RandomErasing',
    'RandAugment',
    'RandAugIncre',
    'AutoAugment']


class TorchAugmentation(Augmentation):
    def __init__(self):
        self.op = lambda x: x

    def augment(self, data):
        output = copy.copy(data)
        output.image = self.op(data.image)
        return output


@AUGMENTATION_REGISTRY.register('torch_random_resized_crop')
class RandomResizedCrop(TorchAugmentation):
    def __init__(self, size, **kwargs):
        if 'interpolation' in kwargs:
            kwargs['interpolation'] = _str_to_torch_interpolation[kwargs['interpolation']]
        self.op = transforms.RandomResizedCrop(size, **kwargs)


@AUGMENTATION_REGISTRY.register('torch_random_horizontal_flip')
class RandomHorizontalFlip(TorchAugmentation):
    def __init__(self, p=0.5):
        self.p = p

    def augment(self, data):
        output = copy.copy(data)
        if torch.rand(1) < self.p:
            output.image = TF.hflip(data.image)
        return output


@AUGMENTATION_REGISTRY.register('torch_color_jitter')
class PILColorJitter(TorchAugmentation):
    def __init__(self, brightness, contrast, saturation, hue=0.):
        self.op = transforms.ColorJitter(brightness, contrast, saturation, hue)


@AUGMENTATION_REGISTRY.register('torch_resize')
class TorchResize(TorchAugmentation):
    def __init__(self, size, **kwargs):
        if 'interpolation' in kwargs:
            kwargs['interpolation'] = _str_to_torch_interpolation[kwargs['interpolation']]
        self.op = transforms.Resize(size, **kwargs)


@AUGMENTATION_REGISTRY.register('torch_center_crop')
class TorchCenterCrop(TorchAugmentation):
    def __init__(self, size, **kwargs):
        self.op = transforms.CenterCrop(size, **kwargs)


@AUGMENTATION_REGISTRY.register('torch_auto_augmentation')
class AutoAugment(TorchAugmentation):
    def __init__(self, size, **kwargs):
        self.op = ImageNetPolicy()


@AUGMENTATION_REGISTRY.register('torch_random_augmentation')
class RandAugment(TorchAugmentation):
    def __init__(self, n, m, magnitude_std=0.0):
        self.n = n
        self.m = m
        self.augment_list = augment_list()
        self.mstd = magnitude_std

    def augment(self, data):
        output = copy.copy(data)
        ops = random.choices(self.augment_list, k=self.n)
        for op, minval, maxval in ops:
            if random.random() > 0.5:
                continue
            magnitude = random.gauss(self.m, self.mstd)
            magnitude = max(0, min(magnitude, 10))
            val = (float(magnitude) / 10) * float(maxval - minval) + minval
            output.image = op(output.image, val)

        return output


@AUGMENTATION_REGISTRY.register('torch_random_augmentationIncre')
class RandAugIncre(TorchAugmentation):
    def __init__(self, n, m, magnitude_std=0.0):
        self.n = n
        self.m = m
        self.augment_list = augment_increasing()
        self.mstd = magnitude_std

    def augment(self, data):
        output = copy.copy(data)
        ops = random.choices(self.augment_list, k=self.n)
        for op, minval, maxval in ops:
            if random.random() > 0.5:
                continue
            magnitude = random.gauss(self.m, self.mstd)
            magnitude = max(0, min(magnitude, 10))
            val = (float(magnitude) / 10) * float(maxval - minval) + minval
            output.image = op(output.image, val)

        return output


@AUGMENTATION_REGISTRY.register('torch_randerase')
class RandomErasing(TorchAugmentation):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
        This variant of RandomErasing is intended to be applied to either a batch
        or single image tensor after it has been normalized by dataset mean and std.
    Args:
         probability: Probability that the Random Erasing operation will be performed.
         min_area: Minimum percentage of erased area wrt input image area.
         max_area: Maximum percentage of erased area wrt input image area.
         min_aspect: Minimum aspect ratio of erased area.
         mode: pixel color mode, one of 'const', 'rand', or 'pixel'
            'const' - erase block is constant color of 0 for all channels
            'rand'  - erase block is same per-channel random (normal) color
            'pixel' - erase block is per-pixel random (normal) color
        max_count: maximum number of erasing blocks per image, area per box is scaled by count.
            per-image count is randomly chosen between 1 and this value.
    """

    def __init__(
            self,
            probability=0.5, min_area=0.02, max_area=1 / 3, min_aspect=0.3, max_aspect=None,
            mode='pixel', min_count=1, max_count=None, num_splits=0, device='cpu'):
        self.probability = probability
        self.min_area = min_area
        self.max_area = max_area
        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))
        self.min_count = min_count
        self.max_count = max_count or min_count
        self.num_splits = num_splits
        self.mode = mode.lower()
        self.rand_color = False
        self.per_pixel = False
        if self.mode == 'rand':
            self.rand_color = True  # per block random normal
        elif self.mode == 'pixel':
            self.per_pixel = True  # per pixel random normal
        else:
            assert not self.mode or self.mode == 'const'
        self.device = device

    def _erase(self, img, chan, img_h, img_w, dtype):
        if random.random() > self.probability:
            return
        area = img_h * img_w
        count = self.min_count if self.min_count == self.max_count else \
            random.randint(self.min_count, self.max_count)
        for _ in range(count):
            for attempt in range(10):
                target_area = random.uniform(self.min_area, self.max_area) * area / count
                aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
                if w < img_w and h < img_h:
                    top = random.randint(0, img_h - h)
                    left = random.randint(0, img_w - w)
                    img[:, top:top + h, left:left + w] = self._get_pixels(
                        self.per_pixel, self.rand_color, (chan, h, w),
                        dtype=dtype, device=self.device)
                    break

    def augment(self, data):
        image = copy.copy(data).image
        self._erase(image, *image.size(), image.dtype)
        data.image = image
        return data

    def _get_pixels(self, per_pixel, rand_color, patch_size, dtype=torch.float32, device='cuda'):
        if per_pixel:
            return torch.empty(patch_size, dtype=dtype, device=device).normal_()
        elif rand_color:
            return torch.empty((patch_size[0], 1, 1), dtype=dtype, device=device).normal_()
        else:
            return torch.zeros((patch_size[0], 1, 1), dtype=dtype, device=device)

    def __repr__(self):
        fs = self.__class__.__name__ + f'(p={self.probability}, mode={self.mode}'
        fs += f', count=({self.min_count}, {self.max_count}))'
        return fs


@BATCHING_REGISTRY.register('batch_mixup')
class BatchMixup(object):
    def __init__(self, alpha=1.0, num_classes=1000):
        self.alpha = alpha
        self.num_classes = num_classes

    def __call__(self, data):
        """
        Args:
            images: list of tensor
        """
        return mixup(data, self.alpha, self.num_classes)


@BATCHING_REGISTRY.register('batch_cutmix')
class BatchCutMix(object):
    def __init__(self, alpha=1.0, num_classes=1000):
        self.alpha = alpha
        self.num_classes = num_classes

    def __call__(self, data):
        """
        Args:
            images: list of tensor
        """
        return cutmix(data, self.alpha, self.num_classes)


@BATCHING_REGISTRY.register('batch_cutmixup')
class BatchCutMixup(object):
    def __init__(self, mixup_alpha=1.0, cutmix_alpha=1.0, switch_prob=0.5, num_classes=1000):
        self.switch_prob = switch_prob
        self.num_classes = num_classes
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha

    def __call__(self, data):
        use_cutmix = np.random.rand() < self.switch_prob
        if use_cutmix:
            return cutmix(data, self.cutmix_alpha, self.num_classes)
        return mixup(data, self.mixup_alpha, self.num_classes)
