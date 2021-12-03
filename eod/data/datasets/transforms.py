from __future__ import division
import torch
from abc import ABC, abstractmethod
from collections import defaultdict
import numpy as np
from torchvision.transforms import Compose, Normalize
from torchvision.transforms import functional as TF
from eod.utils.general.registry_factory import AUGMENTATION_REGISTRY
import copy
from ..data_utils import (
    is_numpy_image,
    is_pil_image,
    is_tensor_image,
)


__all__ = [
    'has_image',
    'has_gt_bboxes',
    'has_gt_ignores',
    'has_gt_keyps',
    'has_gt_masks',
    'has_gt_semantic_seg',
    'check_fake_gt',
    'Augmentation',
    'ImageNormalize',
    'ImageToTensorInverse',
    'CustomImageToTensor'
]


def has_image(data):
    return data.get('image', None) is not None


def has_gt_bboxes(data):
    return data.get('gt_bboxes', None) is not None


def has_gt_ignores(data):
    return data.get('gt_ignores', None) is not None


def has_gt_keyps(data):
    return data.get('gt_keyps', None) is not None


def has_gt_masks(data):
    return data.get('gt_masks', None) is not None


def has_gt_semantic_seg(data):
    return data.get('gt_semantic_seg', None) is not None


def check_fake_gt(gts):
    if gts is None:
        return False
    if gts.shape[0] == 0:
        return True
    if gts.shape[0] == 1 and gts[0].sum() <= 1:
        return True
    return False


class Augmentation(ABC):

    def __init__(self):
        super(Augmentation, self).__init__()

    def _sanity_check(self, data):
        """ check the data format
        """
        data_format = defaultdict(str)
        if has_image(data):
            if is_pil_image(data.image):
                image_format = 'pil'
            elif is_numpy_image(data.image):
                image_format = 'np'
            elif is_tensor_image(data.image):
                image_format = 'tensor'
            else:
                raise TypeError('{} format is not supported for data augmentation function'.format(type(data.image)))
            data_format['image'] = image_format
        if has_gt_bboxes(data):
            assert torch.is_tensor(data.gt_bboxes)
            data_format['gt_bboxes'] = 'tensor'
        if has_gt_ignores(data):
            assert torch.is_tensor(data.gt_ignores)
            data_format['gt_ignores'] = 'tensor'
        if has_gt_keyps(data):
            assert torch.is_tensor(data.gt_keyps)
            data_format['gt_keyps'] = 'tensor'
        if has_gt_masks(data):
            assert isinstance(data.gt_masks, list)
            data_format['gt_masks'] = 'list'
        if has_gt_semantic_seg(data):
            if isinstance(data.gt_semantic_seg, np.ndarray):
                data_format['gt_semantic_seg'] = 'numpy'
            elif torch.is_tensor(data.gt_semantic_seg):
                data_format['gt_semantic_seg'] = 'tensor'
            else:
                raise TypeError('{} format is not supported for gt_semantic_seg'.format(type(data.gt_semantic_seg)))

    def _allow_format_change(self):
        return False

    def __call__(self, data):
        input_format = self._sanity_check(data)
        augmented_data = self.augment(data)
        output_format = self._sanity_check(augmented_data)
        if not self._allow_format_change():
            assert input_format == output_format, '{} vs {}'.format(input_format, output_format)
        return augmented_data

    @abstractmethod
    def augment(self, data):
        raise NotImplementedError


@AUGMENTATION_REGISTRY.register('normalize')
class ImageNormalize(Augmentation):
    def __init__(self, mean, std, inplace=False):
        self.mean = np.array(mean)
        self.std = np.array(std)
        self.normalize = Normalize(self.mean, self.std, inplace)
        # super(ImageNormalize, self).__init__(self.mean, self.std, inplace)

    def augment(self, data):
        output = copy.copy(data)
        # we need to convert image to float since normalize.mean and std are float tenosrs
        output.image = self.normalize(data.image.float())
        return output


@AUGMENTATION_REGISTRY.register('to_tensor')
class ImageToTensor(Augmentation):
    """ Convert PIL and cv2 image to Tensor image, and convert semantic seg from numpy into tensor
    """

    def _allow_format_change(self):
        return True

    def augment(self, data):
        output = copy.copy(data)
        output.image = self.image_to_tensor(data.image)
        if has_gt_semantic_seg(data):
            output.gt_semantic_seg = self.semantic_seg_to_tensor(data.gt_semantic_seg)
        return output

    def image_to_tensor(self, image):
        """
        Args:
          image: PIL or cv2 image

        Return:
          tensor image

         1. convert image to Tensor from PIL and cv2
         2. convert shape to [C, H, W] from PIL and cv2 shape
        """
        return TF.to_tensor(image)

    def semantic_seg_to_tensor(self, semantic_seg):
        """
        Args:
          semantic_seg (np.ndarray of [1, height, width]): mask
        Returns:
          semantic_seg (Tensor of [1, 1, height, width])
        """
        return torch.as_tensor(semantic_seg).unsqueeze(0)


@AUGMENTATION_REGISTRY.register('normalize_inverse')
class ImageToTensorInverse(Augmentation):

    def augment(self, data):
        output = copy.copy(data)
        output.image = self.inverse_image(data.image)
        if has_gt_semantic_seg(data):
            output.gt_semantic_seg = self.inverse_semantic_seg(data.gt_semantic_seg)
        return output

    def inverse_image(self, image):
        return image * 255

    def inverse_semantic_seg(self, semantic_seg):
        return semantic_seg.squeeze(0).cpu().numpy()


@AUGMENTATION_REGISTRY.register('custom_to_tensor')
class CustomImageToTensor(Augmentation):
    def augment(self, data):
        image = data['image']
        if image.ndim == 3:
            image = image.transpose(2, 0, 1)
        else:
            image = image[None]
        data['image'] = torch.from_numpy(image).float()
        return data


def build_partially_inverse_transformer(compose_transformer):
    """ To transform image tensor to visiable image inversely"""
    inverse_transforms = []
    for transform in compose_transformer.transforms[::-1]:
        if isinstance(transform, ImageToTensor):
            inverse_transforms.append(ImageToTensorInverse())
        elif isinstance(transform, ImageNormalize):
            mean = transform.mean
            std = transform.std
            inverse_transforms.append(ImageNormalize(-mean / std, 1.0 / std))
    return Compose(inverse_transforms)


def build_transformer(cfgs):
    transform_list = [AUGMENTATION_REGISTRY.build(cfg) for cfg in cfgs]
    return Compose(transform_list)
