# Standard Library
import sys
import copy
import numpy as np

# Import from third library

import torch

from up.utils.general.registry_factory import AUGMENTATION_REGISTRY
from up.data.datasets.transforms import Augmentation, has_gt_keyps, has_gt_masks, check_fake_gt
from up.tasks.det.data.datasets.det_transforms import tensor2numpy, numpy2tensor, KeepAspectRatioResize

__all__ = ["RangeCrop"]


@AUGMENTATION_REGISTRY.register('range_crop')
class RangeCrop(Augmentation):
    def __init__(self, crop_size, scales, max_size=sys.maxsize, crop_prob=0.5):
        super().__init__()
        assert len(crop_size) == 2
        self.crop_size = crop_size
        # can't get correct scale_factor if using resize again
        self.resize_tfm = KeepAspectRatioResize(scales=scales, max_size=max_size)
        self.crop_prob = crop_prob

    def get_crop_size(self, img_size):
        h, w = img_size
        assert self.crop_size[0] <= self.crop_size[1]
        cw = np.random.randint(min(w, self.crop_size[0]), min(w, self.crop_size[1]) + 1)
        ch = np.random.randint(min(h, self.crop_size[0]), min(h, self.crop_size[1]) + 1)
        cx = np.random.randint(0, w - cw + 1)
        cy = np.random.randint(0, h - ch + 1)

        return cx, cy, cw, ch

    def crop_image(self, img, size):
        return img[size[1] : size[1] + size[3], size[0] : size[0] + size[2]]

    def crop_gts(self, img, gts, size):
        l, t, r, b = size[0], size[1], size[0] + size[2], size[1] + size[3]
        inside = (gts[:, 0] < r) & (gts[:, 1] < b) & (gts[:, 2] > l) & (gts[:, 3] > t)
        gts = gts[inside]

        if gts.size == 0:
            ig_bboxes_t = torch.zeros((1, 4))
            gt_bboxes_t = torch.zeros((1, 5))
        else:
            gts[:, 0] = np.clip(gts[:, 0] - l, a_min=0, a_max=r - l)
            gts[:, 1] = np.clip(gts[:, 1] - t, a_min=0, a_max=b - t)
            gts[:, 2] = np.clip(gts[:, 2] - l, a_min=0, a_max=r - l)
            gts[:, 3] = np.clip(gts[:, 3] - t, a_min=0, a_max=b - t)

            gt_bboxes_t, ig_bboxes_t = numpy2tensor(gts)

        img = self.crop_image(img, size)
        return img, gt_bboxes_t, ig_bboxes_t

    def augment(self, data):
        assert not has_gt_keyps(data), "key points is not supported !!!"
        assert not has_gt_masks(data), "masks is not supported !!!"

        output = copy.copy(data)

        if np.random.rand() > self.crop_prob:
            return output

        output = self.resize_tfm.augment(output)

        image = output.image
        new_gts = tensor2numpy(output)
        img_size = image.shape[:2]
        crop_size = self.get_crop_size(img_size)

        if check_fake_gt(new_gts):
            output.image = self.crop_image(image, crop_size)
            return output

        img, gt_bboxes, ig_bboxes = self.crop_gts(image, new_gts, crop_size)
        output.image = img
        output.gt_bboxes = gt_bboxes
        output.gt_ignores = ig_bboxes
        return output
