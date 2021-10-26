from __future__ import division

# Import from third library
import os.path as op
import numpy as np
from easydict import EasyDict
import torch
from torch.utils.data import Dataset

# Import from local
from ..image_reader import build_image_reader
from ..metrics.base_evaluator import build_evaluator
from .transforms import build_partially_inverse_transformer, build_transformer


__all__ = ['BaseDataset']


class BaseDataset(Dataset):
    """
    A dataset should implement interface below:

    1. :meth:`__len__` to get size of the dataset, Required
    2. :meth:`__getitem__` to get a single data, Required
    3. evaluate to get metrics, Required
    4. dump to output results, Required
    """

    def __init__(self,
                 meta_file,
                 image_reader,
                 transformer,
                 evaluator=None,
                 class_names=None):
        super(BaseDataset, self).__init__()
        self.meta_file = meta_file
        self.image_reader = build_image_reader(image_reader)
        self.classes = class_names
        if transformer is not None:
            # add datset handler in transform kwargs in need of mosaic/mixup etc.
            for trans in transformer:
                if 'kwargs' in trans and trans['kwargs'].get('extra_input', False):
                    trans['kwargs']['dataset'] = self
            self.transformer = build_transformer(transformer)
            self.visable_transformer = build_partially_inverse_transformer(self.transformer)
        else:
            self.visable_transformer = self.transformer = lambda x: x

        if evaluator is not None:
            self.evaluator = build_evaluator(evaluator)
        else:
            self.evaluator = None

    def get_image_classes(self, img_index):
        """For Repeat Factor Sampler
        """
        raise NotImplementedError

    @property
    def num_classes(self):
        if hasattr(self, '_num_classes') and self._num_classes is not None:
            return self._num_classes
        assert self.classes is not None
        self._num_classes = len(self.classes)
        return self._num_classes

    @num_classes.setter
    def num_classes(self, num_classes):
        self._num_classes = num_classes

    @property
    def num_classes_without_background(self):
        return self.num_classes - 1

    @property
    def class_names(self):
        return self.classes

    @property
    def images_per_class(self):
        """Return a dict with keys of class and values of list of image indices (this idx is called by __iter__)
        """
        raise NotImplementedError

    @property
    def num_images_per_class(self):
        """Return a dict with keys of class and values of number of images of this class
        For Class Aware Balanced Sampler and Repeat Factor Sampler
        """
        raise NotImplementedError

    @property
    def num_instances_per_class(self):
        """Return a dict with keys of class and values of number of instances of this class
        """
        raise NotImplementedError

    def __len__(self):
        """
        Returns dataset length
        """
        raise NotImplementedError

    def __getitem__(self, idx):
        """
        Get a single image data: from dataset

        Arguments:
            - idx (:obj:`int`): index of image, 0 <= idx < len(self)

        """
        raise NotImplementedError

    def _fake_zero_data(self, *size):
        return torch.zeros(size)

    def dump(self, output):
        raise NotImplementedError

    def evaluate(self, res_file, res=None):
        """
        Arguments:
            - res_file (:obj:`str`): filename
        """
        metrics = self.evaluator.eval(res_file, res) if self.evaluator else {}
        return metrics

    def tensor2numpy(self, x):
        if x is None:
            return x
        if torch.is_tensor(x):
            return x.cpu().numpy()
        if isinstance(x, list):
            x = [_.cpu().numpy() if torch.is_tensor(_) else _ for _ in x]
        return x

    def get_np_images(self, images):
        np_images = []
        images = images.cpu()
        for image in images:
            data = EasyDict({'image': image})
            data = self.visable_transformer(data)
            image = data.image.permute(1, 2, 0).contiguous().cpu().float().numpy()
            if self.image_reader.color_mode != 'RGB':
                import cv2
                cvt_color_vis = getattr(cv2, 'COLOR_{}2RGB'.format(self.image_reader.color_mode))
                image = cv2.cvtColor(image, cvt_color_vis)
            np_images.append(image)
        np_images = np.stack(np_images, axis=0)
        return np_images.astype(np.uint8)

    def vis_gt(self, visualizer, input):
        """
        Arguments:
            - visualize (:obj:`BaseVisualizer`):
            - input (:obj:`dict`): output of model

        """
        gt_images = self.get_np_images(input['image'])
        filenames = input['filenames']
        image_info = input['image_info']
        gt_bboxes = self.tensor2numpy(input.get('gt_bboxes', None))
        gt_ignores = self.tensor2numpy(input.get('gt_ignores', None))

        batch_size = len(filenames)
        for b_ix in range(batch_size):
            image_name = op.splitext(op.basename(filenames[b_ix]))[0]
            image = gt_images[b_ix]
            unpad_image_h, unpad_image_w = image_info[b_ix][:2]
            image = image[:unpad_image_h, :unpad_image_w]
            bboxes = classes = ignores = None
            if gt_bboxes is not None:
                bboxes = gt_bboxes[b_ix]
                classes = bboxes[:, 4].astype(np.int32)
                scores = np.ones((bboxes.shape[0], 1))
                bboxes = np.hstack([bboxes[:, :4], scores])
            if gt_ignores is not None:
                ignores = gt_ignores[b_ix]

            visualizer.vis(image, bboxes, classes, image_name, ig_boxes=ignores)

    def vis_dt(self, visualizer, input):
        """
        Arguments:
            - visualizer (:obj:`visualizer`):
            - input (:obj:`dict`): output of model

        """
        dt_images = self.get_np_images(input['image'])
        filenames = input['filenames']
        image_info = input['image_info']
        dt_bboxes = self.tensor2numpy(input.get('dt_bboxes', None))

        if dt_bboxes is None:
            return

        batch_size = len(filenames)
        for b_ix in range(batch_size):
            image_name = op.splitext(op.basename(filenames[b_ix]))[0]
            image = dt_images[b_ix]
            image_h, image_w, _ = image.shape
            unpad_image_h, unpad_image_w = image_info[b_ix][:2]
            image = image[:unpad_image_h, :unpad_image_w]
            keep_ix = np.where(dt_bboxes[:, 0] == b_ix)[0]
            bboxes = dt_bboxes[keep_ix]
            if bboxes.size == 0:
                continue
            classes = bboxes[:, 6].astype(np.int32)
            bboxes = bboxes[:, 1:6]

            visualizer.vis(image, bboxes, classes, image_name)
