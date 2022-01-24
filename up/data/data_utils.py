from __future__ import division

# Standard Library
import collections
import sys

# Import from third library
import numpy as np
import torch
from PIL import Image
try:
    import accimage
except ImportError:
    accimage = None


if sys.version_info < (3, 3):
    Sequence = collections.Sequence
    Iterable = collections.Iterable
else:
    Sequence = collections.abc.Sequence
    Iterable = collections.abc.Iterable


def is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


def is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3


def _is_numpy(img):
    return isinstance(img, np.ndarray)


def is_numpy_image(img):
    return _is_numpy(img) and img.ndim in {2, 3}


def get_image_size(img):
    """ return image size in (h, w) format
    """

    if is_pil_image(img):
        w, h = img.size
    elif is_tensor_image(img):
        h, w = img.shape[-2:]
    elif is_numpy_image(img):
        h, w = img.shape[:2]
    else:
        raise TypeError("Unexpected type {}".format(type(img)))
    return h, w


def coin_tossing(prob):
    """
    """
    return torch.rand(1)[0].item() < prob
