from __future__ import division

# Standard Library
import collections
import sys
import copy

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


class ImageMeta(object):
    """A class that provide a simple interface to encode and decode dict to save memory
    """
    @classmethod
    def encode(self, dict_obj):
        """
        Note:
            As image_height and image_width have no use any more, abandon them to save more memory.

        Arguments:
            dict_obj: single original data read from json

        Returns:
            list_data: [filename, [instances]]
        """
        meta_info = copy.deepcopy(dict_obj)

        if 'instances' in meta_info:
            meta_info.pop('instances')

        instances = []
        for item in dict_obj.get('instances', []):
            polygons = [np.array(poly, dtype=np.float32) for poly in item.get('mask', [])]
            instance = item['bbox'] + [item.get('label', None), item.get('is_ignored', False), polygons]
            instances.append(instance)
        bucket = dict_obj.get('bucket', None)
        neg_target = dict_obj.get('neg_target', 0)
        image_source = dict_obj.get('image_source', 0)
        key = dict_obj.get('key', dict_obj.get('keys', None))
        list_obj = [
            dict_obj['filename'], instances, (bucket, key, neg_target, image_source, meta_info)
        ]
        return list_obj

    @classmethod
    def decode(self, list_obj):
        filename, instances, buckey = list_obj
        decoded_instances = []
        for ins in instances:
            ins = {
                'bbox': ins[:4],
                'label': ins[4],
                'is_ignored': ins[5],
                'mask': ins[6]
            }
            decoded_instances.append(ins)
        dict_obj = {
            'filename': filename,
            'bucket': buckey[0],
            'key': buckey[1],
            'instances': decoded_instances,
            'neg_target': buckey[2],
            'image_source': buckey[3],
            'meta_info': buckey[4]
        }
        return dict_obj
