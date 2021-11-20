# Standard Library
import math
import copy
import random
import cv2
import torch
import numpy as np
import torch.nn.functional as F

from eod.data.datasets.transforms import Augmentation
from eod.utils.general.registry_factory import AUGMENTATION_REGISTRY, BATCHING_REGISTRY
from eod.utils.general.global_flag import ALIGNED_FLAG


__all__ = ['Mosaic']


def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.2):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment, box2 after augment,
    # wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + 1e-16) > area_thr) & (ar < ar_thr)  # candidates


@AUGMENTATION_REGISTRY.register('mosaic')
class Mosaic(Augmentation):
    """ stitch 4 images into one image, used for yolov5, little diff with mosaic
    """

    def __init__(self,
                 extra_input=True,
                 dataset=None,
                 mosaic_prob=1.0,
                 tar_size=640,
                 fill_color=0,
                 mosaic_self=False,
                 memcached=None,
                 clip_box=True):
        super(Mosaic, self).__init__()
        assert extra_input and dataset is not None
        self.dataset = dataset
        self.mosaic_prob = mosaic_prob
        self.tar_size = tar_size
        if isinstance(self.tar_size, int):
            self.tar_size = [tar_size, tar_size]  # (h, w)
        self.fill_color = fill_color
        self.mosaic_self = mosaic_self
        self.clip_box = clip_box
        self.memcached = memcached
        if self.memcached is not None:
            self.init_memcached()

    def init_memcached(self):
        from pymemcache.client.base import PooledClient
        host = self.memcached.get('host', '127.0.0.1')
        port = self.memcached.get('port', '11211')
        pool_size = self.memcached.get('pool_size', 4)
        self.client = PooledClient(f'{host}:{port}', max_pool_size=pool_size)
        self.encode_type = self.memcached.get("encode_type", None)
        self.image_shapes = {}

    def hash_filename(self, filename):
        import hashlib
        md5 = hashlib.md5()
        md5.update(filename.encode('utf-8'))
        hash_filename = md5.hexdigest()
        return hash_filename

    def clip_bboxes(self, data, pad, max_hw):
        padh, padw = pad
        if data.get('gt_bboxes', None) is not None:
            gt_bboxes = data['gt_bboxes']
            gt_bboxes[:, [0, 2]] += padw
            gt_bboxes[:, [1, 3]] += padh
            if self.clip_box:
                gt_bboxes[:, [0, 2]] = torch.clamp(gt_bboxes[:, [0, 2]], 0, max_hw[1])
                gt_bboxes[:, [1, 3]] = torch.clamp(gt_bboxes[:, [1, 3]], 0, max_hw[0])
            data['gt_bboxes'] = gt_bboxes
        if data.get('gt_ignores', None) is not None:
            ig_bboxes = data['gt_ignores']
            ig_bboxes[:, [0, 2]] += padw
            ig_bboxes[:, [1, 3]] += padh
            if self.clip_box:
                ig_bboxes[:, [0, 2]] = torch.clamp(ig_bboxes[:, [0, 2]], 0, max_hw[1])
                ig_bboxes[:, [1, 3]] = torch.clamp(ig_bboxes[:, [1, 3]], 0, max_hw[0])
            data['gt_ignores'] = ig_bboxes
        return data

    def sew(self, inputs, center, out_size):
        assert len(inputs) == 4, 'mosaic must have 4 images once a time'
        yc, xc = center
        channels = 3           # default is rgb
        if len(inputs[0].image.shape) == 2:
            channels = 1
        if not isinstance(self.fill_color, list):
            self.fill_color = [self.fill_color] * channels
        canvas = []
        for c in range(channels):
            canvas_c = np.full((out_size[0] * 2, out_size[1] * 2), self.fill_color[c], dtype=np.uint8)
            canvas.append(canvas_c)
        if channels == 1:
            canvas = canvas[0]
        else:
            canvas = np.stack(canvas, axis=-1)
        # only care about gt_bboxes and gt_ignores
        gt_bboxes = []
        ig_bboxes = []
        for idx, cur_input in enumerate(inputs):
            img = cur_input['image']
            h, w = img.shape[0], img.shape[1]
            if idx == 0:  # top left
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif idx == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, out_size[1] * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif idx == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(out_size[0] * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif idx == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, out_size[1] * 2), min(out_size[0] * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            canvas[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # cannvas[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b
            cur_input = self.clip_bboxes(cur_input, (padh, padw), [size * 2 for size in out_size])
            if cur_input.get('gt_bboxes', None) is not None:
                gt_bboxes.append(cur_input['gt_bboxes'])
            if cur_input.get('gt_ignores', None) is not None:
                ig_bboxes.append(cur_input['gt_ignores'])

        sewed_input = inputs[0]
        sewed_input['image'] = canvas
        if sewed_input.get('gt_bboxes', None) is not None:
            sewed_input['gt_bboxes'] = torch.cat(gt_bboxes)
        if sewed_input.get('gt_ignores', None) is not None:
            sewed_input['gt_ignores'] = torch.cat(ig_bboxes)
        return sewed_input

    def cache_resize_image(self, img, filename, tar_size):
        hash_filename = self.hash_filename(filename)
        if hash_filename not in self.image_shapes:
            re_img = self.resize_img(img, tar_size)
            self.image_shapes[hash_filename] = re_img.shape
            if self.encode_type is not None:
                re_img = cv2.imencode(f'.{self.encode_type}', re_img)[1]
            self.client.set(hash_filename, re_img.tostring(), noreply=False)
        else:
            re_img = self.client.get(hash_filename)
            re_img = np.frombuffer(re_img, dtype=np.uint8)
            if self.encode_type is not None:
                re_img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            else:
                re_img = re_img.reshape(self.image_shapes[hash_filename])
        return re_img

    def resize_img(self, img, tar_size):
        interp = cv2.INTER_LINEAR
        h0, w0 = img.shape[:2]  # orig hw
        r = min(1. * tar_size[0] / h0, 1. * tar_size[1] / w0)
        if r != 1:
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
        return img

    def load_img(self, ori_data, tar_size):
        data = copy.deepcopy(ori_data)
        img = data.image
        h0, w0 = img.shape[:2]
        if self.memcached is None:
            img = self.resize_img(img, tar_size)
        else:
            try:
                img = self.cache_resize_image(img, data['filename'], tar_size)
            except Exception as e: # noqa
                img = self.resize_img(img, tar_size)
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        h1, w1 = img.shape[:2]  # new hw
        data.image = img
        scale_h, scale_w = h1 / h0, w1 / w0
        if data.get('gt_bboxes', None) is not None:
            gt_bboxes = data['gt_bboxes']
            gt_bboxes[:, [0, 2]] *= scale_w
            gt_bboxes[:, [1, 3]] *= scale_h
            data['gt_bboxes'] = gt_bboxes
        if data.get('gt_ignores', None) is not None:
            ig_bboxes = data['gt_ignores']
            ig_bboxes[:, [0, 2]] *= scale_w
            ig_bboxes[:, [1, 3]] *= scale_h
            data['gt_ignores'] = ig_bboxes
        return data

    def get_idx(self):
        return random.randint(0, len(self.dataset) - 1)

    def augment(self, data):
        if not self.mosaic_self and random.random() >= self.mosaic_prob:
            return data
        origin_input = copy.copy(data)
        image_s = self.tar_size  # h, w
        mosaic_border = [-image_s[0] // 2, -image_s[1] // 2]
        yc, xc = [int(random.uniform(-x, 2 * image_s[idx] + x)) for idx, x in enumerate(mosaic_border)]  # noqa
        inputs = []

        # if prob less than mosaic_prob, the mosaic will paste 4 same ori_img
        for i in range(4):
            if i == 0 or self.mosaic_self:
                cur_input = origin_input
            else:
                idx = self.get_idx()
                cur_input = self.dataset.get_input(idx)
            if not (self.mosaic_self and i > 0):
                cur_input = self.load_img(cur_input, image_s)
            if self.mosaic_self and i > 0:
                cur_input = copy.deepcopy(cur_input)
            inputs.append(cur_input)

        output = self.sew(inputs, (yc, xc), image_s)
        output['mosaic'] = True
        return output


@AUGMENTATION_REGISTRY.register('random_perspective')
class RandomPespective(Augmentation):
    def __init__(self,
                 degrees=0.0,
                 translate=0.1,
                 scale=0.5,
                 shear=0.0,
                 perspective=0.0,
                 fill_color=0,
                 max_try=1,
                 border=(0, 0),
                 clip_box=True):
        super(RandomPespective, self).__init__()
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.fill_color = fill_color
        self.border = border
        self.max_try = max_try
        self.clip_box = clip_box

    def label_transform(self, ori_targets, M, width, height, scale):
        targets = copy.deepcopy(ori_targets)
        n = len(targets)
        if n:
            # warp points
            xy = np.ones((n * 4, 3))
            xy[:, :2] = targets[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ M.T  # noqa
            if self.perspective:
                xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
            else:  # affine
                xy = xy[:, :2].reshape(n, 8)
            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # clip boxes
            if self.clip_box:
                xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width - ALIGNED_FLAG.offset)
                xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height - ALIGNED_FLAG.offset)

            # filter candidates
            i = box_candidates(box1=targets[:, :4].cpu().numpy().T * scale, box2=xy.T)
            targets = targets[i]
            targets[:, :4] = torch.tensor(xy[i], device=targets.device)
        return targets

    def generate_trans_matrix(self, img, width, height):
        # Center
        C = np.eye(3)
        C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
        C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

        # Perspective
        P = np.eye(3)
        P[2, 0] = random.uniform(-self.perspective, self.perspective)  # x perspective (about y)
        P[2, 1] = random.uniform(-self.perspective, self.perspective)  # y perspective (about x)

        # Rotation and Scale
        R = np.eye(3)
        a = random.uniform(-self.degrees, self.degrees)
        # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
        s = random.uniform(1 - self.scale, 1 + self.scale)
        # s = 2 ** random.uniform(-scale, scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        # Shear
        S = np.eye(3)
        S[0, 1] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # y shear (deg)

        # Translation
        T = np.eye(3)
        T[0, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * width  # x translation (pixels)
        T[1, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * height  # y translation (pixels)

        # Combined rotation matrix
        M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
        return M, s

    def augment(self, data):
        if 'mosaic' not in data:
            return data
        else:
            data.pop('mosaic')
        origin_input = copy.copy(data)
        img = data['image']
        height = img.shape[0] + self.border[0] * 2  # shape(h,w,c)
        width = img.shape[1] + self.border[1] * 2
        channels = 3  # default is rgb
        if len(img.shape) == 2:
            channels = 1
        if not isinstance(self.fill_color, list):
            self.fill_color = [self.fill_color] * channels

        for idx in range(self.max_try):
            M, s = self.generate_trans_matrix(img, width, height)

            # only care about gt_bboxes and gt_ignores
            if data.get('gt_bboxes', None) is not None:
                gt_targets = data['gt_bboxes']
                targets = self.label_transform(gt_targets, M, width, height, s)
                if targets.shape[0] == 0 and idx != self.max_try - 1:
                    continue
                origin_input['gt_bboxes'] = targets
            if data.get('gt_ignores', None) is not None:
                ig_targets = data['gt_ignores']
                origin_input['gt_ignores'] = self.label_transform(ig_targets, M, width, height, s)

            if (self.border[0] != 0) or (self.border[1] != 0) or (M != np.eye(3)).any():  # image changed
                if self.perspective:
                    img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=self.fill_color)
                else:  # affine
                    img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=self.fill_color)
                origin_input['image'] = img
            break

        return origin_input


@AUGMENTATION_REGISTRY.register('augment_hsv')
class AugmentHSV(Augmentation):
    """ hsv augment
    """
    def __init__(self, hgain=0.5, sgain=0.5, vgain=0.5, color_mode='RGB'):
        super(AugmentHSV, self).__init__()
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain
        self.color_mode = color_mode
        self.hsv2color = getattr(cv2, 'COLOR_HSV2{}'.format(color_mode))
        self.color2hsv = getattr(cv2, 'COLOR_{}2HSV'.format(color_mode))

    def augment(self, data):
        output = copy.copy(data)
        img = output.image
        assert len(img.shape) == 3, 'This augmentation only supports color images!'

        r = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(img, self.color2hsv))
        dtype = img.dtype  # uint8

        x = np.arange(0, 256, dtype=np.int16)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
        cv2.cvtColor(img_hsv, self.hsv2color, dst=img)

        output.image = img
        return output


@BATCHING_REGISTRY.register('center_pad')
class CenterPad(object):
    def __init__(self, alignment=1, pad_value=0):
        self.alignment = alignment
        self.pad_value = pad_value

    def __call__(self, data):
        """
        Args:
            images: list of tensor
        """
        images = data['image']
        gt_bboxes = None
        ig_bboxes = None
        if data.get('gt_bboxes', None) is not None:
            gt_bboxes = data['gt_bboxes']
        if data.get('gt_ignores', None) is not None:
            ig_bboxes = data['gt_ignores']
        max_img_h = max([_.size(-2) for _ in images])
        max_img_w = max([_.size(-1) for _ in images])
        target_h = int(np.ceil(max_img_h / self.alignment) * self.alignment)
        target_w = int(np.ceil(max_img_w / self.alignment) * self.alignment)
        padded_images = []
        padded_gt_bboxes = []
        padded_ig_bboxes = []
        for i, image in enumerate(images):
            assert image.dim() == 3
            src_h, src_w = image.size()[-2:]

            dw = (target_w - src_w) / 2
            dh = (target_h - src_h) / 2
            top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
            left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

            pad_size = (left, right, top, bottom)
            padded_images.append(F.pad(image, pad_size, 'constant', self.pad_value).data)

            # pad_w is left, pad_h is top
            data['image_info'][i][6] = left
            data['image_info'][i][7] = top

            if gt_bboxes is not None:
                gt_bbox = gt_bboxes[i]
                gt_bbox[:, [0, 2]] += left
                gt_bbox[:, [1, 3]] += top
                padded_gt_bboxes.append(gt_bbox)
            if ig_bboxes is not None:
                ig_bbox = ig_bboxes[i]
                ig_bbox[:, [0, 2]] += left
                ig_bbox[:, [1, 3]] += top
                padded_ig_bboxes.append(ig_bbox)
        data['image'] = torch.stack(padded_images)
        if gt_bboxes is not None:
            data['gt_bboxes'] = padded_gt_bboxes
        if ig_bboxes is not None:
            data['ig_bboxes'] = padded_ig_bboxes
        return data
