# Standard Library

# Import from third library
import torch
import cv2
import numpy as np

from up.data.datasets.transforms import Augmentation
from up.utils.general.registry_factory import AUGMENTATION_REGISTRY

__all__ = ['kp_flip',
           'crop_square',
           'random_scale',
           'random_rotate',
           'kp_label',
           'keep_aspect_ratio',
           'kp_crop']


def crop(image, ctr, box_w, box_h, rot, in_w, in_h,
         padding_val=np.array([123.675, 116.28, 103.53])):
    """
    crop image using input params
    :param image: input_image
    :param ctr: center
    :param box_w: box width
    :param box_h: box height
    :param rot: rotate angle
    :param in_w: the network input width
    :param in_h: the network input height
    :param padding_val: padding value when make border or warpAffine use
    :return:
    """
    pad_w, pad_h = int(box_w / 2), int(box_h / 2)

    # pad image
    pad_mat = cv2.copyMakeBorder(image,
                                 top=pad_h,
                                 bottom=pad_h,
                                 left=pad_w,
                                 right=pad_w,
                                 borderType=cv2.BORDER_CONSTANT,
                                 value=padding_val)

    left = ctr[0] - box_w / 2 + pad_w
    top = ctr[1] - box_h / 2 + pad_h
    right = ctr[0] + box_w / 2 + pad_w
    bottom = ctr[1] + box_h / 2 + pad_h
    l, t, r, b = map(int, [left, top, right, bottom])
    image_roi = pad_mat[t:b, l:r, :]
    image_roi = cv2.resize(image_roi, (in_w, in_h))

    # rotate
    rows, cols, channels = image_roi.shape
    assert channels == 3
    pad_ctr = (int(cols / 2), int(rows / 2))
    rot_matrix = cv2.getRotationMatrix2D(pad_ctr, rot, 1)
    ret = cv2.warpAffine(
        image_roi,
        rot_matrix,
        (in_w,
         in_h),
        flags=cv2.INTER_LINEAR,
        borderValue=padding_val)
    return ret


def clamp(x, lb, ub):
    """
    aux func
    :param x: input val
    :param lb: lower bound
    :param ub: upper bound
    :return:
    """
    return max(min(x, ub), lb)


@AUGMENTATION_REGISTRY.register('kp_flip')
class kp_flip(Augmentation):
    def __init__(self, match_parts):
        super(kp_flip, self).__init__()
        self.match_parts = match_parts

    def flip_point(self, input_kpts, roi_width, match_parts):
        """
        Args:
            input_kpts: keypoints for a person in ROI
            roi_width: ROI's width
            match_parts
        Return:
            None
        Side Effect:
            kpts is modified
        """
        kpts = input_kpts.copy()
        for i in range(len(kpts)):
            if kpts[i][0] >= 0:
                kpts[i][0] = roi_width - 1 - kpts[i][0]

        for left_index, right_index in match_parts:
            tmp = kpts[left_index].copy()
            kpts[left_index] = kpts[right_index].copy()
            kpts[right_index] = tmp
        return kpts

    def augment(self, data):
        cx, cy, bw, bh = data['bbox']
        img = data['image']
        kpts = data['kpts']
        if np.random.rand() > 0.5:
            img = cv2.flip(img, 1)
            cx = img.shape[1] - cx - 1
            kpts = self.flip_point(kpts, img.shape[1], self.match_parts)
        data['bbox'] = cx, cy, bw, bh
        data['image'] = img
        data['kpts'] = kpts
        return data


@AUGMENTATION_REGISTRY.register('kp_crop')
class kp_crop(Augmentation):
    def __init__(self, in_w, in_h, img_norm_factor=255.0, means=[0.485, 0.456, 0.406],
                 stds=[0.229, 0.224, 0.225]):
        super(kp_crop, self).__init__()
        self.in_w = in_w
        self.in_h = in_h
        self.img_norm_factor = float(img_norm_factor)
        self.means = np.array(means)
        self.stds = np.array(stds)
        self.padding_val = self.img_norm_factor * self.means

    def augment(self, data):
        img = data['image']
        cx, cy, bw, bh = data['bbox']
        if not data['flip']:
            rot = data['rot']
        else:
            rot = 0
            flip_img = data['flip_image']
            flip_roi_x, flip_roi_y, flip_roi_w, flip_roi_h = data['flip_bbox']
            flip_person = crop(
                flip_img,
                (flip_roi_x, flip_roi_y),
                flip_roi_w,
                flip_roi_h,
                rot,
                self.in_w,
                self.in_h,
                self.padding_val
            )
            flip_person = np.asarray(flip_person)
            flip_person = ((flip_person / self.img_norm_factor) - self.means) / self.stds
            data['flip_image'] = flip_person
        crop_file = crop(img,
                         (cx, cy),
                         bw,
                         bh,
                         rot,
                         self.in_w,
                         self.in_h,
                         self.padding_val)
        crop_file = np.array(crop_file).copy()
        crop_file = ((crop_file / self.img_norm_factor) - self.means) / self.stds
        data['image'] = crop_file
        return data


@AUGMENTATION_REGISTRY.register('kp_crop_square')
class crop_square(Augmentation):
    def augment(self, data):
        roi_x, roi_y, roi_w, roi_h = data['bbox']
        roi_w = roi_h = max(roi_w, roi_h)
        data['bbox'] = np.float32([roi_x, roi_y, roi_w, roi_h])
        if data['flip']:
            flip_roi_x, flip_roi_y, flip_roi_w, flip_roi_h = data['flip_bbox']
            flip_roi_w = flip_roi_h = max(flip_roi_w, flip_roi_h)
            data['flip_bbox'] = np.float32([flip_roi_x, flip_roi_y, flip_roi_w, flip_roi_h])
        return data


@AUGMENTATION_REGISTRY.register('kp_random_scale')
class random_scale(Augmentation):
    def __init__(self, scale_center=1, scale_factor=0.4):
        super(random_scale, self).__init__()
        self.scale_center = scale_center
        self.scale_factor = scale_factor

    def augment(self, data):
        """
        return random scale params
        :param scale_center: usually 1
        :param scale_factor: range of random scale, [1 - scale_scale,1 + scale_scale]

        :return: a final scale params
        """
        cx, cy, bw, bh = data['bbox']
        s = clamp(np.random.normal() * self.scale_factor + self.scale_center,
                  self.scale_center - self.scale_factor,
                  self.scale_center + self.scale_factor)

        bw *= s
        bh *= s
        data['bbox'] = cx, cy, bw, bh
        return data


@AUGMENTATION_REGISTRY.register('kp_random_rotate')
class random_rotate(Augmentation):
    def __init__(self, rotate_angle, rotate_prob=0.6):
        super(random_rotate, self).__init__()
        self.rotate_angle = rotate_angle
        self.rotate_prob = rotate_prob

    def augment(self, data):
        """
        return random rotate angle
        :param rotate_angle: rotate_angle
        :param rotate_prob: rotate probability
        :return:
        """
        # rot = np.clip(np.random.randn() * rotate_angle, -rotate_angle * 2, rotate_angle * 2) \
        #     if np.random.random() <= rotate_prob else 0
        if (np.random.rand() * 100 % 101) / 100.0 < self.rotate_prob:
            rot = 0
        else:
            rot = clamp(self.rotate_angle * np.random.normal(), -self.rotate_angle,
                        self.rotate_angle)
        data['rot'] = rot
        return data


@AUGMENTATION_REGISTRY.register('kp_keep_aspect_ratio')
class keep_aspect_ratio(Augmentation):
    def __init__(self, in_w, in_h):
        super(keep_aspect_ratio, self).__init__()
        self.in_w = in_w
        self.in_h = in_h

    def augment(self, data):
        roi_x, roi_y, roi_w, roi_h = data['bbox']
        gt_scale = 1. * self.in_w / self.in_h
        tmp_roi_w = max(roi_w, roi_h * gt_scale)
        tmp_roi_h = max(roi_h, roi_w / gt_scale)
        roi_w = tmp_roi_w
        roi_h = tmp_roi_h
        data['bbox'] = np.float32([roi_x, roi_y, roi_w, roi_h])
        if data['flip']:
            flip_roi_x, flip_roi_y, flip_roi_w, flip_roi_h = data['flip_bbox']
            tmp_flip_roi_w = max(flip_roi_w, flip_roi_h * gt_scale)
            tmp_flip_roi_h = max(flip_roi_h, flip_roi_w / gt_scale)
            flip_roi_w = tmp_flip_roi_w
            flip_roi_h = tmp_flip_roi_h
            data['flip_bbox'] = np.float32([flip_roi_x, flip_roi_y, flip_roi_w, flip_roi_h])

        return data


@AUGMENTATION_REGISTRY.register('kp_label_trans')
class kp_label(Augmentation):
    def __init__(self, bg, in_h, in_w, num_kpts, label_type, radius, strides):
        super(kp_label, self).__init__()
        self.radius = np.array(radius)  # radius for SCE supervision
        self.strides = np.array(strides)  # feature maps' strides
        self.bg = bg
        # self.strides = strides
        assert len(self.radius) == len(self.strides)
        self.out_h = list(map(int, np.float(in_h) / self.strides))
        self.out_w = list(map(int, np.float(in_w) / self.strides))
        self.num_kpts = num_kpts
        self.label_type = label_type
        # self.radius = radius

    def get_transform(self, center, w, h, rot, res_w, res_h):
        """
        General image processing functions
        """
        # Generate transformation matrix
        scale_w = float(res_w) / w
        scale_h = float(res_h) / h
        t = np.zeros((3, 3))
        t[0, 0] = scale_w
        t[1, 1] = scale_h
        t[2, 2] = 1
        t[0, 2] = -float(center[0]) * scale_w + .5 * res_w
        t[1, 2] = -float(center[1]) * scale_h + .5 * res_h
        if not rot == 0:
            rot = -rot  # To match direction of rotation from cropping
            rot_mat = np.zeros((3, 3))
            rot_rad = rot * np.pi / 180
            sn, cs = np.sin(rot_rad), np.cos(rot_rad)
            rot_mat[0, :2] = [cs, -sn]
            rot_mat[1, :2] = [sn, cs]
            rot_mat[2, 2] = 1
            # Need to rotate around center
            t_mat = np.eye(3)
            t_mat[0, 2] = -res_w / 2
            t_mat[1, 2] = -res_h / 2
            t_inv = t_mat.copy()
            t_inv[:2, 2] *= -1
            t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
        return t

    def kpt_transform(self, pts, center, box_w, box_h, rot, res_w, res_h, invert):
        t = self.get_transform(center, box_w, box_h, rot, res_w, res_h)
        if invert:
            t = np.linalg.inv(t)
        new_pt = np.array([pts[0], pts[1], 1.]).T
        new_pt = np.dot(t, new_pt)
        return new_pt[:2].astype(int)

    def put_gaussian_map(self, label_h, label_w, kpt, sigma=1.0):
        """
        paint gauss heatmap

        :param label_h: gauss heatmap height
        :param label_w: gauss heatmap width
        :param kpt: input keypoints position
        :param sigma: radius
        :return:
        """
        size = 2 * 3 * sigma + 1
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2
        radius = size // 2
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

        ret = np.zeros((label_h, label_w), dtype=np.float32)
        if kpt[0] < 0 or kpt[0] >= label_w or kpt[1] < 0 or kpt[1] >= label_h:
            return ret

        left = max(0, kpt[0] - radius)
        t = max(0, kpt[1] - radius)
        r = min(label_w - 1, kpt[0] + radius)
        b = min(label_h - 1, kpt[1] + radius)

        ml = x0 - min(kpt[0], radius)
        mt = y0 - min(kpt[1], radius)
        mr = x0 + min(label_w - 1 - kpt[0], radius)
        mb = y0 + min(label_h - 1 - kpt[1], radius)
        l, t, r, b = list(map(int, [left, t, r, b]))
        ml, mt, mr, mb = list(map(int, [ml, mt, mr, mb]))
        ret[t:b + 1, l:r + 1] = g[mt:mb + 1, ml:mr + 1]
        return ret

    def trans_kpt_func(self, kpts, ridx, cx, cy, bw, bh, rot, bg, out_h, out_w, num_kpts=17, label_type='Gaussian',
                       radius=None):
        if bg:
            out_channel = num_kpts + 1
        else:
            out_channel = num_kpts
        label = np.zeros((out_channel, out_h[ridx], out_w[ridx]), dtype=np.float32)
        # bg
        if bg:
            label[-1][:][:] = 1
        for kpt_idx in range(num_kpts):
            kpt = kpts[kpt_idx]
            if kpt[0] < 0:
                continue
            trans_kpt = self.kpt_transform(kpt, (cx, cy), bw, bh, rot, out_w[ridx], out_h[ridx], False)
            if label_type == 'SCE':
                r = radius[ridx]
                for x in range(-r, r + 1):
                    for y in range(-r, r + 1):
                        if np.abs(x) + np.abs(y) > r:
                            continue
                        px = trans_kpt[0] + x
                        py = trans_kpt[1] + y
                        if 0 <= px < out_w[ridx] and 0 <= py < out_h[ridx]:
                            label[kpt_idx][py][px] = 1
                            if bg:
                                label[-1][py][px] = 0
            else:
                label[kpt_idx] = self.put_gaussian_map(out_h[ridx], out_w[ridx], trans_kpt, radius[ridx])

        if bg and label_type == 'Gaussian':
            max_val = np.amax(label[:-1], axis=0)
            label[-1] -= max_val

        return torch.Tensor(label)

    def augment(self, data):
        all_labels = list()
        r = data['rot']
        cx, cy, bw, bh = data['bbox']
        for stride in range(len(self.strides)):
            all_labels.append(self.trans_kpt_func(data['kpts'], stride, cx, cy, bw, bh,
                                                  r, self.bg, self.out_h, self.out_w,
                                                  num_kpts=self.num_kpts,
                                                  label_type=self.label_type, radius=self.radius))
        data['label'] = all_labels
        return data
