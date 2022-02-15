from __future__ import division

# Standard Library
import json

# Import from third library
import torch
import cv2
import numpy as np
from easydict import EasyDict

# Import from local
# from spring.data import SPRING_DATASET
from .kp_base_dataset import BaseDataset
# from eod.data.datasets.base_dataset import BaseDataset
from up.utils.general.petrel_helper import PetrelHelper
from up.utils.general.registry_factory import DATASET_REGISTRY

__all__ = ['KeypointTrainDataset', 'KeypointTestDataset']
# TODO: Check GPU usage after move this setting down from upper line
cv2.ocl.setUseOpenCL(False)


# utils
def clamp(x, lb, ub):
    """
    aux func
    :param x: input val
    :param lb: lower bound
    :param ub: upper bound
    :return:
    """
    return max(min(x, ub), lb)


def random_scale(scale_center=1, scale_factor=0.4):
    """
    return random scale params
    :param scale_center: usually 1
    :param scale_factor: range of random scale, [1 - scale_scale,1 + scale_scale]

    :return: a final scale params
    """

    return clamp(np.random.normal() * scale_factor + scale_center,
                 scale_center - scale_factor,
                 scale_center + scale_factor)


def random_rotate(rotate_angle, rotate_prob=0.6):
    """
    return random rotate angle
    :param rotate_angle: rotate_angle
    :param rotate_prob: rotate probability
    :return:
    """
    # rot = np.clip(np.random.randn() * rotate_angle, -rotate_angle * 2, rotate_angle * 2) \
    #     if np.random.random() <= rotate_prob else 0

    if (np.random.rand() * 100 % 101) / 100.0 < rotate_prob:
        rot = 0
    else:
        rot = clamp(rotate_angle * np.random.normal(), -rotate_angle,
                    rotate_angle)
    return rot


def flip_point(input_kpts, roi_width, num_kpts):
    """
    Args:
        input_kpts: keypoints for a person in ROI
        roi_width: ROI's width
        num_kpts : keypoints number
    Return:
        None
    Side Effect:
        kpts is modified
    """
    kpts = input_kpts.copy()
    for i in range(len(kpts)):
        if kpts[i][0] >= 0:
            kpts[i][0] = roi_width - 1 - kpts[i][0]

    # # After flip, the person's left hand will become to right hand and so on.
    if num_kpts == 17:
        match_parts = {1: 2, 3: 4, 5: 6, 7: 8, 9: 10, 11: 12, 13: 14, 15: 16}
    elif num_kpts == 14:
        match_parts = {0: 3, 1: 4, 2: 5, 6: 9, 7: 10, 8: 11}
    else:
        raise NotImplementedError('undefined match relation for kpt_num = ' + str(num_kpts))
    for left_index, right_index in match_parts.items():
        tmp = kpts[left_index].copy()
        kpts[left_index] = kpts[right_index].copy()
        kpts[right_index] = tmp
    return kpts


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


def get_transform(center, w, h, rot, res_w, res_h):
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


def kpt_transform(pts, center, box_w, box_h, rot, res_w, res_h, invert):
    t = get_transform(center, box_w, box_h, rot, res_w, res_h)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pts[0], pts[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int)


def put_gaussian_map(label_h, label_w, kpt, sigma=1.0):
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


def trans_kpt_func(kpts, ridx, cx, cy, bw, bh, rot, bg, out_h, out_w, num_kpts=17, label_type='Gaussian', radius=None):
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
        trans_kpt = kpt_transform(kpt, (cx, cy), bw, bh, rot, out_w[ridx], out_h[ridx], False)
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
            label[kpt_idx] = put_gaussian_map(out_h[ridx], out_w[ridx], trans_kpt, radius[ridx])

    if bg and label_type == 'Gaussian':
        max_val = np.amax(label[:-1], axis=0)
        label[-1] -= max_val

    return torch.Tensor(label)


@DATASET_REGISTRY.register('keypoint_train')
class KeypointTrainDataset(BaseDataset):
    def __init__(self,
                 meta_file,
                 image_reader,
                 source='base',
                 num_kpts=17,
                 radius=[],
                 strides=[],
                 keep_aspect_ratio=True,
                 means=[128.0, 128.0, 128.0],
                 stds=[256.0, 256.0, 256.0],
                 img_norm_factor=1.0,
                 crop_square=False,
                 in_h=256,
                 in_w=256,
                 scale_center=1.25,
                 scale_factor=0.3125,
                 rot=30,
                 label_type='Gaussian',
                 bg=True,
                 flip=None,
                 evaluator=None
                 ):

        super(KeypointTrainDataset, self).__init__(meta_file=meta_file,
                                                   image_reader=image_reader,
                                                   transformer=None,
                                                   evaluator=evaluator,
                                                   # source=source
                                                   )
        self.num_kpts = num_kpts  # predict keypoints num
        self.bg = bg

        # Data
        self.crop_square = crop_square  # square input or not
        self.keep_aspect_ratio = keep_aspect_ratio

        # Normalization image = ((image / self.img_norm_factor) - self.means) /
        # self.stds
        self.means = np.array(means)
        self.stds = np.array(stds)
        self.img_norm_factor = float(img_norm_factor)

        # Network
        self.in_h = in_h  # input height of network
        self.in_w = in_w  # input width of network
        self.label_type = label_type  # Gaussian or SCE supervision
        self.radius = np.array(radius)  # radius for SCE supervision
        self.strides = np.array(strides)  # feature maps' strides

        # Data augmentation
        self.scale_center = scale_center
        self.scale_factor = scale_factor
        self.rot = rot
        self.flip = flip

        assert len(self.radius) == len(self.strides)
        self.out_h = list(map(int, np.float(in_h) / self.strides))
        self.out_w = list(map(int, np.float(in_w) / self.strides))

        # Read from json
        annos = PetrelHelper.load_json(meta_file)

        id_path_hash = {}
        for img_info in annos['images']:
            id_path_hash[img_info['id']] = img_info['file_name']

        self.path = []
        self.bbox = []
        self.kpts = []
        for anno in annos['annotations']:
            if anno['num_keypoints'] == 0:
                continue
            self.path.append(id_path_hash[anno['image_id']])
            _bbox = anno['bbox']
            # xywh -> cxcywh
            bbox = _bbox[0] + _bbox[2] / 2, _bbox[1] + _bbox[3] / 2, _bbox[2], _bbox[3]
            self.bbox.append(bbox)
            flags = anno['keypoints'][2::3]
            assert (len(flags) == self.num_kpts)
            kpts = []
            for flag_idx, the_flag in enumerate(flags):
                if the_flag == 0:
                    kpts.append([-1, -1, the_flag])
                else:
                    kpts.append(
                        anno['keypoints'][flag_idx * 3:flag_idx * 3 + 3])
            kpts = np.reshape(kpts, (self.num_kpts, 3))
            self.kpts.append(kpts)

    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx):
        filename = self.path[idx]
        bbox = self.bbox[idx]
        kpts = self.kpts[idx].copy()

        # image
        img = self.image_reader(filename)
        # box
        cx, cy, bw, bh = bbox

        # keep aspect ratio
        if self.keep_aspect_ratio:
            gt_scale = 1. * self.in_w / self.in_h
            tmp_bw = max(bw, bh * gt_scale)
            tmp_bh = max(bh, bw / gt_scale)
            bw = tmp_bw
            bh = tmp_bh
        # random scale
        s = random_scale(
            scale_center=self.scale_center,
            scale_factor=self.scale_factor)
        # random rotate
        rot = random_rotate(rotate_angle=self.rot)

        bw *= s
        bh *= s

        if self.crop_square:
            bw = bh = max(bw, bh)

        # flip
        if self.flip is not None and np.random.rand() > 0.5:
            img = cv2.flip(img, 1)
            cx = img.shape[1] - cx - 1
            kpts = flip_point(kpts, img.shape[1], self.num_kpts)

        # crop person
        person = crop(img, (cx, cy), bw, bh, rot, self.in_w,
                      self.in_h, padding_val=self.img_norm_factor * self.means)
        person = np.asarray(person).copy()

        # mean and std
        person = ((person / self.img_norm_factor) - self.means) / self.stds

        all_labels = list()
        for stride in range(len(self.strides)):
            all_labels.append(trans_kpt_func(kpts, stride, cx, cy, bw, bh,
                                             rot, self.bg, self.out_h, self.out_w,
                                             num_kpts=self.num_kpts,
                                             label_type=self.label_type, radius=self.radius))
        person = np.transpose(person, (2, 0, 1))  # HxWxC -> CxHxW
        item = EasyDict({
            'image': torch.Tensor(person),
            'label': all_labels,
            'image_id': idx,
            'filename': filename
        })
        return item

    def dump(self, output):
        all_res = output['res']
        out_res = []
        for _idx in range(len(all_res)):
            path = all_res[_idx]['image_id']
            bbox = all_res[_idx]['bbox']
            box_score = all_res[_idx]['box_score']
            kpts = all_res[_idx]['keypoints']
            kpt_score = all_res[_idx]['kpt_score']
            if bbox[2] * bbox[3] < 32 * 32:
                continue
            res = {
                'image_id': path,
                'bbox': bbox,
                'box_score': float('%.8f' % box_score),
                'keypoints': kpts,
                'kpt_score': float('%.8f' % kpt_score),
                'category_id': 1,
                'score': box_score * kpt_score
            }
            #     writer.write(json.dumps(res, ensure_ascii=False) + '\n')
            # writer.flush()
            out_res.append(res)
        return out_res


@DATASET_REGISTRY.register('keypoint_test')
class KeypointTestDataset(BaseDataset):
    def __init__(self,
                 meta_file,
                 image_reader,
                 source='base',
                 num_kpts=17,
                 keep_aspect_ratio=True,
                 means=[128.0, 128.0, 128.0],
                 stds=[256.0, 256.0, 256.0],
                 img_norm_factor=1.0,
                 crop_square=False,
                 in_h=256,
                 in_w=256,
                 box_scale=1.25,
                 inst_score_thres=0,
                 evaluator=None
                 ):

        super(KeypointTestDataset, self).__init__(meta_file=meta_file,
                                                  image_reader=image_reader,
                                                  transformer=None,
                                                  evaluator=evaluator,
                                                  # source=source
                                                  )
        self.num_kpts = num_kpts  # predict keypoints num

        # Normalization image = ((image / self.img_norm_factor) - self.means) /
        # self.stds
        self.means = np.array(means)
        self.stds = np.array(stds)
        self.img_norm_factor = float(img_norm_factor)
        self.padding_val = self.img_norm_factor * self.means
        # Data
        self.crop_square = crop_square  # crop square or not
        self.keep_aspect_ratio = keep_aspect_ratio
        self.box_scale = box_scale  # scale factor for scale the bbox

        # Network
        self.in_h = in_h
        self.in_w = in_w

        annos = PetrelHelper.load_json(meta_file)
        assert 'annotations' in annos
        annos = annos['annotations']

        self.path = []
        self.bbox = []
        for anno in annos:
            img_name = '{0:012d}.jpg'.format(anno['image_id'])
            box = anno['bbox']  # x, y, w, h
            if 'score' in anno.keys():
                score = anno['score']
            else:
                score = 1.0
            if score < inst_score_thres:
                continue
            if box[2] < 2 or box[3] < 2:
                continue
            box.extend([score])
            box[2] = box[0] + box[2] - 1
            box[3] = box[1] + box[3] - 1
            self.path.append(img_name)
            self.bbox.append(box)

        assert len(self.path) == len(self.bbox)

    def __getitem__(self, idx):
        filename = self.path[idx]
        bbox = self.bbox[idx]

        # image
        img = self.image_reader(filename)

        # box
        x1, y1, x2, y2, box_score = bbox
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img.shape[1] - 1, x2)
        y2 = min(img.shape[0] - 1, y2)

        flip_img = cv2.flip(img, 1)
        flip_x1 = img.shape[1] - 1 - x2
        flip_x2 = img.shape[1] - 1 - x1
        flip_x1 = max(0, flip_x1)
        flip_x2 = min(img.shape[1] - 1, flip_x2)
        flip_y1 = y1
        flip_y2 = y2

        roi_w = (x2 - x1 + 1) * self.box_scale
        roi_h = (y2 - y1 + 1) * self.box_scale
        if self.crop_square:
            roi_w = roi_h = max(roi_w, roi_h)

        # keep aspect ratio
        if self.keep_aspect_ratio:
            gt_scale = 1. * self.in_w / self.in_h
            tmp_roi_w = max(roi_w, roi_h * gt_scale)
            tmp_roi_h = max(roi_h, roi_w / gt_scale)
            roi_w = tmp_roi_w
            roi_h = tmp_roi_h

        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        person = crop(
            img,
            center,
            roi_w,
            roi_h,
            0,
            self.in_w,
            self.in_h,
            self.padding_val
        )
        person = np.asarray(person)

        flip_roi_w = (flip_x2 - flip_x1 + 1) * self.box_scale
        flip_roi_h = (flip_y2 - flip_y1 + 1) * self.box_scale
        if self.crop_square:
            flip_roi_w = flip_roi_h = max(flip_roi_w, flip_roi_h)

        # keep aspect ratio
        if self.keep_aspect_ratio:
            gt_scale = 1. * self.in_w / self.in_h
            tmp_flip_roi_w = max(flip_roi_w, flip_roi_h * gt_scale)
            tmp_flip_roi_h = max(flip_roi_h, flip_roi_w / gt_scale)
            flip_roi_w = tmp_flip_roi_w
            flip_roi_h = tmp_flip_roi_h

        flip_center = ((flip_x1 + flip_x2) / 2, (flip_y1 + flip_y2) / 2)
        flip_person = crop(
            flip_img,
            flip_center,
            flip_roi_w,
            flip_roi_h,
            0,
            self.in_w,
            self.in_h,
            self.padding_val
        )
        flip_person = np.asarray(flip_person)

        # mean and std
        person = ((person / self.img_norm_factor) - self.means) / self.stds
        flip_person = ((flip_person / self.img_norm_factor) - self.means) / self.stds

        person = np.transpose(person, (2, 0, 1))  # HxWxC -> CxHxW
        flip_person = np.transpose(flip_person, (2, 0, 1))  # HxWxC -> CxHxW

        item = EasyDict({
            'image': torch.Tensor(person),
            'flip_image': torch.Tensor(flip_person),
            'filename': filename,
            'bbox': np.float32([center[0], center[1], roi_w, roi_h]),
            'dt_box': np.float32([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1 + 1, y2 - y1 + 1]),
            'box_score': np.float32(box_score), 'imw': img.shape[1], 'imh': img.shape[0]
        })
        return item

    def __len__(self):
        return len(self.path)

    def dump(self, output):
        all_res = output['res']
        out_res = []
        for _idx in range(len(all_res)):
            path = all_res[_idx]['image_id']
            bbox = all_res[_idx]['bbox']
            box_score = all_res[_idx]['box_score']
            kpts = all_res[_idx]['keypoints']
            kpt_score = all_res[_idx]['kpt_score']
            if bbox[2] * bbox[3] < 32 * 32:
                continue
            res = {
                'image_id': path,
                'bbox': bbox,
                'box_score': float('%.8f' % box_score),
                'keypoints': kpts,
                'kpt_score': float('%.8f' % kpt_score),
                'category_id': 1,
                'score': box_score * kpt_score
            }
            #     writer.write(json.dumps(res, ensure_ascii=False) + '\n')
            # writer.flush()
            out_res.append(res)
        return out_res
