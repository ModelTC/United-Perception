from __future__ import division
# Standard Library

# Import from third library
import cv2
import numpy as np
from easydict import EasyDict

# Import from local
# from spring.data import SPRING_DATASET
from up.data.datasets.base_dataset import BaseDataset
# from eod.data.datasets.base_dataset import BaseDataset
from up.utils.general.petrel_helper import PetrelHelper
from up.utils.general.registry_factory import DATASET_REGISTRY

__all__ = ['KeypointDataset']
# TODO: Check GPU usage after move this setting down from upper line
cv2.ocl.setUseOpenCL(False)


@DATASET_REGISTRY.register('keypoint')
class KeypointDataset(BaseDataset):
    def __init__(self,
                 meta_file,
                 image_reader,
                 num_kpts=17,
                 mode='train',
                 inst_score_thres=0,
                 box_scale=1.25,
                 evaluator=None,
                 transformer=None,
                 flip=False
                 ):

        super(KeypointDataset, self).__init__(meta_file=meta_file,
                                              image_reader=image_reader,
                                              transformer=transformer,
                                              evaluator=evaluator
                                              )
        self.num_kpts = num_kpts  # predict keypoints num
        self.mode = mode
        self.flip = flip
        self.box_scale = box_scale
        annos = PetrelHelper.load_json(meta_file)
        self.path = []
        self.bbox = []
        if self.mode == 'train':
            id_path_hash = {}
            for img_info in annos['images']:
                id_path_hash[img_info['id']] = img_info['file_name']

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
        else:
            assert 'annotations' in annos
            annos = annos['annotations']
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

    def __len__(self):
        return len(self.path)

    def get_input(self, idx):
        filename = self.path[idx]
        bbox = self.bbox[idx]
        img = self.image_reader(filename)
        if self.mode == 'train':
            kpts = self.kpts[idx].copy()
            input = EasyDict({
                'image': img,
                'bbox': bbox,
                'kpts': kpts,
                'image_id': idx,
                'flip': self.flip,
                'filename': filename
            })
        else:
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
            center = ((x1 + x2) / 2, (y1 + y2) / 2)

            flip_roi_w = (flip_x2 - flip_x1 + 1) * self.box_scale
            flip_roi_h = (flip_y2 - flip_y1 + 1) * self.box_scale
            flip_center = ((flip_x1 + flip_x2) / 2, (flip_y1 + flip_y2) / 2)

            input = EasyDict({
                'image': img,
                'flip_image': flip_img,
                'image_id': idx,
                'flip': self.flip,
                'filename': filename,
                'bbox': np.float32([center[0], center[1], roi_w, roi_h]),
                'flip_bbox': np.float32([flip_center[0], flip_center[1], flip_roi_w, flip_roi_h]),
                'dt_box': np.float32([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1 + 1, y2 - y1 + 1]),
                'box_score': np.float32(box_score), 'imw': img.shape[1], 'imh': img.shape[0]
            })
        return input

    def __getitem__(self, idx):
        input = self.get_input(idx)
        input = self.transformer(input)
        return input

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
