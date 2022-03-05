# Import from third library
import copy
import numpy as np
import pickle
import os
from collections import defaultdict
from up.tasks.det_3d.data.box_utils import boxes3d_lidar_to_kitti_camera, boxes3d_kitti_camera_to_imageboxes
from up.utils.general.registry_factory import DATASET_REGISTRY
from up.tasks.det_3d.data.data_utils import get_pad_params
from up.data.datasets.base_dataset import BaseDataset


@DATASET_REGISTRY.register('kitti')
class KittiDataset(BaseDataset):
    def __init__(self,
                 meta_file,
                 image_reader,
                 transformer,
                 evaluator=None,
                 class_names=None,
                 get_item_list=None,
                 fov_points_only=True,
                 training=True
                 ):
        super(KittiDataset, self).__init__(meta_file, image_reader, transformer, evaluator, class_names)
        self.root_dir = self.image_reader.root_dir
        self.training = training
        self.get_item_list = get_item_list
        self.fov_points_only = fov_points_only
        self.split = 'train' if self.training else 'val'
        self.root_split = 'training' if self.training else 'testing'
        self.kitti_infos = []
        self.load_kitti_info()

    def load_kitti_info(self):
        kitti_infos = []
        with open(self.meta_file, 'rb') as f:
            infos = pickle.load(f)
            kitti_infos.extend(infos)

        self.kitti_infos.extend(kitti_infos)

    def __len__(self):
        return len(self.kitti_infos)

    def drop_info_with_name(self, info, name):
        ret_info = {}
        keep_indices = [i for i, x in enumerate(info['name']) if x != name]
        for key in info.keys():
            ret_info[key] = info[key][keep_indices]
        return ret_info

    def boxes3d_kitti_camera_to_lidar(self, boxes3d_camera, calib):
        """
        Args:
            boxes3d_camera: (N, 7) [x, y, z, l, h, w, r] in rect camera coords
            calib:

        Returns:
            boxes3d_lidar: [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

        """
        boxes3d_camera_copy = copy.deepcopy(boxes3d_camera)
        xyz_camera, r = boxes3d_camera_copy[:, 0:3], boxes3d_camera_copy[:, 6:7]
        l, h, w = boxes3d_camera_copy[:, 3:4], boxes3d_camera_copy[:, 4:5], boxes3d_camera_copy[:, 5:6]
        xyz_lidar = calib.rect_to_lidar(xyz_camera)
        xyz_lidar[:, 2] += h[:, 0] / 2
        return np.concatenate([xyz_lidar, l, w, h, -(r + np.pi / 2)], axis=-1)

    def calib_to_matricies(self, calib):
        """
        Converts calibration object to transformation matricies
        Args:
            calib: calibration.Calibration, Calibration object
        Returns
            V2R: (4, 4), Lidar to rectified camera transformation matrix
            P2: (3, 4), Camera projection matrix
        """
        V2C = np.vstack((calib.V2C, np.array([0, 0, 0, 1], dtype=np.float32)))  # (4, 4)
        R0 = np.hstack((calib.R0, np.zeros((3, 1), dtype=np.float32)))  # (3, 4)
        R0 = np.vstack((R0, np.array([0, 0, 0, 1], dtype=np.float32)))  # (4, 4)
        V2R = R0 @ V2C
        P2 = calib.P2
        return V2R, P2

    @staticmethod
    def get_fov_flag(pts_rect, img_shape, calib):
        """
        Args:
            pts_rect:
            img_shape:
            calib:

        Returns:

        """
        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

        return pts_valid_flag

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """
        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
                'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
                'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
                'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
                'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict

        def generate_single_sample_dict(batch_index, box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            calib = batch_dict['calib'][batch_index]
            image_shape = batch_dict['image_shape'][batch_index].cpu().numpy()
            pred_boxes_camera = boxes3d_lidar_to_kitti_camera(pred_boxes, calib)
            pred_boxes_img = boxes3d_kitti_camera_to_imageboxes(
                pred_boxes_camera, calib, image_shape=image_shape
            )

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['alpha'] = -np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0]) + pred_boxes_camera[:, 6]
            pred_dict['bbox'] = pred_boxes_img
            pred_dict['dimensions'] = pred_boxes_camera[:, 3:6]
            pred_dict['location'] = pred_boxes_camera[:, 0:3]
            pred_dict['rotation_y'] = pred_boxes_camera[:, 6]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            frame_id = batch_dict['frame_id'][index]

            single_pred_dict = generate_single_sample_dict(index, box_dict)
            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)

            if output_path is not None:
                cur_det_file = os.path.join(output_path, '{}.txt'.format(frame_id))
                with open(cur_det_file, 'w') as f:
                    bbox = single_pred_dict['bbox']
                    loc = single_pred_dict['location']
                    dims = single_pred_dict['dimensions']  # lhw -> hwl

                    for idx in range(len(bbox)):
                        print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                              % (single_pred_dict['name'][idx], single_pred_dict['alpha'][idx],
                                 bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3],
                                 dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0],
                                 loc[idx][1], loc[idx][2], single_pred_dict['rotation_y'][idx],
                                 single_pred_dict['score'][idx]), file=f)

        return annos

    def dump(self, output):
        pred_dicts, ret_dict = output

    def evaluate(self, res_file, class_names, kitti_infos, res=None):
        """
        Arguments:
            - res_file (:obj:`str`): filename
        """
        metrics = self.evaluator.eval(res_file, class_names, kitti_infos, res) if self.evaluator else {}
        return metrics

    @staticmethod
    def collate_batch(batch_list, _unused=False):
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_size = len(batch_list)
        ret = {}

        for key, val in data_dict.items():
            try:
                if key in ['voxels', 'voxel_num_points']:
                    ret[key] = np.concatenate(val, axis=0)
                elif key in ['points', 'voxel_coords']:
                    coors = []
                    for i, coor in enumerate(val):
                        coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                        coors.append(coor_pad)
                    ret[key] = np.concatenate(coors, axis=0)
                elif key in ['gt_boxes']:
                    max_gt = max([len(x) for x in val])
                    batch_gt_boxes3d = np.zeros((batch_size, max_gt, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_gt_boxes3d
                elif key in ['gt_boxes2d']:
                    max_boxes = 0
                    max_boxes = max([len(x) for x in val])
                    batch_boxes2d = np.zeros((batch_size, max_boxes, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        if val[k].size > 0:
                            batch_boxes2d[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_boxes2d
                elif key in ["images", "depth_maps"]:
                    # Get largest image size (H, W)
                    max_h = 0
                    max_w = 0
                    for image in val:
                        max_h = max(max_h, image.shape[0])
                        max_w = max(max_w, image.shape[1])

                    # Change size of images
                    images = []
                    for image in val:
                        pad_h = get_pad_params(desired_size=max_h, cur_size=image.shape[0])
                        pad_w = get_pad_params(desired_size=max_w, cur_size=image.shape[1])
                        pad_width = (pad_h, pad_w)
                        # Pad with nan, to be replaced later in the pipeline.
                        pad_value = np.nan

                        if key == "images":
                            pad_width = (pad_h, pad_w, (0, 0))
                        elif key == "depth_maps":
                            pad_width = (pad_h, pad_w)

                        image_pad = np.pad(image,
                                           pad_width=pad_width,
                                           mode='constant',
                                           constant_values=pad_value)

                        images.append(image_pad)
                    ret[key] = np.stack(images, axis=0)
                elif key in ['training', 'class_names', 'voxel_infos']:
                    ret[key] = val[0]
                else:
                    ret[key] = np.stack(val, axis=0)
            except BaseException:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError

        ret['batch_size'] = batch_size
        return ret

    def get_input(self, idx):
        info = copy.deepcopy(self.kitti_infos[idx])
        sample_idx = info['point_cloud']['lidar_idx']
        img_shape = info['image']['image_shape']
        calib = self.image_reader.get_calib(sample_idx)

        input_dict = {
            'frame_id': sample_idx,
            'calib': calib,
        }
        if 'annos' in info:
            annos = info['annos']
            annos = self.drop_info_with_name(annos, name='DontCare')
            loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
            gt_names = annos['name']
            gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
            gt_boxes_lidar = self.boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calib)

            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })
            if "gt_boxes2d" in self.get_item_list:
                input_dict['gt_boxes2d'] = annos["bbox"]

            road_plane = self.image_reader.get_road_plane(sample_idx)
            if road_plane is not None:
                input_dict['road_plane'] = road_plane
        if "points" in self.get_item_list:
            points = self.image_reader.get_lidar(sample_idx)
            if self.fov_points_only:
                pts_rect = calib.lidar_to_rect(points[:, 0:3])
                fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)
                points = points[fov_flag]
            input_dict['points'] = points
        if "images" in self.get_item_list:
            input_dict['images'] = self.image_reader.get_image(sample_idx)
        if "depth_maps" in self.get_item_list:
            input_dict['depth_maps'] = self.image_reader.get_depth_map(sample_idx)
        if "calib_matricies" in self.get_item_list:
            input_dict["trans_lidar_to_cam"], input_dict["trans_cam_to_img"] = self.calib_to_matricies(calib)
        input_dict['image_shape'] = img_shape

        if self.training:
            assert 'gt_boxes' in input_dict, 'gt_boxes should be provided for training'
            gt_boxes_mask = np.array([n in self.class_names for n in input_dict['gt_names']], dtype=np.bool_)
            input_dict.update({'gt_boxes_mask': gt_boxes_mask})
        return input_dict

    def __getitem__(self, idx):
        input = self.get_input(idx)
        input.update({'training': self.training, 'class_names': self.class_names})
        input = self.transformer(input)
        if self.training and len(input['gt_boxes']) == 0:
            new_index = np.random.randint(self.__len__())
            return self.__getitem__(new_index)
        input.pop('gt_names', None)
        return input
