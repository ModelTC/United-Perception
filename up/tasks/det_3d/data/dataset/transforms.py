import numpy as np
import copy
import os
from up.utils.general.registry_factory import AUGMENTATION_REGISTRY
from up.data.datasets.transforms import Augmentation
from up.extensions.python import iou3d_nms_utils
from up.tasks.det_3d.data.box_utils import boxes3d_kitti_fakelidar_to_lidar, enlarge_box3d, remove_points_in_boxes3d
from up.tasks.det_3d.data.box_utils import mask_boxes_outside_range_numpy
from up.tasks.det_3d.data.data_utils import rotate_points_along_z, limit_period, keep_arrays_by_name
from up.utils.general.petrel_helper import PetrelHelper
tv = None
try:
    import cumm.tensorview as tv
except BaseException:
    pass


@AUGMENTATION_REGISTRY.register('point_sampling')
class PointAugSampling(Augmentation):
    def __init__(self, root_path, class_names, db_info_paths, db_info_filters, sample_groups,
                 num_point_features, remove_extra_width, limit_whole_scene=False, use_road_plane=True,
                 database_with_fakelidar=False, logger=None):
        self.root_path = root_path
        self.class_names = class_names
        self.num_point_features = num_point_features
        self.remove_extra_width = remove_extra_width
        self.limit_whole_scene = limit_whole_scene
        self.use_road_plane = use_road_plane
        self.database_with_fakelidar = database_with_fakelidar
        self.logger = logger
        self.db_infos = self.get_db_infos(db_info_paths, db_info_filters)
        self.sample_groups_dict, self.sample_class_num = self.get_sample_group(sample_groups)

    def augment(self, data):
        data_dict = copy.copy(data)
        gt_boxes = data_dict['gt_boxes']
        gt_names = data_dict['gt_names'].astype(str)
        existed_boxes = gt_boxes
        total_valid_sampled_dict = []
        for class_name, sample_group in self.sample_groups_dict.items():
            if self.limit_whole_scene:
                num_gt = np.sum(class_name == gt_names)
                sample_group['sample_num'] = str(int(self.sample_class_num[class_name]) - num_gt)

            if int(sample_group['sample_num']) > 0:
                sampled_dict = self.sample_with_fixed_number(class_name, sample_group)
                sampled_boxes = np.stack([x['box3d_lidar'] for x in sampled_dict], axis=0).astype(np.float32)

                if self.database_with_fakelidar:
                    sampled_boxes = boxes3d_kitti_fakelidar_to_lidar(sampled_boxes)

                iou1 = iou3d_nms_utils.boxes_bev_iou_cpu(sampled_boxes[:, 0:7], existed_boxes[:, 0:7])
                iou2 = iou3d_nms_utils.boxes_bev_iou_cpu(sampled_boxes[:, 0:7], sampled_boxes[:, 0:7])
                iou2[range(sampled_boxes.shape[0]), range(sampled_boxes.shape[0])] = 0
                iou1 = iou1 if iou1.shape[1] > 0 else iou2
                valid_mask = ((iou1.max(axis=1) + iou2.max(axis=1)) == 0).nonzero()[0]
                valid_sampled_dict = [sampled_dict[x] for x in valid_mask]
                valid_sampled_boxes = sampled_boxes[valid_mask]
                existed_boxes = np.concatenate((existed_boxes, valid_sampled_boxes), axis=0)
                total_valid_sampled_dict.extend(valid_sampled_dict)

        sampled_gt_boxes = existed_boxes[gt_boxes.shape[0]:, :]
        if total_valid_sampled_dict.__len__() > 0:
            data_dict = self.add_sampled_boxes_to_scene(data_dict, sampled_gt_boxes, total_valid_sampled_dict)
        data_dict.pop('gt_boxes_mask')
        return data_dict

    def get_db_infos(self, db_info_paths, db_info_filters):
        db_infos = {}
        for class_name in self.class_names:
            db_infos[class_name] = []
        for db_info_path in db_info_paths:
            infos = PetrelHelper.load_pk(db_info_path, mode='rb')
            [db_infos[cur_class].extend(infos[cur_class]) for cur_class in self.class_names]
        for func_name, val in db_info_filters.items():
            db_infos = getattr(self, func_name)(db_infos, val)
        return db_infos

    def get_sample_group(self, sample_groups):
        sample_groups_dict = {}
        sample_class_num = {}
        for x in sample_groups:
            class_name, sample_num = x.split(':')
            if class_name not in self.class_names:
                continue
            sample_class_num[class_name] = sample_num
            sample_groups_dict[class_name] = {
                'sample_num': sample_num,
                'pointer': len(self.db_infos[class_name]),
                'indices': np.arange(len(self.db_infos[class_name]))
            }
        return sample_groups_dict, sample_class_num

    def filter_by_min_points(self, db_infos, min_gt_points_list):
        for name_num in min_gt_points_list:
            name, min_num = name_num.split(':')
            min_num = int(min_num)
            if min_num > 0 and name in db_infos.keys():
                filtered_infos = []
                for info in db_infos[name]:
                    if info['num_points_in_gt'] >= min_num:
                        filtered_infos.append(info)

                if self.logger is not None:
                    self.logger.info('Database filter by min points %s: %d => %d' %
                                     (name, len(db_infos[name]), len(filtered_infos)))
                db_infos[name] = filtered_infos

        return db_infos

    def filter_by_difficulty(self, db_infos, removed_difficulty):
        new_db_infos = {}
        for key, dinfos in db_infos.items():
            pre_len = len(dinfos)
            new_db_infos[key] = [
                info for info in dinfos
                if info['difficulty'] not in removed_difficulty
            ]
            if self.logger is not None:
                self.logger.info('Database filter by difficulty %s: %d => %d' % (key, pre_len, len(new_db_infos[key])))
        return new_db_infos

    def sample_with_fixed_number(self, class_name, sample_group):
        """
        Args:
            class_name:
            sample_group:
        Returns:

        """
        sample_num, pointer, indices = int(sample_group['sample_num']), sample_group['pointer'], sample_group['indices']
        if pointer >= len(self.db_infos[class_name]):
            indices = np.random.permutation(len(self.db_infos[class_name]))
            pointer = 0

        sampled_dict = [self.db_infos[class_name][idx] for idx in indices[pointer: pointer + sample_num]]
        pointer += sample_num
        sample_group['pointer'] = pointer
        sample_group['indices'] = indices
        return sampled_dict

    @staticmethod
    def put_boxes_on_road_planes(gt_boxes, road_planes, calib):
        """
        Only validate in KITTIDataset
        Args:
            gt_boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
            road_planes: [a, b, c, d]
            calib:

        Returns:
        """
        a, b, c, d = road_planes
        center_cam = calib.lidar_to_rect(gt_boxes[:, 0:3])
        cur_height_cam = (-d - a * center_cam[:, 0] - c * center_cam[:, 2]) / b
        center_cam[:, 1] = cur_height_cam
        cur_lidar_height = calib.rect_to_lidar(center_cam)[:, 2]
        mv_height = gt_boxes[:, 2] - gt_boxes[:, 5] / 2 - cur_lidar_height
        gt_boxes[:, 2] -= mv_height  # lidar view
        return gt_boxes, mv_height

    def add_sampled_boxes_to_scene(self, data_dict, sampled_gt_boxes, total_valid_sampled_dict):
        gt_boxes_mask = data_dict['gt_boxes_mask']
        gt_boxes = data_dict['gt_boxes'][gt_boxes_mask]
        gt_names = data_dict['gt_names'][gt_boxes_mask]
        points = data_dict['points']
        if self.use_road_plane:
            sampled_gt_boxes, mv_height = self.put_boxes_on_road_planes(
                sampled_gt_boxes, data_dict['road_plane'], data_dict['calib']
            )
            data_dict.pop('calib')
            data_dict.pop('road_plane')

        obj_points_list = []
        for idx, info in enumerate(total_valid_sampled_dict):
            file_path = os.path.join(self.root_path, info['path'])
            f = PetrelHelper._petrel_helper.load_data(file_path, ceph_read=False, fs_read=True, mode='rb')
            obj_points = np.frombuffer(f, dtype=np.float32).reshape(
                [-1, self.num_point_features]).copy()
            obj_points[:, :3] += info['box3d_lidar'][:3]
            if self.use_road_plane:
                # mv height
                obj_points[:, 2] -= mv_height[idx]
            obj_points_list.append(obj_points)

        obj_points = np.concatenate(obj_points_list, axis=0)
        sampled_gt_names = np.array([x['name'] for x in total_valid_sampled_dict])
        large_sampled_gt_boxes = enlarge_box3d(
            sampled_gt_boxes[:, 0:7], extra_width=self.remove_extra_width
        )
        points = remove_points_in_boxes3d(points, large_sampled_gt_boxes)
        points = np.concatenate([obj_points, points], axis=0)
        gt_names = np.concatenate([gt_names, sampled_gt_names], axis=0)
        gt_boxes = np.concatenate([gt_boxes, sampled_gt_boxes], axis=0)
        data_dict['gt_boxes'] = gt_boxes
        data_dict['gt_names'] = gt_names
        data_dict['points'] = points
        return data_dict


@AUGMENTATION_REGISTRY.register('point_flip')
class PointFlip(Augmentation):
    def __init__(self, along_axis_list):
        self.along_axis_list = along_axis_list

    def augment(self, data):
        data = copy.copy(data)
        gt_boxes, points = data['gt_boxes'], data['points']
        for cur_axis in self.along_axis_list:
            assert cur_axis in ['x', 'y']
            gt_boxes, points = getattr(self, 'random_flip_along_%s' % cur_axis)(
                gt_boxes, points,
            )
        data['gt_boxes'] = gt_boxes
        data['points'] = points
        return data

    def random_flip_along_x(self, gt_boxes, points):
        """
        Args:
            gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
            points: (M, 3 + C)
        Returns:
        """
        enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
        if enable:
            gt_boxes[:, 1] = -gt_boxes[:, 1]
            gt_boxes[:, 6] = -gt_boxes[:, 6]
            points[:, 1] = -points[:, 1]

            if gt_boxes.shape[1] > 7:
                gt_boxes[:, 8] = -gt_boxes[:, 8]

        return gt_boxes, points

    def random_flip_along_y(self, gt_boxes, points):
        """
        Args:
            gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
            points: (M, 3 + C)
        Returns:
        """
        enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
        if enable:
            gt_boxes[:, 0] = -gt_boxes[:, 0]
            gt_boxes[:, 6] = -(gt_boxes[:, 6] + np.pi)
            points[:, 0] = -points[:, 0]

            if gt_boxes.shape[1] > 7:
                gt_boxes[:, 7] = -gt_boxes[:, 7]

        return gt_boxes, points


@AUGMENTATION_REGISTRY.register('point_rotation')
class PointRotation(Augmentation):
    def __init__(self, rot_range):
        self.rot_range = rot_range

    def augment(self, data):
        output = copy.copy(data)
        output = self.random_world_rotation(output)
        return output

    def random_world_rotation(self, data_dict):
        if not isinstance(self.rot_range, list):
            self.rot_range = [-self.rot_range, self.rot_range]
        gt_boxes, points = self.global_rotation(
            data_dict['gt_boxes'], data_dict['points'], rot_range=self.rot_range
        )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def global_rotation(self, gt_boxes, points, rot_range):
        """
        Args:
            gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
            points: (M, 3 + C),
            rot_range: [min, max]
        Returns:
        """
        noise_rotation = np.random.uniform(rot_range[0], rot_range[1])
        points = rotate_points_along_z(points[np.newaxis, :, :], np.array([noise_rotation]))[0]
        gt_boxes[:, 0:3] = rotate_points_along_z(gt_boxes[np.newaxis, :, 0:3], np.array([noise_rotation]))[0]
        gt_boxes[:, 6] += noise_rotation
        if gt_boxes.shape[1] > 7:
            gt_boxes[:, 7:9] = rotate_points_along_z(
                np.hstack((gt_boxes[:, 7:9], np.zeros((gt_boxes.shape[0], 1))))[np.newaxis, :, :],
                np.array([noise_rotation])
            )[0][:, 0:2]

        return gt_boxes, points


@AUGMENTATION_REGISTRY.register('point_scaling')
class PointScaling(Augmentation):
    def __init__(self, scale_range):
        self.scale_range = scale_range

    def augment(self, data):
        output = copy.copy(data)
        output = self.random_world_scaling(output)
        return output

    def random_world_scaling(self, data_dict):
        gt_boxes, points = self.global_scaling(
            data_dict['gt_boxes'], data_dict['points'], self.scale_range
        )
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def global_scaling(self, gt_boxes, points, scale_range):
        """
        Args:
            gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading]
            points: (M, 3 + C),
            scale_range: [min, max]
        Returns:
        """
        if scale_range[1] - scale_range[0] < 1e-3:
            return gt_boxes, points
        noise_scale = np.random.uniform(scale_range[0], scale_range[1])
        points[:, :3] *= noise_scale
        gt_boxes[:, :6] *= noise_scale
        return gt_boxes, points


@AUGMENTATION_REGISTRY.register('point_to_voxel')
class PointProcessor(Augmentation):
    def __init__(self, point_cloud_range, num_point_features, voxel_size, max_points_per_voxel, max_number_of_voxels,
                 remove_outside_boxes=True, shuffle_enabled=False):
        self.point_cloud_range = point_cloud_range
        self.num_point_features = num_point_features
        self.voxel_size = voxel_size
        self.max_number_of_voxels = max_number_of_voxels
        self.max_points_per_voxel = max_points_per_voxel
        self.remove_outside_boxes = remove_outside_boxes
        self.shuffle_enabled = shuffle_enabled
        grid_size = (np.array(self.point_cloud_range[3:6])
                     - np.array(self.point_cloud_range[0:3])) / np.array(self.voxel_size)
        self.grid_size = np.round(grid_size).astype(np.int64)

    def augment(self, data):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
        """
        data_dict = copy.copy(data)
        self.training = data_dict['training']
        self.class_names = data_dict['class_names']
        points = data_dict.get('points', None)
        gt_boxes = data_dict.get('gt_bboxes', None)

        data_dict = self.before_to_voxel(data_dict)
        points = self.points_encoding(points)
        points, gt_boxes = self.mask_points_and_boxes_outside_range(points, gt_boxes)
        voxels, coordinates, num_points = self.transform_points_to_voxels(points)
        voxel_infos = {'grid_size': self.grid_size, 'num_point_features': self.num_point_features,
                       'voxel_size': self.voxel_size, 'point_cloud_range': self.point_cloud_range}
        data_dict.update({'points': points, 'voxels': voxels, 'voxel_coords': coordinates,
                          'voxel_num_points': num_points, 'voxel_infos': voxel_infos})
        return data_dict

    def before_to_voxel(self, data_dict):
        if self.training:
            data_dict['gt_boxes'][:, 6] = limit_period(
                data_dict['gt_boxes'][:, 6], offset=0.5, period=2 * np.pi
            )
            if 'calib' in data_dict:
                data_dict.pop('calib')
            if 'road_plane' in data_dict:
                data_dict.pop('road_plane')
            if 'gt_boxes_mask' in data_dict:
                gt_boxes_mask = data_dict['gt_boxes_mask']
                data_dict['gt_boxes'] = data_dict['gt_boxes'][gt_boxes_mask]
                data_dict['gt_names'] = data_dict['gt_names'][gt_boxes_mask]
                if 'gt_boxes2d' in data_dict:
                    data_dict['gt_boxes2d'] = data_dict['gt_boxes2d'][gt_boxes_mask]
                data_dict.pop('gt_boxes_mask')
        if data_dict.get('gt_boxes', None) is not None:
            selected = keep_arrays_by_name(data_dict['gt_names'], self.class_names)
            data_dict['gt_boxes'] = data_dict['gt_boxes'][selected]
            data_dict['gt_names'] = data_dict['gt_names'][selected]
            gt_classes = np.array([self.class_names.index(n) + 1 for n in data_dict['gt_names']], dtype=np.int32)
            gt_boxes = np.concatenate((data_dict['gt_boxes'], gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)
            data_dict['gt_boxes'] = gt_boxes

            if data_dict.get('gt_boxes2d', None) is not None:
                data_dict['gt_boxes2d'] = data_dict['gt_boxes2d'][selected]
        return data_dict

    def mask_points_and_boxes_outside_range(self, points, gt_boxes):
        if points is not None:
            mask = mask_points_by_range(points, self.point_cloud_range)
            points = points[mask]

        if gt_boxes is not None and self.remove_outside_boxes and self.training:
            mask = mask_boxes_outside_range_numpy(gt_boxes, self.point_cloud_range)
            gt_boxes = gt_boxes[mask]
        return points, gt_boxes

    def transform_points_to_voxels(self, points):
        if self.shuffle_enabled:
            shuffle_idx = np.random.permutation(points.shape[0])
            points = points[shuffle_idx]

        self.voxel_generator = VoxelGeneratorWrapper(
            vsize_xyz=self.voxel_size,
            coors_range_xyz=self.point_cloud_range,
            num_point_features=self.num_point_features,
            max_num_points_per_voxel=self.max_points_per_voxel,
            max_num_voxels=self.max_number_of_voxels,
        )
        voxel_output = self.voxel_generator.generate(points)  # Point2VoxelCPU3d
        voxels, coordinates, num_points = voxel_output
        if not self.use_lead_xyz:
            voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)

        return voxels, coordinates, num_points

    def points_encoding(self, points):
        used_feature_list = ['x', 'y', 'z', 'intensity']
        src_feature_list = ['x', 'y', 'z', 'intensity']
        if points is None:
            num_output_features = len(used_feature_list)
            self.use_lead_xyz = False
            return num_output_features

        point_feature_list = [points[:, 0:3]]
        for x in used_feature_list:
            if x in ['x', 'y', 'z']:
                continue
            idx = src_feature_list.index(x)
            point_feature_list.append(points[:, idx:idx + 1])
        point_features = np.concatenate(point_feature_list, axis=1)
        self.use_lead_xyz = True
        return point_features


def mask_points_by_range(points, limit_range):
    mask = (points[:, 0] >= limit_range[0]) & (points[:, 0] <= limit_range[3]) \
        & (points[:, 1] >= limit_range[1]) & (points[:, 1] <= limit_range[4])
    return mask


class VoxelGeneratorWrapper():
    def __init__(self, vsize_xyz, coors_range_xyz, num_point_features, max_num_points_per_voxel, max_num_voxels):
        # try:
        #     from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
        #     self.spconv_ver = 1
        # except BaseException:
        #     try:
        #         from spconv.utils import VoxelGenerator
        #         self.spconv_ver = 1
        #     except BaseException:
        #         from spconv.utils import Point2VoxelCPU3d as VoxelGenerator
        #         self.spconv_ver = 2

        from up.tasks.det_3d.models.utils import VoxelGenerator
        self.spconv_ver = 1

        if self.spconv_ver == 1:
            self._voxel_generator = VoxelGenerator(
                voxel_size=vsize_xyz,
                point_cloud_range=coors_range_xyz,
                max_num_points=max_num_points_per_voxel,
                max_voxels=max_num_voxels
            )
        else:  # Point2VoxelCPU3d
            self._voxel_generator = VoxelGenerator(
                vsize_xyz=vsize_xyz,
                coors_range_xyz=coors_range_xyz,
                num_point_features=num_point_features,
                max_num_points_per_voxel=max_num_points_per_voxel,
                max_num_voxels=max_num_voxels
            )

    def generate(self, points):
        if self.spconv_ver == 1:
            voxel_output = self._voxel_generator.generate(points)
            if isinstance(voxel_output, dict):
                voxels, coordinates, num_points = \
                    voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
            else:
                voxels, coordinates, num_points = voxel_output
        else:
            assert tv is not None, "Unexpected error, library: 'cumm' wasn't imported properly."
            voxel_output = self._voxel_generator.point_to_voxel(tv.from_numpy(points))
            tv_voxels, tv_coordinates, tv_num_points = voxel_output
            # make copy with numpy(), since numpy_view() will disappear as soon as the generator is deleted
            voxels = tv_voxels.numpy()
            coordinates = tv_coordinates.numpy()
            num_points = tv_num_points.numpy()
        return voxels, coordinates, num_points
