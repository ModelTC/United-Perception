import torch
import torch.nn.functional as F
from up.utils.general.registry_factory import ROI_PREDICTOR_REGISTRY
from up.tasks.det_3d.models.utils.model_nms_utils import class_agnostic_nms
from up.tasks.det_3d.models.utils.center_utils import _transpose_and_gather_feat, _topk


__all__ = ['CenterPredictor']


@ROI_PREDICTOR_REGISTRY.register('center_head')
class CenterPredictor(object):
    def __init__(self, score_thresh, post_center_limit_range, max_obj_per_sample,
                 feature_map_stride, cfg=None, output_raw_score=False, predict_boxes_when_training=False,
                 hm_pool_type=None, hm_pool_size=None):
        super().__init__()
        self.score_thresh = score_thresh
        self.output_raw_score = output_raw_score
        self.feature_map_stride = feature_map_stride
        self.nms_cfg = cfg
        self.post_center_limit_range = post_center_limit_range
        self.max_obj_per_sample = max_obj_per_sample
        self.predict_boxes_when_training = predict_boxes_when_training
        self.hm_pool_type = hm_pool_type
        self.hm_pool_size = hm_pool_size

    def predict(self, batch_size, voxel_infos, pred_dict):
        self.voxel_size = voxel_infos['voxel_size']
        self.point_cloud_range = voxel_infos['point_cloud_range']
        post_center_limit_range = torch.tensor(self.post_center_limit_range).cuda().float()
        ret_dict = [{
            'pred_boxes': [],
            'pred_scores': [],
            'pred_labels': [],
        } for k in range(batch_size)]

        batch_hm = pred_dict['hm'].sigmoid()
        batch_center = pred_dict['center']
        batch_center_z = pred_dict['center_z']
        batch_dim = pred_dict['dim'].exp()
        batch_rot_cos = pred_dict['rot'][:, 0].unsqueeze(dim=1)
        batch_rot_sin = pred_dict['rot'][:, 1].unsqueeze(dim=1)
        batch_vel = None
        circle_nms = (self.nms_cfg['nms_type'] == 'circle_nms') if self.nms_cfg else None
        final_pred_dict = self.decode_bbox_from_heatmap(
            heatmap=batch_hm, rot_cos=batch_rot_cos, rot_sin=batch_rot_sin,
            center=batch_center, center_z=batch_center_z, dim=batch_dim, vel=batch_vel,
            point_cloud_range=self.point_cloud_range, voxel_size=self.voxel_size,
            feature_map_stride=self.feature_map_stride,
            K=self.max_obj_per_sample,
            circle_nms=circle_nms,
            score_thresh=self.score_thresh,
            post_center_limit_range=post_center_limit_range
        )
        # nms
        for k, final_dict in enumerate(final_pred_dict):
            final_dict['pred_labels'] = final_dict['pred_labels'].long()
            if self.nms_cfg and (self.nms_cfg['nms_type'] != 'circle_nms'):
                selected, selected_scores = class_agnostic_nms(
                    box_scores=final_dict['pred_scores'], box_preds=final_dict['pred_boxes'],
                    nms_config=self.nms_cfg,
                    score_thresh=None
                )
                final_dict['pred_boxes'] = final_dict['pred_boxes'][selected]
                final_dict['pred_scores'] = selected_scores
                final_dict['pred_labels'] = final_dict['pred_labels'][selected]
            ret_dict[k]['pred_boxes'].append(final_dict['pred_boxes'])
            ret_dict[k]['pred_scores'].append(final_dict['pred_scores'])
            ret_dict[k]['pred_labels'].append(final_dict['pred_labels'])

        for k in range(batch_size):
            ret_dict[k]['pred_boxes'] = torch.cat(ret_dict[k]['pred_boxes'], dim=0)
            ret_dict[k]['pred_scores'] = torch.cat(ret_dict[k]['pred_scores'], dim=0)
            ret_dict[k]['pred_labels'] = torch.cat(ret_dict[k]['pred_labels'], dim=0) + 1
        output = {}
        if self.predict_boxes_when_training:
            rois, roi_scores, roi_labels = self.reorder_rois_for_refining(batch_size, ret_dict)
            output.update({'rois': rois, 'roi_scores': roi_scores, 'roi_labels': roi_labels, 'has_class_labels': True})
        else:
            output.update({'pred_list': ret_dict})
        return output

    @staticmethod
    def reorder_rois_for_refining(batch_size, pred_dicts):
        num_max_rois = max([len(cur_dict['pred_boxes']) for cur_dict in pred_dicts])
        num_max_rois = max(1, num_max_rois)  # at least one faked rois to avoid error
        pred_boxes = pred_dicts[0]['pred_boxes']

        rois = pred_boxes.new_zeros((batch_size, num_max_rois, pred_boxes.shape[-1]))
        roi_scores = pred_boxes.new_zeros((batch_size, num_max_rois))
        roi_labels = pred_boxes.new_zeros((batch_size, num_max_rois)).long()

        for bs_idx in range(batch_size):
            num_boxes = len(pred_dicts[bs_idx]['pred_boxes'])

            rois[bs_idx, :num_boxes, :] = pred_dicts[bs_idx]['pred_boxes']
            roi_scores[bs_idx, :num_boxes] = pred_dicts[bs_idx]['pred_scores']
            roi_labels[bs_idx, :num_boxes] = pred_dicts[bs_idx]['pred_labels']
        return rois, roi_scores, roi_labels

    def decode_bbox_from_heatmap(self, heatmap, rot_cos, rot_sin, center, center_z, dim,
                                 point_cloud_range=None, voxel_size=None, feature_map_stride=None, vel=None, K=100,
                                 circle_nms=False, score_thresh=None, post_center_limit_range=None):
        batch_size, num_class, _, _ = heatmap.size()

        if self.hm_pool_type:
            heatmap = self.hm_pool(heatmap, self.hm_pool_type, self.hm_pool_size)

        scores, inds, class_ids, ys, xs = _topk(heatmap, K=K)

        center = _transpose_and_gather_feat(center, inds).view(batch_size, K, 2)
        rot_sin = _transpose_and_gather_feat(rot_sin, inds).view(batch_size, K, 1)
        rot_cos = _transpose_and_gather_feat(rot_cos, inds).view(batch_size, K, 1)
        center_z = _transpose_and_gather_feat(center_z, inds).view(batch_size, K, 1)
        dim = _transpose_and_gather_feat(dim, inds).view(batch_size, K, 3)
        angle = torch.atan2(rot_sin, rot_cos)
        xs = xs.view(batch_size, K, 1) + center[:, :, 0:1]
        ys = ys.view(batch_size, K, 1) + center[:, :, 1:2]
        xs = xs * feature_map_stride * voxel_size[0] + point_cloud_range[0]
        ys = ys * feature_map_stride * voxel_size[1] + point_cloud_range[1]

        box_part_list = [xs, ys, center_z, dim, angle]
        if vel is not None:
            vel = _transpose_and_gather_feat(vel, inds).view(batch_size, K, 2)
            box_part_list.append(vel)
        final_box_preds = torch.cat((box_part_list), dim=-1)
        final_scores = scores.view(batch_size, K)
        final_class_ids = class_ids.view(batch_size, K)

        assert post_center_limit_range is not None
        mask = (final_box_preds[..., :3] >= post_center_limit_range[:3]).all(2)
        mask &= (final_box_preds[..., :3] <= post_center_limit_range[3:]).all(2)
        if score_thresh is not None:
            mask &= (final_scores > score_thresh)

        ret_pred_dict = []
        for k in range(batch_size):
            cur_mask = mask[k]
            cur_boxes = final_box_preds[k, cur_mask]
            cur_scores = final_scores[k, cur_mask]
            cur_labels = final_class_ids[k, cur_mask]
            ret_pred_dict.append({
                'pred_boxes': cur_boxes,
                'pred_scores': cur_scores,
                'pred_labels': cur_labels
            })
        return ret_pred_dict

    def hm_pool(self, heatmap, hm_pool_type, hm_pool_size):
        if hm_pool_type == 'max_pool':
            heatmap = hm_maxpool_square(heatmap, hm_pool_size)
        return heatmap


def hm_maxpool_square(heatmap, pool_size=3):
    r"""
    apply max pooling to get the same effect of nms

    Args:
        fmap(Tensor): output tensor of previous step
        pool_size(int): size of max-pooling
    """
    if isinstance(pool_size, int):
        pad = (pool_size - 1) // 2
    else:
        assert pool_size[0] != pool_size[1]
        pad = (pool_size[0] - 1) // 2

    heatmap_max = F.max_pool2d(heatmap, pool_size, stride=1, padding=pad)
    keep = (heatmap_max == heatmap).float()
    return heatmap * keep
