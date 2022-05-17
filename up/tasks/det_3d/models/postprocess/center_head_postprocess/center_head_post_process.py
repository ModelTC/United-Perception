import torch
import torch.nn as nn
from up.utils.general.registry_factory import MODULE_ZOO_REGISTRY
from up.tasks.det.models.postprocess.roi_supervisor import build_roi_supervisor
from up.tasks.det.models.postprocess.roi_predictor import build_roi_predictor
from up.tasks.det_3d.models.utils.center_utils import _transpose_and_gather_feat
from up.models.losses import build_loss

__all__ = ['CenterPostProcess']


@MODULE_ZOO_REGISTRY.register('center_head_post')
class CenterPostProcess(nn.Module):
    def __init__(self, cfg):
        super(CenterPostProcess, self).__init__()
        self.prefix = self.__class__.__name__
        self.supervisor = build_roi_supervisor(cfg['roi_supervisor'])
        self.predictor = build_roi_predictor(cfg['roi_predictor'])
        self.hm_loss = build_loss(cfg['hm_loss'])
        self.loc_loss = build_loss(cfg['loc_loss'])

    def forward(self, input):
        class_names, voxel_infos, batch_size, gt_boxes, pred_dict = input['class_names'],\
            input['voxel_infos'], input['batch_size'], input['gt_boxes'], input['pred_dict']
        feature_map_size = input['spatial_features_2d'].size()[2:]
        output = {}
        if self.training:
            targets = self.supervisor.get_targets(class_names, voxel_infos, gt_boxes, feature_map_size)
            losses = self.get_loss(targets, pred_dict)
            output.update(losses)
        else:
            results = self.predictor.predict(batch_size, voxel_infos, pred_dict)
            output.update(results)
        return output

    def get_loss(self, targets, pred_dict):
        heatmaps_target, boxes_target, inds, masks = targets
        output = {}
        normalizer = torch.clamp_min(masks.float().sum(), min=1.0)
        # fix the order of heads
        boxes_pred = torch.cat([pred_dict[head_name] for head_name in ['center', 'center_z', 'dim', 'rot']], dim=1)
        boxes_pred = _transpose_and_gather_feat(boxes_pred, inds)
        masks = masks.unsqueeze(2).expand_as(boxes_target).float()
        isnotnan = (~ torch.isnan(boxes_target)).float()
        masks *= isnotnan
        boxes_pred = boxes_pred * masks
        boxes_target = boxes_target * masks

        loc_loss = self.loc_loss(boxes_pred, boxes_target, normalizer_override=normalizer)
        hm_loss = self.hm_loss(pred_dict['hm'], heatmaps_target)

        output.update({self.prefix + '.hm_loss': hm_loss, self.prefix + '.loc_loss': loc_loss})
        return output
