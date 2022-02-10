import torch
import copy
from eod.utils.general.registry_factory import MASK_SUPERVISOR_REGISTRY
from eod.utils.general.fp16_helper import to_float32
from eod.utils.general.log_helper import default_logger as logger


@MASK_SUPERVISOR_REGISTRY.register('condinst')
class MaskSupervisorCondinst(object):
    def __init__(self, max_proposals, topk_proposals_per_im):
        self.max_proposals = max_proposals
        self.topk_proposals_per_im = topk_proposals_per_im

    @to_float32
    def get_targets(self, input, controller):
        mlvl_locations = input['mlvl_locations']
        targets = input['targets']
        gt_bboxes = input['gt_bboxes']
        gt_inds = input['gt_inds']

        assert (self.max_proposals == -1) or (self.topk_proposals_per_im == -1), \
            "MAX_PROPOSALS and TOPK_PROPOSALS_PER_IM cannot be used at the same time."

        num_loc_list = [len(loc) for loc in mlvl_locations]
        K = sum(num_loc_list)
        cls_target, loc_target, cls_mask, loc_mask = targets
        B = loc_mask.shape[0]
        pos_inds = self._transpose(loc_mask, num_loc_list)
        pos_inds = torch.cat([x.reshape(-1) for x in pos_inds])
        targets_num = len(torch.nonzero(pos_inds))

        if self.max_proposals != -1:
            if self.max_proposals < targets_num:
                inds = torch.randperm(targets_num, device=pos_inds.device).long()[:self.max_proposals]
                clip_index = torch.arange(0, K * B, device=pos_inds.device, dtype=torch.long)[pos_inds][inds]
                pos_inds = pos_inds.new_zeros(K * B, device=pos_inds.device, dtype=torch.bool)
                pos_inds[clip_index] = True
                logger.info("clipping proposals from {} to {}".format(targets_num, self.max_proposals))
        im_inds = [pos_inds.new_ones(K, dtype=torch.long) * i for i in range(B)]
        im_inds = self._transpose(im_inds, num_loc_list)
        im_inds = torch.cat([x.reshape(-1) for x in im_inds])[pos_inds]

        mask_gen_params = controller[0].shape[2]
        mask_head_params = torch.cat([x.reshape(-1, mask_gen_params) for x in controller])[pos_inds]

        num_targets = 0
        target_inds = copy.deepcopy(gt_inds)
        for im_i in range(B):
            target_inds[im_i] += num_targets
            num_targets += len(gt_bboxes[im_i])
        target_inds = self._transpose(target_inds, num_loc_list)
        gt_inds = torch.cat([x.reshape(-1) for x in target_inds])[pos_inds]

        instance_locations = torch.cat([x.repeat(B, 1) for x in mlvl_locations])[pos_inds]

        fpn_levels = [loc.new_ones(len(loc), dtype=torch.long) * level for level, loc in enumerate(mlvl_locations)]
        fpn_levels = torch.cat([x.repeat(B) for x in fpn_levels])[pos_inds]
        return gt_inds, mask_head_params, fpn_levels, instance_locations, im_inds

    def _transpose(self, training_targets, num_loc_list):
        '''
        This function is used to transpose image first training targets to level first ones
        :return: level first training targets
        '''
        targets_list = []
        for im_i in range(len(training_targets)):
            targets_list.append(torch.split(training_targets[im_i], num_loc_list, dim=0))

        targets_level_first = []
        for targets_per_level in zip(*targets_list):
            targets_level_first.append(
                torch.cat(targets_per_level, dim=0)
            )
        return targets_level_first


def build_mask_supervisor(supervisor_cfg):
    return MASK_SUPERVISOR_REGISTRY.build(supervisor_cfg)
