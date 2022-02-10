import torch
import copy
import numpy as np
from torch import nn
from torch.nn import functional as F
from pycocotools import mask as mask_utils
from eod.utils.general.context import config
from eod.tasks.det.plugins.condinst.models.head.condinst_head import aligned_bilinear
from eod.tasks.det.models.utils.anchor_generator import build_anchor_generator
from eod.utils.general.registry_factory import MODULE_ZOO_REGISTRY
from .condinst_supervisor import build_mask_supervisor
from .condinst_predictor import build_mask_predictor


def poly_to_mask(polygons, height, width):
    rles = mask_utils.frPyObjects(polygons, height, width)
    rle = mask_utils.merge(rles)
    mask = mask_utils.decode(rle).astype(np.bool)
    mask = torch.from_numpy(mask)
    return mask


def add_bitmasks(gt_masks, image, mask_out_stride):
    """
        - gt_masks(list of list of list of array):[B * [num_inst * [array]]]
    """
    im_h = image.shape[-2]
    im_w = image.shape[-1]
    device = image.device
    gt_bitmasks = []
    gt_bitmasks_full = []
    for gt_masks_per_img in gt_masks:
        per_im_bitmasks = []
        per_im_bitmasks_full = []
        for polygon in gt_masks_per_img:
            bitmask = poly_to_mask(polygon, im_h, im_w)
            bitmask = bitmask.to(device).float()
            start = int(mask_out_stride // 2)
            bitmask_full = bitmask.clone()
            bitmask = bitmask[start::mask_out_stride, start::mask_out_stride]
            assert bitmask.size(0) * mask_out_stride == im_h
            assert bitmask.size(1) * mask_out_stride == im_w
            per_im_bitmasks.append(bitmask)
            per_im_bitmasks_full.append(bitmask_full)
        gt_bitmasks.append(torch.stack(per_im_bitmasks, dim=0))
        gt_bitmasks_full.append(torch.stack(per_im_bitmasks_full, dim=0))
    return gt_bitmasks, gt_bitmasks_full


def parse_dynamic_params(params, channels, weight_nums, bias_nums):
    assert params.dim() == 2
    assert len(weight_nums) == len(bias_nums)
    assert params.size(1) == sum(weight_nums) + sum(bias_nums)

    num_insts = params.size(0)
    num_layers = len(weight_nums)

    params_splits = list(torch.split_with_sizes(
        params, weight_nums + bias_nums, dim=1
    ))

    weight_splits = params_splits[:num_layers]
    bias_splits = params_splits[num_layers:]

    for layer_id in range(num_layers):
        if layer_id < num_layers - 1:
            # out_channels x in_channels x 1 x 1
            weight_splits[layer_id] = weight_splits[layer_id].reshape(num_insts * channels, -1, 1, 1)
            bias_splits[layer_id] = bias_splits[layer_id].reshape(num_insts * channels)
        else:
            # out_channels x in_channels x 1 x 1
            weight_splits[layer_id] = weight_splits[layer_id].reshape(num_insts * 1, -1, 1, 1)
            bias_splits[layer_id] = bias_splits[layer_id].reshape(num_insts)

    return weight_splits, bias_splits


class DynamicMaskHead(nn.Module):
    def __init__(self, in_channels, channels, num_convs, mask_out_stride, mask_feat_stride, sizes_of_interest,
                 disable_rel_coords):
        super(DynamicMaskHead, self).__init__()
        self.channels = channels
        self.num_convs = num_convs
        self.mask_out_stride = mask_out_stride
        self.disable_rel_coords = disable_rel_coords
        self.in_channels = in_channels
        self.mask_feat_stride = mask_feat_stride
        self.register_buffer("sizes_of_interest", torch.tensor(sizes_of_interest + [sizes_of_interest[-1] * 2]))

        weight_nums, bias_nums = [], []
        for convs_id in range(self.num_convs):
            if convs_id == 0:
                if not self.disable_rel_coords:
                    weight_nums.append((self.in_channels + 2) * self.channels)
                else:
                    weight_nums.append(self.in_channels * self.channels)
                bias_nums.append(self.channels)
            elif convs_id == self.num_convs - 1:
                weight_nums.append(self.channels * 1)
                bias_nums.append(1)
            else:
                weight_nums.append(self.channels * self.channels)
                bias_nums.append(self.channels)
        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.num_gen_params = sum(weight_nums) + sum(bias_nums)

    def mask_heads_forward_with_coords(
            self,
            mask_feats,
            locations,
            mask_head_params,
            fpn_levels,
            instance_locations,
            im_inds):
        N, _, H, W = mask_feats.size()
        n_inst = len(mask_head_params)

        if not self.disable_rel_coords:
            relative_coords = instance_locations.reshape(-1, 1, 2) - locations.reshape(1, -1, 2)
            relative_coords = relative_coords.permute(0, 2, 1).float()
            soi = self.sizes_of_interest[fpn_levels]
            relative_coords = relative_coords.div_(soi.reshape(-1, 1, 1)).to(dtype=mask_feats.dtype)
            mask_head_inputs = torch.cat([
                relative_coords, mask_feats[im_inds].reshape(n_inst, self.in_channels, H * W)
            ], dim=1)
        else:
            mask_head_inputs = mask_feats[im_inds].reshape(n_inst, self.in_channels, H * W)
        mask_head_inputs = mask_head_inputs.reshape(1, -1, H, W)

        weights, biases = parse_dynamic_params(
            mask_head_params, self.channels,
            self.weight_nums, self.bias_nums
        )

        mask_logits = self.mask_heads_forward(mask_head_inputs, weights, biases, n_inst)
        mask_logits = mask_logits.reshape(-1, 1, H, W)
        assert self.mask_feat_stride >= self.mask_out_stride
        assert self.mask_feat_stride % self.mask_out_stride == 0
        mask_logits = aligned_bilinear(mask_logits, int(self.mask_feat_stride / self.mask_out_stride))
        return mask_logits

    def mask_heads_forward(self, features, weights, biases, num_insts):
        assert features.dim() == 4
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv2d(
                x, w, bias=b,
                stride=1, padding=0,
                groups=num_insts
            )
            if i < n_layers - 1:
                x = F.relu(x)
        return x


def build_condinst_dynamic_mask_head(cfg):
    return DynamicMaskHead(cfg)


@MODULE_ZOO_REGISTRY.register('condinst_post')
class CondinstPostProcess(nn.Module):
    def __init__(
            self,
            dense_points,
            mask_gen_params,
            branch_num_outputs,
            head_channels,
            head_num_convs,
            mask_out_stride,
            mask_feat_stride,
            disable_rel_coords,
            sizes_of_interest,
            cfg):
        super(CondinstPostProcess, self).__init__()

        self.mask_out_stride = mask_out_stride
        self.mask_feat_stride = mask_feat_stride
        self.mask_gen_params = mask_gen_params
        self.dense_points = dense_points
        with config(cfg, 'train') as train_cfg:
            self.supervisor = build_mask_supervisor(train_cfg['condinst_supervisor'])
        with config(cfg, 'test') as test_cfg:
            self.predictor = build_mask_predictor(test_cfg['condinst_predictor'])
        self.mask_head = DynamicMaskHead(
            branch_num_outputs,
            head_channels,
            head_num_convs,
            self.mask_out_stride,
            self.mask_feat_stride,
            sizes_of_interest,
            disable_rel_coords)
        self.center_generator = build_anchor_generator(cfg['center_generator'])
        self.prefix = 'CondinstMask'
        self.cfg = copy.deepcopy(cfg)

    def forward(self, input):
        mlvl_controller = input['mlvl_controller']
        mask_feats = input['mask_feats']
        controller = self.get_controller(mlvl_controller)
        locations = self.center_generator.compute_locations_per_lever(
            mask_feats.shape[2], mask_feats.shape[3], self.mask_feat_stride, device=mask_feats.device)
        if self.training:
            gt_inds, mask_head_params, fpn_levels, instance_locations, im_inds = self.supervisor.get_targets(
                input, controller)
            mask_losses = self.get_loss(input, mask_feats, locations, gt_inds, mask_head_params,
                                        fpn_levels, instance_locations, im_inds)
            return mask_losses
        else:
            if len(input['dt_bboxes']) > 0:
                results = self.predictor.predict(self.mask_head, input, locations, controller, self.mask_gen_params)
            return results

    def get_controller(self, mlvl_controller):
        """Permute preds from [B, A*C, H, W] to [B, H*W*A, C] """
        mlvl_permuted_controller = []
        for lvl_idx, preds in enumerate(mlvl_controller):
            b, _, h, w = preds.shape
            k = self.dense_points * h * w
            preds = preds.permute(0, 2, 3, 1).contiguous().view(b, k, -1)
            mlvl_permuted_controller.append(preds)
        return mlvl_permuted_controller

    def get_loss(self, input, mask_feats, locations, gt_inds, mask_head_params, fpn_levels,
                 instance_locations, im_inds):
        if 'gt_bitmasks' not in input.keys():
            image = input['image']
            gt_masks = input['gt_masks']
            gt_bitmasks, gt_bitmasks_full = add_bitmasks(gt_masks, image, self.mask_out_stride)
        else:
            gt_bitmasks = input['gt_bitmasks']

        if len(im_inds) == 0:
            mask_loss = mask_feats.sum() * 0 + mask_head_params.sum() * 0
            return {self.prefix + ".mask_loss": mask_loss}
        else:
            mask_logits = self.mask_head.mask_heads_forward_with_coords(
                mask_feats, locations, mask_head_params, fpn_levels, instance_locations, im_inds)
            mask_scores = mask_logits.sigmoid()
            gt_bitmasks = torch.cat([gt_bitmask for gt_bitmask in gt_bitmasks])
            gt_bitmasks = gt_bitmasks[gt_inds].unsqueeze(dim=1).to(dtype=mask_logits.dtype)
            mask_losses = self.dice_coefficient(mask_scores, gt_bitmasks)
            return {self.prefix + '.mask_loss': mask_losses.mean()}

    def dice_coefficient(self, x, target):
        eps = 1e-5
        n_inst = x.size(0)
        x = x.reshape(n_inst, -1)
        target = target.reshape(n_inst, -1)
        intersection = (x * target).sum(dim=1)
        union = (x ** 2.0).sum(dim=1) + (target ** 2.0).sum(dim=1) + eps
        loss = 1. - (2 * intersection / union)
        return loss
