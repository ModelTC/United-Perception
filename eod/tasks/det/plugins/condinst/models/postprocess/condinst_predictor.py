import torch
from torch.nn import functional as F
from eod.utils.general.registry_factory import MASK_PREDICTOR_REGISTRY
from eod.utils.general.fp16_helper import to_float32
from eod.tasks.det.plugins.condinst.models.head.condinst_head import aligned_bilinear


@MASK_PREDICTOR_REGISTRY.register('condinst')
class MaskPredictorCondinst(object):
    def __init__(self,):
        pass

    @torch.no_grad()
    @to_float32
    def predict(self, mask_head, input, locations, controller, mask_gen_params):
        mask_feats = input['mask_feats']
        image_info = input['image_info']
        image = input['image']
        bboxes = input['dt_bboxes']

        mask_head_params, fpn_levels, instance_locations, im_inds, pred_boxes = self.get_pred_instances(
            input, controller, mask_gen_params)
        mask_logits = mask_head.mask_heads_forward_with_coords(
            mask_feats, locations, mask_head_params, fpn_levels, instance_locations, im_inds)
        pred_global_masks = mask_logits.sigmoid()

        dt_bboxes = []
        dt_masks = []
        for im_id, (image_size,) in enumerate(zip(image_info)):
            ind_per_im = torch.nonzero(im_inds == im_id)[:, 0]
            pred_masks, ind_per_im_keep = self.postprocess(
                image, ind_per_im, image_size, pred_boxes, pred_global_masks
            )
            dt_bboxes.append(bboxes[ind_per_im_keep])
            for idx in range(len(ind_per_im_keep)):
                dt_masks.append(pred_masks[idx].detach().cpu().numpy())
        dt_bboxes = torch.cat(dt_bboxes, dim=0)
        return {'dt_masks': dt_masks, 'dt_bboxes': dt_bboxes}

    def get_pred_instances(self, input, controller, mask_gen_params):
        B = controller[0].shape[0]
        K = sum([x.shape[1] for x in controller])
        bboxes = input['dt_bboxes']
        pos_inds = input['pos_inds']
        im_inds, cls_rois, scores, cls = torch.split(bboxes, [1, 4, 1, 1], dim=1)
        im_inds = im_inds.squeeze().type(torch.LongTensor).to(pos_inds.device)
        pos_inds = pos_inds.squeeze().add(im_inds * K).type(torch.LongTensor)
        mask_head_params = torch.cat(controller, dim=1).reshape(-1, mask_gen_params)[pos_inds]
        mlvl_locations = input['mlvl_locations']
        instance_locations = torch.cat(mlvl_locations).repeat(B, 1)[pos_inds]
        fpn_levels = torch.cat([mlvl_locations[lvl_num].new_ones(len(mlvl_locations[lvl_num]),
                               dtype=torch.long) * lvl_num for lvl_num in range(len(mlvl_locations))])
        fpn_levels = fpn_levels.repeat(B)[pos_inds].type(torch.LongTensor)
        return mask_head_params, fpn_levels, instance_locations, im_inds, cls_rois

    def postprocess(self, image, ind_per_im, image_size, pred_boxes, pred_global_masks=None, mask_threshold=0.5):
        padded_im_h, padded_im_w = (image.shape[-2], image.shape[-1])
        resized_im_h, resized_im_w = (image_size[0], image_size[1])
        output_height, output_width = (image_size[3], image_size[4])
        scale_x, scale_y = (output_width / resized_im_w, output_height / resized_im_h)

        output_boxes = pred_boxes[ind_per_im]

        output_boxes[:, 0::2] *= scale_x
        output_boxes[:, 1::2] *= scale_y
        output_boxes[:, 0] = torch.clamp(output_boxes[:, 0], min=0, max=output_width)
        output_boxes[:, 1] = torch.clamp(output_boxes[:, 1], min=0, max=output_height)
        output_boxes[:, 2] = torch.clamp(output_boxes[:, 2], min=0, max=output_width)
        output_boxes[:, 3] = torch.clamp(output_boxes[:, 3], min=0, max=output_height)

        keep_inds = ((output_boxes[:, 2] - output_boxes[:, 0])
                     > 0.0) & ((output_boxes[:, 3] - output_boxes[:, 1]) > 0.0)
        ind_per_im = ind_per_im[keep_inds]

        if pred_global_masks is not None:
            pred_global_masks = pred_global_masks[ind_per_im]
            mask_h, mask_w = pred_global_masks.size()[-2:]
            factor_h = padded_im_h // mask_h
            factor_w = padded_im_w // mask_w
            assert factor_h == factor_w
            factor = factor_h
            pred_global_masks = aligned_bilinear(
                pred_global_masks, factor
            )
            pred_global_masks = pred_global_masks[:, :, :resized_im_h, :resized_im_w]
            pred_global_masks = F.interpolate(
                pred_global_masks,
                size=(output_height, output_width),
                mode="bilinear", align_corners=False
            )
            pred_global_masks = pred_global_masks[:, 0, :, :]
            pred_masks = (pred_global_masks > mask_threshold).float()

        return pred_masks, ind_per_im


def build_mask_predictor(predictor_cfg):
    return MASK_PREDICTOR_REGISTRY.build(predictor_cfg)
