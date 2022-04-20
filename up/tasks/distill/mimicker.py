import torch

from .models.utils import mlvl_extract_roi_features, mlvl_extract_gt_masks, match_gts  # noqa
from up.utils.general.log_helper import default_logger as logger
from up.utils.general.registry_factory import MIMIC_REGISTRY, MIMIC_LOSS_REGISTRY


__all__ = ["Mimicker", "Base_Mimicker", "Sample_Feature_Mimicker", "FRS_Mimicker", "DeFeat_Mimicker"]


class Mimicker(object):
    def __init__(self, teacher_model=None, student_model=None, teacher_names=None,
                 teacher_mimic_names=None, student_mimic_names=None,
                 loss_weight=1.0, warm_up_iters=-1, configs=None):
        """
        Args:
            teacher_model: Teacher model.
            student_model: Student model.
        """
        logger.info("name: spring.distiller.mimicker")
        self.teacher_model = teacher_model
        self.student_model = student_model

        assert isinstance(self.teacher_model, list), 'teacher_model must be a list'
        self.teacher_nums = len(self.teacher_model)
        self.teacher_names = teacher_names

        self.loss_weight = loss_weight
        self.warm_up_iters = warm_up_iters
        self.configs = configs

        self.teacher_mimic_names = teacher_mimic_names
        self.student_mimic_names = student_mimic_names

        assert isinstance(self.student_mimic_names, list), 'student mimic name must be a list'
        for tdx in range(self.teacher_nums):
            assert isinstance(self.teacher_mimic_names[tdx], list) and \
                len(self.teacher_mimic_names[tdx]) == len(self.student_mimic_names), 'teacher mimic names error'

        self.s_handles = []
        self.t_handles = [[] for _ in range(self.teacher_nums)]
        self.s_output_maps = {}
        self.t_output_maps = [{} for _ in range(self.teacher_nums)]

        self.build_losses()

    def _register_losses(self, loss_name, default_configs):
        default_type = default_configs['type']
        default_kwargs = default_configs.get('kwargs', {})
        if loss_name in self.configs:
            loss_type = self.configs[loss_name].get('type', default_type)
            if loss_type == default_type:
                default_kwargs.update(self.configs[loss_name].get('kwargs', {}))
                loss_kwargs = default_kwargs
            else:
                loss_kwargs = self.configs[loss_name].get('kwargs', {})
        else:
            loss_type = default_type
            loss_kwargs = default_kwargs
        setattr(self, loss_name, MIMIC_LOSS_REGISTRY[loss_type](**loss_kwargs))

    def build_losses(self):
        """predefined for each mimic method"""
        raise NotImplementedError

    def clear(self):
        """End variables lifetime within mimic() function."""
        self.s_output_maps = {}
        self.t_output_maps = [{} for _ in range(self.teacher_nums)]
        for handle in self.s_handles:
            handle.remove()
        for thandles in self.t_handles:
            for handle in thandles:
                handle.remove()
        self.s_handles = []
        self.t_handles = [[] for _ in range(self.teacher_nums)]

    def _find_module(self, model, layer_name):
        if not layer_name:
            return model

        split_name = layer_name.split('.')
        module = model
        is_found = True
        for i, part_name in enumerate(split_name):
            is_found = False
            for child_name, child_module in module.named_children():
                if part_name == child_name:
                    module = child_module
                    is_found = True
                    break
            if not is_found:
                raise Exception("layer_name {} doesn't exist".format(layer_name))
        return module

    def _register_hooks(self, model, layer_name, output_map, handles):
        def hook(module, input, output):
            output_map[layer_name] = output

        module = self._find_module(model, layer_name)

        if layer_name not in output_map:
            handles.append(module.register_forward_hook(hook))

    def _register_forward_hooks(self):
        for tdx in range(self.teacher_nums):
            for _name in self.teacher_mimic_names[tdx]:
                self._register_hooks(self.teacher_model[tdx], _name, self.t_output_maps[tdx], self.t_handles[tdx])
        for _name in self.student_mimic_names:
            self._register_hooks(self.student_model, _name, self.s_output_maps, self.s_handles)

    def mimic(self, **kwargs):
        """Excute all of the registered mimicjobs.
        Returns:
            mimic_losses: Dict. A list of losses of registered mimic jobs.
        """
        raise NotImplementedError

    def loss_post_process(self, losses, cur_iter):
        for k in losses:
            if 'loss' in k:
                if self.warm_up_iters > cur_iter:
                    losses[k] = self.loss_weight * (cur_iter / self.warm_up_iters) * losses[k]
                else:
                    losses[k] = self.loss_weight * losses[k]
        return losses

    def prepare(self):
        """Pre-work of each forward step."""
        self.clear()
        self._register_forward_hooks()


@MIMIC_REGISTRY.register('base')
class Base_Mimicker(Mimicker):
    def __init__(self, teacher_model=None, student_model=None, teacher_names=None,
                 teacher_mimic_names=None, student_mimic_names=None,
                 loss_weight=1.0, warm_up_iters=-1, configs=None):
        super(Base_Mimicker, self).__init__(teacher_model, student_model, teacher_names,
                                            teacher_mimic_names, student_mimic_names,
                                            loss_weight, warm_up_iters, configs)

    def build_losses(self):
        self._register_losses('loss', {'type': 'l2_loss', 'kwargs': {'feat_norm': True, 'batch_mean': True}})

    def mimic(self, **kwargs):
        s_output = kwargs['s_output']
        t_output = kwargs['t_output']  # noqa
        mimic_losses = {}
        for tdx in range(self.teacher_nums):
            feature_s = [self.s_output_maps[_name] for _name in self.student_mimic_names]
            feature_t = [self.t_output_maps[tdx][_name] for _name in self.teacher_mimic_names[tdx]]

            loss = self.loss(feature_s, feature_t) / len(feature_s)
            mimic_losses.update({self.teacher_names[tdx] + '.loss': loss})
        mimic_losses = self.loss_post_process(mimic_losses, s_output['cur_iter'])
        self.clear()
        return mimic_losses


@MIMIC_REGISTRY.register('SampleFeature')
class Sample_Feature_Mimicker(Mimicker):
    """
    Mimicking Very Efficient Network for Object Detection, CVPR 2017
    https://openaccess.thecvf.com/content_cvpr_2017/papers/Li_Mimicking_Very_Efficient_CVPR_2017_paper.pdf
    """
    def __init__(self, teacher_model=None, student_model=None, teacher_names=None,
                 teacher_mimic_names=None, student_mimic_names=None,
                 loss_weight=1.0, warm_up_iters=-1, configs=None):
        super(Sample_Feature_Mimicker, self).__init__(teacher_model, student_model, teacher_names,
                                                      teacher_mimic_names, student_mimic_names,
                                                      loss_weight, warm_up_iters, configs)

    def build_losses(self):
        self._register_losses('feature_loss', {'type': 'l2_loss', 'kwargs': {'feat_norm': False, 'batch_mean': False}})

    def mimic(self, **kwargs):
        s_output = kwargs['s_output']
        t_output = kwargs['t_output']
        mimic_losses = {}

        rcnn_fpn_levels = self.configs.get('rcnn_fpn_levels', [0, 1, 2, 3])
        rcnn_fpn_strides = self.configs.get('rcnn_fpn_strides', [4, 8, 16, 32])
        rcnn_base_scale = self.configs.get('rcnn_fpn_base_scale', 56)
        teacher_proposal = self.configs.get('teacher_proposal', True)
        student_roipooler = self.configs.get('student_roipooler', 'bbox_head_pre_process.roipool')
        teacher_roipooler = self.configs.get('teacher_roipooler', ['bbox_head_pre_process.roipool'])
        assert len(teacher_roipooler) == self.teacher_nums, 'teacher_roipooler names must match teacher numbers'
        rois = s_output['rpn_dt_bboxes']

        for tdx in range(self.teacher_nums):
            feature_s = [self.s_output_maps[_name] for _name in self.student_mimic_names]
            feature_t = [self.t_output_maps[tdx][_name] for _name in self.teacher_mimic_names[tdx]]
            if teacher_proposal:
                rois = t_output[tdx]['rpn_dt_bboxes']
            with torch.no_grad():
                t_pooled_feats = mlvl_extract_roi_features(rois, feature_t, rcnn_fpn_levels,
                                                           rcnn_fpn_strides, rcnn_base_scale,
                                                           self._find_module(self.teacher_model[tdx], teacher_roipooler[tdx]))  # noqa
            s_pooled_feats = mlvl_extract_roi_features(rois, feature_s, rcnn_fpn_levels,
                                                       rcnn_fpn_strides, rcnn_base_scale,
                                                       self._find_module(self.student_model, student_roipooler))
            feature_loss = self.feature_loss([t_pooled_feats], [s_pooled_feats]) / 2.0
            mimic_losses.update({self.teacher_names[tdx] + '.feature_loss': feature_loss})
        mimic_losses = self.loss_post_process(mimic_losses, s_output['cur_iter'])
        self.clear()
        return mimic_losses


@MIMIC_REGISTRY.register('FRS')
class FRS_Mimicker(Mimicker):
    """
    Distilling Object Detectors with Feature Richness, NIPS 2021
    https://arxiv.org/pdf/2111.00674.pdf
    """
    def __init__(self, teacher_model=None, student_model=None, teacher_names=None,
                 teacher_mimic_names=None, student_mimic_names=None,
                 loss_weight=1.0, warm_up_iters=-1, configs=None):
        super(FRS_Mimicker, self).__init__(teacher_model, student_model, teacher_names,
                                           teacher_mimic_names, student_mimic_names,
                                           loss_weight, warm_up_iters, configs)

    def build_losses(self):
        self._register_losses('neck_loss', {'type': 'l2_loss', 'kwargs': {'feat_norm': False, 'batch_mean': False}})
        self._register_losses('pred_loss', {'type': 'bce_loss', 'kwargs': {'T': 1.0, 'batch_mean': False}})

    def mimic(self, **kwargs):
        s_output = kwargs['s_output']
        t_output = kwargs['t_output']
        mimic_losses = {}

        for tdx in range(self.teacher_nums):
            feature_s = [self.s_output_maps[_name] for _name in self.student_mimic_names]
            feature_t = [self.t_output_maps[tdx][_name] for _name in self.teacher_mimic_names[tdx]]
            if 'rpn_preds' in s_output:
                # two stage
                stu_cls_score = [p[0] for p in s_output['rpn_preds']]
            else:
                # one stage
                stu_cls_score = [p[0] for p in s_output['preds']]
            if 'rpn_preds' in t_output[tdx]:
                tea_cls_score = [p[0] for p in t_output[tdx]['rpn_preds']]
            else:
                tea_cls_score = [p[0] for p in t_output[tdx]['preds']]
            masks = []
            pred_s = []
            pred_t = []
            for ldx in range(len(feature_t)):
                stu_cls_score_sigmoid = stu_cls_score[ldx].sigmoid()
                tea_cls_score_sigmoid = tea_cls_score[ldx].sigmoid()
                mask = torch.max(tea_cls_score_sigmoid, dim=1).values
                mask = mask.detach()
                masks.append(mask[:, None, :, :])
                pred_s.append(stu_cls_score_sigmoid)
                pred_t.append(tea_cls_score_sigmoid)
            neck_loss = self.neck_loss(feature_s, feature_t, masks=masks)
            pred_loss = self.pred_loss(pred_s, pred_t, masks=masks)

            mimic_losses.update({self.teacher_names[tdx] + '.neck_loss': neck_loss,
                                 self.teacher_names[tdx] + '.pred_loss': pred_loss})
        mimic_losses = self.loss_post_process(mimic_losses, s_output['cur_iter'])
        self.clear()
        return mimic_losses


@MIMIC_REGISTRY.register('DeFeat')
class DeFeat_Mimicker(Mimicker):
    """
    Distilling Object Detectors via Decoupled Features, CVPR 2021
    https://arxiv.org/pdf/2103.14475.pdf
    """
    def __init__(self, teacher_model=None, student_model=None, teacher_names=None,
                 teacher_mimic_names=None, student_mimic_names=None,
                 loss_weight=1.0, warm_up_iters=-1, configs=None):
        super(DeFeat_Mimicker, self).__init__(teacher_model, student_model, teacher_names,
                                              teacher_mimic_names, student_mimic_names,
                                              loss_weight, warm_up_iters, configs)

    def build_losses(self):
        self._register_losses('backbone_loss', {'type': 'l2_loss', 'kwargs': {'feat_norm': False, 'batch_mean': False}})
        self._register_losses('neck_fg_loss', {'type': 'l2_loss', 'kwargs': {'feat_norm': False, 'batch_mean': False}})
        self._register_losses('neck_bg_loss', {'type': 'l2_loss', 'kwargs': {'feat_norm': False, 'batch_mean': False}})
#        self._register_losses('head_fg_loss', {'type': 'kd_loss', 'kwargs': {'T': 1.0}})
#        self._register_losses('head_bg_loss', {'type': 'kd_loss', 'kwargs': {'T': 1.0}})

    def mimic(self, **kwargs):
        s_output = kwargs['s_output']
        t_output = kwargs['t_output']       # noqa
        mimic_losses = {}

        rpn_fpn_levels = self.configs.get('rpn_fpn_levels', [0, 1, 2, 3, 4])
        rpn_fpn_strides = self.configs.get('rpn_fpn_strides', [4, 8, 16, 32, 64])
        rpn_base_scale = self.configs.get('rpn_fpn_base_scale', 56)
#        rcnn_fpn_levels = self.configs.get('rcnn_fpn_levels', [0, 1, 2, 3])
#        rcnn_fpn_strides = self.configs.get('rcnn_fpn_strides', [4, 8, 16, 32])
#        rcnn_base_scale = self.configs.get('rcnn_fpn_base_scale', 56)
#        teacher_proposal = self.configs.get('teacher_proposal', True)
#        rois = s_output['rpn_dt_bboxes']

        neck_feature_nums = self.configs.get('neck_feature_nums', 5)
        for tdx in range(self.teacher_nums):
            feature_s = [self.s_output_maps[_name] for _name in self.student_mimic_names]
            feature_t = [self.t_output_maps[tdx][_name] for _name in self.teacher_mimic_names[tdx]]

            bb_nums = len(feature_s) - neck_feature_nums
            student_bb_feat = feature_s[:bb_nums]
            teacher_bb_feat = feature_t[:bb_nums]

            feature_s = feature_s[bb_nums:]
            feature_t = feature_t[bb_nums:]

            # mimic backbone
            backbone_loss = self.backbone_loss(student_bb_feat, teacher_bb_feat) / bb_nums / 2.0

            # mimic neck
            adapt_neck_s = s_output['adapt_neck_features']
            featmap_sizes = [featmap.shape for featmap in adapt_neck_s]
            gt_masks = mlvl_extract_gt_masks(s_output['gt_bboxes'], rpn_fpn_levels,
                                             rpn_fpn_strides, rpn_base_scale, featmap_sizes)
            neck_fg_masks = [m.unsqueeze(1).repeat(1, featmap_sizes[idx][1], 1, 1) for idx, m in enumerate(gt_masks)]
            neck_bg_masks = [1 - m.unsqueeze(1).repeat(1, featmap_sizes[idx][1], 1, 1) for idx, m in enumerate(gt_masks)]   # noqa
            neck_fg_loss = self.neck_fg_loss(adapt_neck_s, feature_t, masks=neck_fg_masks) / neck_feature_nums / 2.0
            neck_bg_loss = self.neck_bg_loss(adapt_neck_s, feature_t, masks=neck_bg_masks) / neck_feature_nums / 2.0

# mimic head is useless and unstable
#            # mimic head
#            if teacher_proposal:
#                rois = t_output[tdx]['rpn_dt_bboxes']
#            labels = match_gts(rois, s_output['gt_bboxes'],
#                               s_output.get('gt_ignores', None), s_output['image_info'],
#                               self.student_model.bbox_head.supervisor.matcher)
#
#            with torch.no_grad():
#                t_pooled_feats, recover_inds = mlvl_extract_roi_features(rois, feature_t,
#                                                                         rcnn_fpn_levels,
#                                                                         rcnn_fpn_strides,
#                                                                         rcnn_base_scale,
#                                                                         self.teacher_model[tdx].bbox_head.roipool,
#                                                                         return_recover_inds=True)
#                t_cls_pred, _ = self.teacher_model[tdx].bbox_head.forward_net(t_pooled_feats)
#                t_cls_pred = t_cls_pred[recover_inds]
#            s_pooled_feats, recover_inds = mlvl_extract_roi_features(rois, feature_s,
#                                                                     rcnn_fpn_levels,
#                                                                     rcnn_fpn_strides,
#                                                                     rcnn_base_scale,
#                                                                     self.student_model.bbox_head.roipool,
#                                                                     return_recover_inds=True)
#            s_cls_pred, _ = self.student_model.bbox_head.forward_net(s_pooled_feats)
#            s_cls_pred = s_cls_pred[recover_inds]
#            head_fg_loss = self.head_fg_loss([s_cls_pred], [t_cls_pred], masks=[labels])
#            head_bg_loss = self.head_bg_loss([s_cls_pred], [t_cls_pred], masks=[(1 - labels)])

            mimic_losses.update({self.teacher_names[tdx] + '.backbone_loss': backbone_loss,
                                 self.teacher_names[tdx] + '.neck_fg_loss': neck_fg_loss,
                                 self.teacher_names[tdx] + '.neck_bg_loss': neck_bg_loss})
        mimic_losses = self.loss_post_process(mimic_losses, s_output['cur_iter'])
        self.clear()
        return mimic_losses
