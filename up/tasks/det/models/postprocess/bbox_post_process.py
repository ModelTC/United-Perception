import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from up.utils.model import accuracy as A
from up.tasks.det.models.losses import ohem_loss
from up.models.losses import build_loss
from up.utils.general.registry_factory import MODULE_PROCESS_REGISTRY
from up.tasks.det.models.postprocess.bbox_predictor import build_bbox_predictor

__all__ = ['Bbox_PostProcess']


@MODULE_PROCESS_REGISTRY.register('bbox_post')
class Bbox_PostProcess(nn.Module):
    def __init__(self, cfg):
        super(Bbox_PostProcess, self).__init__()

        self.prefix = 'BboxNet'

        self.predictor = build_bbox_predictor(cfg['bbox_predictor'])

        self.cls_loss = build_loss(cfg['cls_loss'])
        self.loc_loss = build_loss(cfg['loc_loss'])

        self.cfg = copy.deepcopy(cfg)

    def forward(self, input):
        if self.training:
            cls_pred = input['bbox_preds'][0]
            loc_pred = input['bbox_preds'][1]
            rois = input['rois']
            recover_inds = input['recover_inds']
            sample_record = input['sample_record']
            cls_target = input['cls_target']
            loc_target = input['loc_target']
            loc_weight = input['loc_weight']

            image_info = input['image_info']
            B = len(image_info)

            cls_pred = cls_pred.float()
            loc_pred = loc_pred.float()

            cls_pred = cls_pred[recover_inds]
            loc_pred = loc_pred[recover_inds]

            cls_inds = cls_target
            if self.cfg.get('share_location', 'False'):
                cls_inds = cls_target.clamp(max=0)

            N = loc_pred.shape[0]
            loc_pred = loc_pred.reshape(N, -1, 4)
            inds = torch.arange(N, dtype=torch.int64, device=loc_pred.device)
            if self.cls_loss.activation_type == 'sigmoid' and not self.cfg.get('share_location', 'False'):
                cls_inds = cls_inds - 1
            loc_pred = loc_pred[inds, cls_inds].reshape(-1, 4)

            reduction_override = None if 'ohem' not in self.cfg else 'none'
            normalizer_override = cls_target.shape[0] if 'ohem' not in self.cfg else None

            loc_loss_key_fields = getattr(self.loc_loss, "key_fields", set())
            loc_loss_kwargs = {}
            if "anchor" in loc_loss_key_fields:
                loc_loss_kwargs["anchor"] = rois
            if "bbox_normalize" in loc_loss_key_fields:
                loc_loss_kwargs["bbox_normalize"] = self.cfg.get('bbox_normalize', None)

            cls_loss = self.cls_loss(cls_pred, cls_target, reduction_override=reduction_override)
            loc_loss = self.loc_loss(loc_pred * loc_weight, loc_target,
                                     reduction_override=reduction_override,
                                     normalizer_override=normalizer_override, **loc_loss_kwargs)

            if self.cfg.get('ohem', None):
                cls_loss, loc_loss = ohem_loss(cls_loss, loc_loss, B, self.cfg['ohem'])

            if self.cls_loss.activation_type == 'softmax':
                acc = A.accuracy(cls_pred, cls_target)[0]
            else:
                try:
                    acc = A.binary_accuracy(cls_pred, self.cls_loss.expand_target)[0]
                except:  # noqa
                    acc = cls_pred.new_zeros(1)
            if self.cfg.get('grid', None):
                loc_loss = loc_loss * 0
            return {
                'sample_record': sample_record,
                self.prefix + '.cls_loss': cls_loss,
                self.prefix + '.loc_loss': loc_loss,
                self.prefix + '.accuracy': acc
            }

        else:
            cls_pred = input['bbox_preds'][0]
            loc_pred = input['bbox_preds'][1]
            rois = input['rois']
            image_info = input['image_info']
            cls_pred = cls_pred.float()
            loc_pred = loc_pred.float()

            if self.cls_loss.activation_type == 'sigmoid':
                cls_pred = torch.sigmoid(cls_pred)
            elif self.cls_loss.activation_type == 'softmax':
                cls_pred = F.softmax(cls_pred, dim=1)
            else:
                cls_pred = self.cls_loss.get_activation(cls_pred)

            start_idx = 0 if self.cls_loss.activation_type == 'sigmoid' else 1
            output = self.predictor.predict(rois, (cls_pred, loc_pred), image_info, start_idx=start_idx)

            return output
