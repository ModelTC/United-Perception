import torch
from up.utils.general.registry_factory import ROI_PREDICTOR_REGISTRY
from up.tasks.det_3d.models.utils.model_nms_utils import class_agnostic_nms, multi_classes_nms

__all__ = ['AxisAlignedPredictor']


@ROI_PREDICTOR_REGISTRY.register('axis_aligned')
class AxisAlignedPredictor(object):
    def __init__(self, recall_thresh_list, score_thresh, output_raw_score, cfg):
        self.recall_thresh_list = recall_thresh_list
        self.score_thresh = score_thresh
        self.output_raw_score = output_raw_score
        self.nms_cfg = cfg

    def predict(self, batch_dict, batch_preds):
        batch_cls_preds = batch_preds['batch_cls_preds']
        batch_box_preds = batch_preds['batch_box_preds']
        cls_preds_normalized = batch_preds['cls_preds_normalized']

        pred_list = self.get_predict_list(
            batch_dict, batch_cls_preds, batch_box_preds, cls_preds_normalized)
        return pred_list

    def get_predict_list(self, batch_dict, batch_cls_preds, batch_box_preds, cls_preds_normalized):
        """
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                                or [(B, num_boxes, num_class1), (B, num_boxes, num_class2) ...]
                multihead_label_mapping: [(num_class1), (num_class2), ...]
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
                has_class_labels: True/False
                roi_labels: (B, num_rois)  1 .. num_classes
                batch_pred_labels: (B, num_boxes, 1)
        Returns:

        """

        self.num_classes = len(batch_dict['class_names'])
        batch_size = batch_dict['batch_size']
        pred_list = []
        for index in range(batch_size):
            if batch_dict.get('batch_index', None) is not None:
                assert batch_box_preds.shape.__len__() == 2
                batch_mask = (batch_dict['batch_index'] == index)
            else:
                assert batch_box_preds.shape.__len__() == 3
                batch_mask = index
            box_preds = batch_box_preds[batch_mask]

            if not isinstance(batch_cls_preds, list):
                cls_preds = batch_cls_preds[batch_mask]
                src_cls_preds = cls_preds
                assert cls_preds.shape[1] in [1, self.num_classes]

                if not cls_preds_normalized:
                    cls_preds = torch.sigmoid(cls_preds)
            else:
                cls_preds = [x[batch_mask] for x in batch_cls_preds]
                src_cls_preds = cls_preds
                if not cls_preds_normalized:
                    cls_preds = [torch.sigmoid(x) for x in cls_preds]

            if self.nms_cfg['multi_classes_nms']:
                if not isinstance(cls_preds, list):
                    cls_preds = [cls_preds]
                    multihead_label_mapping = [torch.arange(1, self.num_classes, device=cls_preds[0].device)]
                else:
                    multihead_label_mapping = batch_dict['multihead_label_mapping']

                cur_start_idx = 0
                pred_scores, pred_labels, pred_boxes = [], [], []
                for cur_cls_preds, cur_label_mapping in zip(cls_preds, multihead_label_mapping):
                    assert cur_cls_preds.shape[1] == len(cur_label_mapping)
                    cur_box_preds = box_preds[cur_start_idx: cur_start_idx + cur_cls_preds.shape[0]]
                    cur_pred_scores, cur_pred_labels, cur_pred_boxes = multi_classes_nms(
                        cls_scores=cur_cls_preds, box_preds=cur_box_preds,
                        nms_config=self.nms_cfg,
                        score_thresh=self.score_thresh
                    )
                    cur_pred_labels = cur_label_mapping[cur_pred_labels]
                    pred_scores.append(cur_pred_scores)
                    pred_labels.append(cur_pred_labels)
                    pred_boxes.append(cur_pred_boxes)
                    cur_start_idx += cur_cls_preds.shape[0]

                final_scores = torch.cat(pred_scores, dim=0)
                final_labels = torch.cat(pred_labels, dim=0)
                final_boxes = torch.cat(pred_boxes, dim=0)
            else:
                cls_preds, label_preds = torch.max(cls_preds, dim=-1)
                if batch_dict.get('has_class_labels', False):
                    label_key = 'roi_labels' if 'roi_labels' in batch_dict else 'batch_pred_labels'
                    label_preds = batch_dict[label_key][index]
                else:
                    label_preds = label_preds + 1

                selected, selected_scores = class_agnostic_nms(
                    box_scores=cls_preds, box_preds=box_preds,
                    nms_config=self.nms_cfg,
                    score_thresh=self.score_thresh
                )

                if self.output_raw_score:
                    max_cls_preds, _ = torch.max(src_cls_preds, dim=-1)
                    selected_scores = max_cls_preds[selected]

                final_scores = selected_scores
                final_labels = label_preds[selected]
                final_boxes = box_preds[selected]

            record_dict = {
                'pred_boxes': final_boxes,
                'pred_scores': final_scores,
                'pred_labels': final_labels
            }
            pred_list.append(record_dict)

        return pred_list
