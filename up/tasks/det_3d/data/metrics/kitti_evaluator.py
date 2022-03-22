# Standard Library
import builtins
import json
import copy
import numpy as np

# Import from third library
from up.utils.general.log_helper import default_logger as logger
from up.utils.general.registry_factory import EVALUATOR_REGISTRY
from up.data.metrics.base_evaluator import Evaluator
from up.utils.general.petrel_helper import PetrelHelper
from up.data.metrics import Metric

# fix pycocotools py2-style bug
builtins.unicode = str

__all__ = ['KittiEvaluator']


@EVALUATOR_REGISTRY.register('kitti')
class KittiEvaluator(Evaluator):
    def __init__(self, gt_file, recall_thresh_list=[0.3, 0.5, 0.7]):
        """
        Arguments:
            gt_file (str): directory or json file of annotations
            iou_types (str): list of iou types of [keypoints, bbox, segm]
        """
        super(KittiEvaluator, self).__init__()
        self.gt_file = gt_file
        self.recall_thresh_list = recall_thresh_list

    def load_dts(self, res_file, res):
        out = []
        if res is not None:
            out = res
        else:
            logger.info(f'loading res from {res_file}')
            out = []
            with PetrelHelper.open(res_file) as f:
                for line in f:
                    out.append(json.loads(line))
        out = [x for res_gpus in out for res_bs in res_gpus for x in res_bs]
        for idx in range(len(out)):
            for k, v in out[idx].items():
                if isinstance(v, list):
                    out[idx][k] = np.array(v)
        return out

    def get_metric(self, ret):
        metric = {
            'gt_num': 0,
        }
        for cur_thresh in self.recall_thresh_list:
            metric['recall_roi_%s' % str(cur_thresh)] = 0
            metric['recall_rcnn_%s' % str(cur_thresh)] = 0

        for i in range(len(ret)):
            recall_dict = ret[i]['recall_dict']
            for cur_thresh in self.recall_thresh_list:
                metric['recall_roi_%s' % str(cur_thresh)] += recall_dict.get('roi_%s' % str(cur_thresh), 0)
                metric['recall_rcnn_%s' % str(cur_thresh)] += recall_dict.get('rcnn_%s' % str(cur_thresh), 0)
            metric['gt_num'] += recall_dict.get('gt', 0)

            disp_dict = {}
            min_thresh = self.recall_thresh_list[0]
            disp_dict['recall_%s' % str(min_thresh)] = \
                '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)],
                                   metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])
        gt_num_cnt = metric['gt_num']
        recall_dict = {}
        for cur_thresh in self.recall_thresh_list:
            cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
            cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
            logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
            logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
            recall_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
            recall_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

        return recall_dict

    def eval(self, res_file, class_names, kitti_infos, res, **kwargs):
        if 'annos' not in kitti_infos[0].keys():
            return None, Metric
        det_annos = self.load_dts(res_file, res)
        from .kitti_object_eval_python import eval as kitti_eval
        eval_det_annos = copy.deepcopy(det_annos)
        eval_det_annos = sorted(eval_det_annos, key=lambda e: e['frame_id'])
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in sorted(kitti_infos,
                                                                         key=lambda e:e['point_cloud']['lidar_idx'])]
        recall_dict = self.get_metric(eval_det_annos)
        result, recall_dict = kitti_eval.get_official_eval_result(eval_gt_annos, eval_det_annos, class_names)
        ave_recall = []
        for k, v in recall_dict.items():
            if "3d/moderate" in k:
                ave_recall.append(v)
        recall_dict["AVE_3d/moderate"] = np.array(ave_recall).mean()
        recall_dict = Metric(recall_dict)
        recall_dict.set_cmp_key("AVE_3d/moderate")
        logger.info(json.dumps(result, indent=2))
        return recall_dict

    @staticmethod
    def add_subparser(name, subparsers):
        subparser = subparsers.add_parser(
            name,
            help='subcommand for kitty evaluation')
        subparser.add_argument(
            '--anno_dir',
            required=True,
            help='directory holding kitty annotations')

        subparser.add_argument(
            '--res_file',
            required=True,
            help='file with each line of a result in json string format')
        return subparser
