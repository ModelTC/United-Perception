from __future__ import division

# Standard Library
import builtins
import json
import os

# Import from third library
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from eod.utils.general.log_helper import default_logger as logger
from eod.utils.general.registry_factory import EVALUATOR_REGISTRY
from eod.data.metrics.base_evaluator import Evaluator, Metric

# fix pycocotools py2-style bug
builtins.unicode = str


__all__ = ['CocoEvaluator']


@EVALUATOR_REGISTRY.register('COCO')
class CocoEvaluator(Evaluator):
    """Evaluator for coco"""

    def __init__(self, gt_file, iou_types=['bbox']):
        """
        Arguments:
            gt_file (str): directory or json file of annotations
            iou_types (str): list of iou types of [keypoints, bbox, segm]
        """
        super(CocoEvaluator, self).__init__()
        anno_file, self.iou_types, self.use_cats = self.iou_type_to_setting(iou_types)
        if os.path.isdir(gt_file):
            gt_file = os.path.join(gt_file, anno_file)
        self.gt_file = gt_file
        # self.gt_loaded = False

    def iou_type_to_setting(self, itypes):
        """Get anno_file, real iou types and use-category flags"""
        file_list, type_list, cats_list = [], [], []
        for itype in itypes:
            anno_file, iou_type, use_cats = {
                'keypoints': ('person_keypoints_val2017.json', 'keypoints', True),
                'person_bbox': ('person_keypoints_val2017.json', 'bbox', True),
                'bbox': ('instances_val2017.json', 'bbox', True),
                'segm': ('instances_val2017.json', 'segm', True),
                'proposal': ('instances_val2017.json', 'bbox', False),
                'person_proposal': ('person_keypoints_val2017.json', 'bbox', False)
            }[itype]
            file_list.append(anno_file)
            type_list.append(iou_type)
            cats_list.append(use_cats)
        return file_list[0], type_list, cats_list

    def format(self, aps, iou_type):
        if iou_type.find('keypoints') >= 0:
            metric_name = [
                'AP', 'AP.5', 'AP.75', 'AP_M', 'AP_L', 'AR', 'AR.5', 'AR.75',
                'AR_M', 'AR_L'
            ]
        elif iou_type.find('proposals') >= 0:
            metric_name = [
                'AP_100', 'AP.5_1k', 'AP.75_1k', 'AP_S_1k', 'AP_M_1k', 'AP_L_1k',
                'AR_1', 'AR_100', 'AR_1k', 'AR_S_1k', 'AR_M_1k', 'AR_L_1k'
            ]
        else:
            assert iou_type.find('segm') or iou_type.find('bbox'), iou_type
            metric_name = [
                'AP', 'AP.5', 'AP.75', 'AP_S', 'AP_M', 'AP_L',
                'AR_1', 'AR_10', 'AR', 'AR_S', 'AR_M', 'AR_L'
            ]
        metric_name = ["{}.{}".format(iou_type, item) for item in metric_name]
        assert len(metric_name) == len(aps), f'{len(metric_name)} vs {len(aps)}'
        metric = Metric(zip(metric_name, aps))
        metric.set_cmp_key(metric_name)
        return metric

    def load_dts(self, res_file, res):
        out = []
        if res is not None:
            for device_res in res:
                for lines in device_res:
                    for line in lines:
                        out.append(line)
        else:
            logger.info(f'loading res from {res_file}')
            out = [json.loads(line) for line in open(res_file, 'r')]
        return out

    def eval(self, res_file, res=None):
        """This should return a dict with keys of metric names, values of metric values"""
        # if not self.gt_loaded:
        #     self.gts = COCO(self.gt_file)
        #     self.gt_loaded = True
        self.gts = COCO(self.gt_file)
        dts = self.load_dts(res_file, res)
        dts = self.gts.loadRes(dts)

        metrics = Metric({})
        for itype, class_agnostic in zip(self.iou_types, self.use_cats):
            coco_eval = COCOeval(self.gts, dts, itype)
            if not class_agnostic:
                coco_eval.params.useCats = 0
                coco_eval.params.maxDets = [1, 100, 1000]
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            aps = coco_eval.stats
            COCO_AP = 0
            COCO_AP50 = 1
            COCO_AP75 = 2
            COCO_APS = 3
            COCO_APM = 4
            COCO_APL = 5
            s = coco_eval.stats
            res = {}
            res['AP'] = s[COCO_AP]
            res['AP50'] = s[COCO_AP50]
            res['AP75'] = s[COCO_AP75]
            res['APs'] = s[COCO_APS]
            res['APm'] = s[COCO_APM]
            res['APl'] = s[COCO_APL]
            res_names = res.keys()
            res_vals = ['{:.4f}'.format(v) for v in res.values()]
            print('copypaste: Task {}'.format(itype))
            print('copypaste: ' + ','.join(res_names))
            print('copypaste: ' + ','.join(res_vals))
            logger.info('copypaste: Task {}'.format(itype))
            logger.info('copypaste: ' + ','.join(res_names))
            logger.info('copypaste: ' + ','.join(res_vals))
            metrics.update(self.format(aps, itype))
        metrics.set_cmp_key(list(metrics.keys()))
        return metrics

    @staticmethod
    def add_subparser(name, subparsers):
        subparser = subparsers.add_parser(
            name,
            help='subcommand for COCO evaluation')
        subparser.add_argument(
            '--anno_dir',
            required=True,
            help='directory holding mscoco annotations')
        subparser.add_argument(
            '--iou_types',
            required=True,
            nargs='+',
            choices=['segm', 'bbox', 'keypoints', 'person_bbox', 'proposal', 'person_proposal'],
            help='type of task: segm, bbox or keypoints, specially, person_bbox')
        subparser.add_argument(
            '--res_file',
            required=True,
            help='file with each line of a result in json string format')
        return subparser

    @classmethod
    def from_args(cls, args):
        return cls(args.anno_dir, args.iou_types)
