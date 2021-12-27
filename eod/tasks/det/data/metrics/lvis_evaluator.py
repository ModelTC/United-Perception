"""
Wrapper of evaluation of pycocotools. Support
    1. bbox, proposal, person_bbox
    2. keypoints
    3. segm
"""

from __future__ import division

# Standard Library
import builtins
import json
import os

# Import from third library
import numpy as np
from lvis import LVIS, LVISEval

# Import from eod
from eod.utils.general.log_helper import default_logger as logger
from eod.utils.general.registry_factory import EVALUATOR_REGISTRY

# Import from local
from eod.data.metrics.base_evaluator import Evaluator, Metric

# fix pycocotools py2-style bug
builtins.unicode = str


def _check_cls_ap(self, summary_type, iou_thr=None, area_rng="all", freq_group_idx=None):
    aidx = [
        idx
        for idx, _area_rng in enumerate(self.params.area_rng_lbl)
        if _area_rng == area_rng
    ]

    if summary_type == 'ap':
        s = self.eval["precision"]
        if iou_thr is not None:
            tidx = np.where(iou_thr == self.params.iou_thrs)[0]
            s = s[tidx]
        s = s[:, :, :, aidx]
    else:
        s = self.eval["recall"]
        if iou_thr is not None:
            tidx = np.where(iou_thr == self.params.iou_thrs)[0]
            s = s[tidx]
        s = s[:, :, aidx]
    num_cats = s.shape[-2]
    num_area = s.shape[-1]
    s = s.reshape(-1, num_cats, num_area)
    s = np.transpose(s, (1, 0, 2)).reshape(num_cats, -1)
    mask = s > -1
    num_pos = mask.sum(axis=1)
    pos_inds = np.where(num_pos > 0)[0]
    s = (s * mask).sum(axis=1)
    mean_s = -np.ones(num_cats)
    mean_s[pos_inds] = s[pos_inds] / num_pos[pos_inds]
    return mean_s


@EVALUATOR_REGISTRY.register('LVIS')
class LvisEvaluator(Evaluator):
    """Evaluator for coco"""

    def __init__(self, gt_file, iou_types=['bbox']):
        """
        Arguments:
            gt_file (str): directory or json file of annotations
            iou_types (str): list of iou types of [keypoints, bbox, segm]
        """
        super(LvisEvaluator, self).__init__()
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
        self.gts = LVIS(self.gt_file)
        dts = self.load_dts(res_file, res)

        metrics = Metric({})
        for itype, class_agnostic in zip(self.iou_types, self.use_cats):
            lvis_eval = LVISEval(self.gts, dts, itype)
            LVISEval._check_cls_ap = _check_cls_ap
            if not class_agnostic:
                lvis_eval.params.useCats = 0
                lvis_eval.params.maxDets = [1, 100, 1000]
            lvis_eval.evaluate()
            lvis_eval.accumulate()
            lvis_eval.summarize()
            lvis_eval.print_results()
            logger.info('eval {} results: {}'.format(itype, lvis_eval.results))
            for k, v in lvis_eval.results.items():
                metrics[itype + '.' + k] = v
            # save cls ap result to file
            all_cls_aps = lvis_eval._check_cls_ap('ap')
            all_cls_ars = lvis_eval._check_cls_ap('ar')
            with open('all_cls_{}_aps.log'.format(itype), 'w') as f:
                cats_names = [_['name'] for _ in lvis_eval.lvis_gt.cats.values()]
                cats_freqs = [_['frequency'] for _ in lvis_eval.lvis_gt.cats.values()]
                for i in range(all_cls_aps.shape[0]):
                    cls_name = cats_names[i]
                    cls_freq = cats_freqs[i]
                    f.write('{} {} {} {:.3f} {:.3f}\n'.format(i + 1, cls_name, cls_freq,
                                                              all_cls_ars[i], all_cls_aps[i]))
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
