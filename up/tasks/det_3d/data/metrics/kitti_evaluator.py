# Standard Library
import builtins
import json
import copy

# Import from third library
from up.utils.general.log_helper import default_logger as logger
from up.utils.general.registry_factory import EVALUATOR_REGISTRY
from up.data.metrics.base_evaluator import Evaluator

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
            for device_res in res:
                for lines in device_res:
                    for line in lines:
                        out.append(line)
        else:
            logger.info(f'loading res from {res_file}')
            out = [json.loads(line) for line in open(res_file, 'r')]
        return out

    def eval(self, res_file, class_names, kitti_infos, res, **kwargs):
        if 'annos' not in kitti_infos[0].keys():
            return None, {}
        # det_annos = self.load_dts(res_file, res)
        from .kitti_object_eval_python import eval as kitti_eval
        det_annos = res
        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in kitti_infos]
        result, ret_dict = kitti_eval.get_official_eval_result(eval_gt_annos, eval_det_annos, class_names)
        return result, ret_dict

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
