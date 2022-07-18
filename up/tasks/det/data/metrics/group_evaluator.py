# Standard Library
import json
import copy
from collections import Counter, OrderedDict, defaultdict

# Import from third library
import numpy as np
import pandas
import yaml
from prettytable import PrettyTable

from up.utils.general.log_helper import default_logger as logger
from up.utils.general.yaml_loader import IncludeLoader
from up.utils.general.registry_factory import EVALUATOR_REGISTRY
from up.utils.general.petrel_helper import PetrelHelper
from up.data.metrics.base_evaluator import Metric
from up.tasks.det.data.metrics.custom_evaluator import CustomEvaluator


__all__ = ['GroupRecallEvaluator']


class GroupEvaluator(CustomEvaluator):

    def load_gts(self, gt_files):
        # maintain a dict to store original img information
        # key is image dir,value is image_height,image_width,instances
        original_gt = {}
        gts = {
            'bbox_num': Counter(),
            'gt_num': Counter(),
            'image_num': 0,
            'image_ids': list()
        }
        if not isinstance(gt_files, list):
            gt_files = [gt_files]
        for gt_file_idx, gt_file in enumerate(gt_files):
            gt_img_ids = set()
            with PetrelHelper.open(gt_file) as f:
                for i, line in enumerate(f):
                    img = json.loads(line)
                    if self.label_mapping is not None:
                        img = self.set_label_mapping(img, gt_file_idx)
                    image_id = img['filename']
                    original_gt[img['filename']] = copy.deepcopy(img)
                    gt_img_ids.add(image_id)
                    gts['image_num'] += 1
                    for idx, instance in enumerate(img.get('instances', [])):
                        instance['tps_fns_fppi'] = {}
                        instance['detected_dt_indices_in_all'] = []
                        instance['detected_dt_indices_in_all_fppi'] = {}
                        instance['detected_dt_indices_in_class_sorted'] = []
                        # remember the original index within an image of annoated format so
                        # we can recover from distributed format into original format
                        is_ignore = instance.get('is_ignored', False)
                        instance['index_in_image'] = idx
                        label = instance.get('label', 0)
                        # ingore mode
                        # 0 indicates all classes share ignore region, label is set to -1
                        # 1 indicates different classes different ignore region, ignore label must be provided
                        # 2 indicates we ingore all ignore regions
                        if is_ignore and self.ignore_mode == 0:
                            label = -1
                        box_by_label = gts.setdefault(label, {})
                        box_by_img = box_by_label.setdefault(image_id, {'gts': []})
                        gt_by_img = box_by_img['gts']
                        gts['bbox_num'][label] += 1
                        if not is_ignore:
                            gt_by_img.append(instance)
                            gts['gt_num'][label] += 1
                        else:
                            ign_by_img = box_by_img.setdefault('ignores', [])
                            ign_by_img.append(instance)
                gts['image_ids'].append(gt_img_ids)
        return gts, original_gt

    def reset_detected_flag(self):
        for cls in range(1, self.num_classes):
            if cls in self.gts.keys():
                for img_id, gts in self.gts[cls].items():
                    for instance in gts['gts']:
                        instance['tps_fns_fppi'] = {}
                        instance['detected_dt_indices_in_all'] = []
                        instance['detected_dt_indices_in_all_fppi'] = {}
                        instance['detected_dt_indices_in_class_sorted'] = []

    def load_dts(self, res_file):
        dts = {}
        with open(res_file, 'r') as f:
            for index, line in enumerate(f):
                dt = json.loads(line)
                dt_by_label = dts.setdefault(dt['label'], [])
                dt['index_in_all'] = index
                dt_by_label.append(dt)
        return dts

    def calIof_bbox_mask(self, bbox, mask):
        mask = mask.astype(np.bool)
        bbox = np.maximum(bbox.astype(np.int), 0)
        l, t, r, b = bbox
        # assert r >= l and b >= t, bbox
        if not (r >= l and b >= t):
            l, t = r, b
            print(bbox)
        area1 = (r - l) * (b - t)
        inter_area = np.sum(mask[t: b, l: r])
        return inter_area / np.maximum(area1, 1)

    def get_cls_gt_tp(self, dts_cls, gts_cls, fppi_dt_indices):
        tps = []
        for fppi_index, dt_index in enumerate(fppi_dt_indices):
            _tps = 0
            for img_id, gts in gts_cls.items():
                for gt_group in gts['gts']:
                    dt_bboxes = []
                    indices = []
                    for detected_dt_index in gt_group['detected_dt_indices_in_class_sorted']:
                        if detected_dt_index < dt_index:
                            dt_bboxes.append(dts_cls[detected_dt_index]['bbox'])
                            indices.append(dts_cls[detected_dt_index]['index_in_all'])
                    gt_group['detected_dt_indices_in_all_fppi'][self.fppi[fppi_index]] = indices
                    gt_bboxes = np.array(gt_group['bboxes']).round().astype(np.int)
                    tps_fns_fppi = [False] * len(gt_bboxes)
                    if len(dt_bboxes) > 0:
                        dt_bboxes = np.array(dt_bboxes).round().astype(np.int)
                        dt_bboxes = np.maximum(dt_bboxes, 0)
                        w = np.amax(dt_bboxes[:, 2])
                        h = np.amax(dt_bboxes[:, 3])
                        mask = np.zeros((h, w), dtype=np.bool)
                        for dt_bbox in dt_bboxes:
                            l, t, r, b = dt_bbox
                            mask[t: b, l: r] = 1
                        for gt_bbox_index, gt_bbox in enumerate(gt_bboxes):
                            iof = self.calIof_bbox_mask(gt_bbox, mask)
                            if iof >= self.iou_thresh:
                                tps_fns_fppi[gt_bbox_index] = True
                                _tps += 1
                    gt_group["tps_fns_fppi"][self.fppi[fppi_index]] = tps_fns_fppi
            tps.append(_tps)
        return np.array(tps)

    def get_cls_dt_tp_and_fp(self, dts_cls, gts_cls):
        """
        Arguments:
            dts_cls (list): det results of one specific class.
            gts_cls (dict): ground truthes bboxes of all images for one specific class
        """
        fps, tps = np.zeros((len(dts_cls))), np.zeros((len(dts_cls)))
        for i, dt in enumerate(dts_cls):
            img_id = dt['image_id']
            dt_bbox = np.array(dt['bbox']).round().astype(np.int).reshape(4)
            m_gt_iof, m_gt, m_ign_iof, m_ign = -1, -1, -1, -1
            if img_id in gts_cls:
                gts = gts_cls[img_id]
                igns = gts
                gt_masks = [_['mask'] for _ in gts['gts']]
                ign_masks = [_['mask'] for _ in gts['ignores']] if 'ignores' in gts else []
                m_gt_iof, m_gt = self.match(dt_bbox, gt_masks)
                m_ign_iof, m_ign = self.match(dt_bbox, ign_masks)
            if m_gt_iof >= self.iou_thresh:
                fps[i] = 0
                tps[i] = 1
                dt['match'] = 'tp'
                gts['gts'][m_gt]['detected_dt_indices_in_all'].append(dt['index_in_all'])
                gts['gts'][m_gt]['detected_dt_indices_in_class_sorted'].append(i)
            else:
                fps[i] = 1
                tps[i] = 0
                dt['match'] = 'fp'
            if self.ignore_mode == 0:
                # if we share ignore region between classes, some images may only have ignore region
                # and the m_ign_iof will be set to -1, this is wrong, so we need re calculate m_ign_iof of ignore region

                igns_cls = self.gts.get(-1, {})
                if img_id in igns_cls:
                    igns = igns_cls.get(img_id, {})
                    ign_masks = [_['mask'] for _ in igns['ignores']] if 'ignores' in igns else []
                    m_ign_iof, m_ign = self.match(dt_bbox, ign_masks)
            if self.ignore_mode == 2:
                m_ign_iof = -1
            if fps[i] == 1 and m_ign_iof >= self.ign_iou_thresh:
                fps[i] = 0
                tps[i] = 0
                dt['match'] = 'ig'
                igns['ignores'][m_ign]['detected_dt_indices_in_all'].append(dt['index_in_all'])
                igns['ignores'][m_ign]['detected_dt_indices_in_class_sorted'].append(i)
            dt['m_gt_iof'] = m_gt_iof
            dt['m_gt'] = m_gt
            dt['m_ign_iof'] = m_ign_iof
            dt['m_ign'] = m_ign

        return np.array(tps), np.array(fps)

    def match(self, bbox, mask):
        bbox = np.array(bbox).reshape(4)
        if len(mask) > 0:
            iofs = [self.calIof_bbox_mask(bbox, _) for _ in mask]
            m_index = int(np.argmax(iofs))
            m_iof = iofs[m_index]
        else:
            m_iof = -1
            m_index = -1
        return m_iof, m_index


@EVALUATOR_REGISTRY.register('GroupRecall')
class GroupRecallEvaluator(GroupEvaluator):
    """Calculate Recall@FPPI for dataset with dense objects"""
    def __init__(self,
                 gt_file,
                 num_classes,
                 iou_thresh,
                 class_names=None,
                 fppi=np.array([0.1, 0.5, 1]),
                 metrics_csv='metrics.csv',
                 label_mapping=None,
                 analysis_json='bad_case.jsonl',
                 ignore_mode=0,
                 ign_iou_thresh=0.5,
                 iou_types=['bbox'],
                 eval_best='bbox',
                 cmp_key=0.1,
                 eval_class_idxs=[],
                 cross_cfg=None):

        super(GroupRecallEvaluator, self).__init__(gt_file,
                                                   num_classes,
                                                   iou_thresh,
                                                   metrics_csv=metrics_csv,
                                                   label_mapping=label_mapping,
                                                   ignore_mode=ignore_mode,
                                                   ign_iou_thresh=ign_iou_thresh,
                                                   cross_cfg=cross_cfg)

        if len(eval_class_idxs) == 0:
            eval_class_idxs = list(range(1, num_classes))
        self.eval_class_idxs = eval_class_idxs
        self.analysis_json = analysis_json

        self.fppi = np.array(fppi)
        self.class_names = class_names
        self.metrics_csv = metrics_csv
        assert num_classes == len(class_names)
        self.iou_types = iou_types
        self.eval_best = eval_best
        self.cmp_key = cmp_key

    def get_miss_rate(self, fp, scores, image_num, return_index=False):
        """
        input: accumulated tps & fps
        len(tp) == len(fp) == len(scores) == len(box)
        """
        N = len(self.fppi)
        maxfps = self.fppi * image_num
        fppi_scores = np.zeros(N)

        indices = []
        for i, f in enumerate(maxfps):
            idxs = np.where(fp > f)[0]
            if len(idxs) > 0:
                idx = idxs[0]  # the last fp@fppi
                fppi_scores[i] = scores[idx]
            else:
                idx = len(fp)
                fppi_scores[i] = 0.0
            indices.append(idx)
        if return_index:
            return fppi_scores, indices
        else:
            return fppi_scores

    def export(self, output, max_recall, fppi_scores, recalls_at_fppi):
        fg_class_names = self.class_names
        if self.num_classes == len(self.class_names):
            fg_class_names = fg_class_names[1:]

        assert len(fg_class_names) == self.num_classes - 1

        csv_metrics = OrderedDict()
        csv_metrics['Class'] = fg_class_names
        csv_metrics['Recall'] = max_recall[1:].tolist()
        for idx, fppi in enumerate(self.fppi):
            csv_metrics['Recall@FPPI={:.3f}'.format(fppi)] = recalls_at_fppi[1:, idx].tolist()
        for idx, fppi in enumerate(self.fppi):
            csv_metrics['Score@FPPI={:.3f}'.format(fppi)] = fppi_scores[1:, idx].tolist()

        # Summary
        m_rec = np.mean(max_recall[1:]).tolist()
        m_score_at_fppi = np.mean(fppi_scores[1:], axis=0).tolist()
        m_rec_at_fppi = np.mean(recalls_at_fppi[1:], axis=0).tolist()

        csv_metrics['Class'].append('Mean')
        csv_metrics['Recall'].append(m_rec)
        for fppi, score, recall in zip(self.fppi, m_score_at_fppi, m_rec_at_fppi):
            csv_metrics['Score@FPPI={:.3f}'.format(fppi)].append(score)
            csv_metrics['Recall@FPPI={:.3f}'.format(fppi)].append(recall)

        csv_metrics_table = pandas.DataFrame(csv_metrics)
        csv_metrics_table.to_csv(output, index=False, float_format='%.4f')
        return output, csv_metrics_table, csv_metrics

    def pretty_print(self, metric_table):
        table = PrettyTable()
        table.field_names = list(metric_table.columns)
        table.align = 'l'
        table.float_format = '.4'
        table.title = 'Evaluation Results with FPPI={}'.format(self.fppi)
        for index, row in metric_table.iterrows():
            table.add_row(list(row))
        logger.info('\n{}'.format(table))

    def _expand_bboxes(self, bboxes, ratio):
        bboxes = np.array(bboxes, dtype=np.float32)
        center_x = (bboxes[:, 0] + bboxes[:, 2]) / 2
        center_y = (bboxes[:, 1] + bboxes[:, 3]) / 2
        bboxes[:, 0] = bboxes[:, 0] + (bboxes[:, 0] - center_x) * ratio[0]
        bboxes[:, 1] = bboxes[:, 1] + (bboxes[:, 1] - center_y) * ratio[1]
        bboxes[:, 2] = bboxes[:, 2] + (bboxes[:, 2] - center_x) * ratio[0]
        bboxes[:, 3] = bboxes[:, 3] + (bboxes[:, 3] - center_y) * ratio[1]
        bboxes[:, 0] = np.maximum(0, bboxes[:, 0])
        bboxes[:, 1] = np.maximum(0, bboxes[:, 1])
        return bboxes

    def _connect(self, ious):
        visited = [False] * len(ious)
        groups = []
        for i in range(len(ious)):
            if not visited[i]:
                group = []
                queue = [i]
                visited[i] = True
                while len(queue):
                    index = queue.pop(0)
                    group.append(index)
                    for j in range(len(ious)):
                        if not visited[j] and ious[index, j] > 0:
                            queue.append(j)
                            visited[j] = True
                groups.append(group)
        return groups

    def dump(self, dts):
        image_ids = set()
        for _image_ids in self.gts['image_ids']:
            image_ids = image_ids | _image_ids
        new_dts = defaultdict(list)
        dt_index_map = {}
        for class_i in self.eval_class_idxs:  # range(1, self.num_classes):
            if class_i in dts:
                for dt in dts[class_i]:
                    image_id = dt['image_id']
                    del dt['image_id'], dt['meta_info'], dt['height'], dt['width']
                    dt_index_map[dt['index_in_all']] = len(new_dts[image_id])
                    new_dts[image_id].append(dt)
        results = []
        for image_id in image_ids:
            instances = []
            for class_i in self.eval_class_idxs + [-1]:  # range(1, self.num_classes):
                if class_i in self.gts:
                    gts = self.gts[class_i].get(image_id, {})
                    for instance_tmp in (gts.get('gts', []) + gts.get('ignores', [])):
                        instance = copy.deepcopy(instance_tmp)
                        del instance['mask']
                        del instance["detected_dt_indices_in_class_sorted"]
                        indices = instance["detected_dt_indices_in_all"]
                        instance["detected_dt_indices_in_image"] = [dt_index_map[_] for _ in indices]
                        del instance["detected_dt_indices_in_all"]
                        instance["detected_dt_indices_in_image_fppi"] = {}
                        for fppi, indices in instance["detected_dt_indices_in_all_fppi"].items():
                            instance["detected_dt_indices_in_image_fppi"][fppi] = [dt_index_map[_] for _ in indices]
                        del instance["detected_dt_indices_in_all_fppi"]
                        instance["index_in_image"] = len(instances)
                        instances.append(instance)
            predict_instances = new_dts.get(image_id, [])
            for i in range(len(predict_instances)):
                predict_instances[i]["index_in_image"] = i
            result = {
                "filename": image_id,
                "instances": instances,
                "predict_instances": predict_instances,
            }
            results.append(json.dumps(result, ensure_ascii=False) + '\n')
        with open(self.analysis_json, 'w') as fw:
            fw.writelines(results)

    def get_gt_group_masks(self):
        for class_i in self.eval_class_idxs + [-1]:  # range(1, self.num_classes):
            for filename in self.gts.get(class_i, {}):
                for typename in self.gts[class_i][filename]:
                    instances = self.gts[class_i][filename][typename]
                    if len(instances) > 0:
                        bboxes = np.array([_['bbox'] for _ in instances]).round().astype(np.int)
                        bboxes = np.maximum(bboxes, 0)
                        new_instances = []
                        w = np.amax(bboxes[:, 2])
                        h = np.amax(bboxes[:, 3])
                        bboxes_expand = self._expand_bboxes(bboxes, (0.1, 0.1))
                        ious = self.calIoU(bboxes_expand, bboxes_expand)
                        groups = self._connect(ious)
                        for group_index, group in enumerate(groups):
                            mask = np.zeros((h, w), dtype=np.bool)
                            for instance_index in group:
                                l, t, r, b = bboxes[instance_index]
                                mask[t: b, l: r] = 1
                            is_ignored = [instances[_]['is_ignored'] for _ in group]
                            label = [instances[_]['label'] for _ in group]
                            indices_in_image = [instances[_]['index_in_image'] for _ in group]
                            assert all(is_ignored) == any(is_ignored)
                            if class_i != -1:
                                assert set(label) == {class_i}
                            assert typename in ('ignores', 'gts')
                            new_instances.append(
                                {
                                    'is_ignored': all(is_ignored),
                                    'bboxes': [instances[_]['bbox'] for _ in group],
                                    'mask': mask,
                                    'label': class_i,
                                    'indices_in_image': indices_in_image,
                                    'tps_fns_fppi': {},
                                    'detected_dt_indices_in_all': [],
                                    'detected_dt_indices_in_all_fppi': {},
                                    'detected_dt_indices_in_class_sorted': [],
                                }
                            )
                        self.gts[class_i][filename][typename] = new_instances

    def eval(self, res_file):
        metric_res = Metric({})
        for itype in self.iou_types:
            num_mean_cls = 0
            if not self.gt_loaded:
                self.gts, _ = self.load_gts(self.gt_file)
                self.get_gt_group_masks()
                self.gt_loaded = True
            else:
                self.reset_detected_flag()
            dts = self.load_dts(res_file)
            max_recall = np.zeros(self.num_classes)
            fppi_scores = np.zeros([self.num_classes, len(self.fppi)])
            recalls_at_fppi = np.zeros([self.num_classes, len(self.fppi)])
            for class_i in self.eval_class_idxs:  # range(1, self.num_classes):
                sum_dt = self.gts['bbox_num'][class_i]
                sum_gt = self.gts['gt_num'][class_i]
                results_i = dts.get(class_i, [])
                results_i = sorted(results_i, key=lambda x: -x['score'])
                class_from = self.class_from.get(class_i, [0])
                gts_img_id_set = set()
                for data_idx in class_from:
                    gts_img_id_set = gts_img_id_set.union(self.gts['image_ids'][data_idx])
                if class_i not in self.gts:
                    self.gts[class_i] = {}
                cur_gt = self.get_cur_gt(self.gts[class_i], gts_img_id_set)
                # get the detection results from federated dataset for class_i
                results_i = self.get_cur_dts(results_i, gts_img_id_set)
                logger.info('sum_gt vs sum_dt: {} vs {}'.format(sum_gt, len(results_i)))
                if sum_gt > 0:
                    num_mean_cls += 1
                if sum_dt == 0 or len(results_i) == 0:
                    continue

                _, fps = self.get_cls_dt_tp_and_fp(results_i, cur_gt)
                fp = np.cumsum(fps)
                scores = [_['score'] for _ in results_i]
                image_num = len(gts_img_id_set)
                s_fppi, indices = self.get_miss_rate(fp, scores, image_num, return_index=True)
                gt_tp = self.get_cls_gt_tp(results_i, cur_gt, indices)
                rec = gt_tp / sum_gt
                max_recall[class_i] = np.max(rec)
                fppi_scores[class_i] = s_fppi
                recalls_at_fppi[class_i] = rec

            _, metric_table, csv_metrics = self.export(self.metrics_csv, max_recall, fppi_scores, recalls_at_fppi)
            self.pretty_print(metric_table)
            metric_res.update(csv_metrics)
            if itype == self.eval_best:
                metric_res.set_cmp_key(f"Recall@FPPI={self.cmp_key:.3f}")
            self.dump(dts)

        return metric_res

    @staticmethod
    def add_subparser(name, subparsers):
        subparser = subparsers.add_parser(name, help='subcommand for CustomDataset of mAP metric')
        subparser.add_argument('--gt_file', required=True, help='annotation file')
        subparser.add_argument('--res_file', required=True, help='results file of detection')
        subparser.add_argument('--num_classes',
                               type=int,
                               default=None,
                               help='number of classes including __background__ class')
        subparser.add_argument('--class_names',
                               type=lambda x: x.split(','),
                               default=None,
                               help='names of classes including __background__ class')
        subparser.add_argument('--iou_thresh',
                               type=float,
                               default=0.5,
                               help='iou thresh to classify true positives & false postitives')
        subparser.add_argument('--ign_iou_thresh',
                               type=float,
                               default=0.5,
                               help='ig iou thresh to ignore false postitives')
        subparser.add_argument('--metrics_csv',
                               type=str,
                               default='metrics.csv',
                               help='file to save evaluation metrics')
        subparser.add_argument('--eval_class_idxs',
                               type=lambda x: list(map(int, x.split(','))),
                               default=[],
                               help='eval subset of all classes, 1,3,4'
                               )
        subparser.add_argument('--ignore_mode',
                               type=int,
                               default=0,
                               help='ignore mode, default as 0')
        subparser.add_argument('--config', type=str, default='', help='training config for eval')
        subparser.add_argument('--iou_types',
                               type=list,
                               default=['bbox'],
                               help='iou type to select eval mode')

        return subparser

    @classmethod
    def from_args(cls, args):
        if args.config != '':   # load from training config
            cfg = yaml.load(open(args.config, 'r'), Loader=IncludeLoader)
            eval_kwargs = cfg['dataset']['test']['dataset']['kwargs']['evaluator']['kwargs']
            eval_kwargs['metrics_csv'] = args.metrics_csv
            return cls(**eval_kwargs)
        if args.num_classes is None:
            args.class_names = cls.get_classes(args.gt_file)
            args.num_classes = len(args.class_names)
        return cls(args.gt_file,
                   args.num_classes,
                   args.iou_thresh,
                   args.class_names,
                   metrics_csv=args.metrics_csv,
                   ignore_mode=args.ignore_mode,
                   eval_class_idxs=args.eval_class_idxs,
                   ign_iou_thresh=args.ign_iou_thresh,
                   iou_types=args.iou_types,)
