# Standard Library
import json
import copy
from collections import Counter, OrderedDict

# Import from third library
import numpy as np
import pandas
import yaml
from prettytable import PrettyTable

from eod.utils.general.log_helper import default_logger as logger
from eod.utils.general.yaml_loader import IncludeLoader
from eod.utils.general.registry_factory import EVALUATOR_REGISTRY
from eod.data.metrics.base_evaluator import Evaluator, Metric


__all__ = ['MREvaluator']


class CustomEvaluator(Evaluator):
    def __init__(self, gt_file, num_classes, iou_thresh=0.5, ign_iou_thresh=0.5, metrics_csv='metrics.csv', label_mapping=None, ignore_mode=0): # noqa
        super(CustomEvaluator, self).__init__()
        self.gt_file = gt_file
        self.iou_thresh = iou_thresh
        self.ign_iou_thresh = ign_iou_thresh
        self.num_classes = num_classes
        self.gt_loaded = False
        self.metrics_csv = metrics_csv
        self.label_mapping = label_mapping
        self.class_from = {}
        if self.label_mapping is not None:
            for idx, label_map in enumerate(self.label_mapping):
                for label in label_map:
                    self.class_from[label] = [idx]
        else:
            if not isinstance(gt_file, list):
                gt_file = [gt_file]
            for label in range(1, self.num_classes):
                self.class_from[label] = list(range(len(gt_file)))
        self.ignore_mode = ignore_mode

    def set_label_mapping(self, data, idx):
        if len(self.label_mapping[0]) == 0:
            return data
        instances = data.get('instances', [])
        for instance in instances:
            if instance.get("is_ignored", False):
                if self.ignore_mode == 1:
                    pass
                else:
                    continue
            instance["label"] = self.label_mapping[idx][int(instance["label"] - 1)]
        return data

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
            with open(gt_file, 'r') as f:
                for i, line in enumerate(f):
                    img = json.loads(line)
                    if self.label_mapping is not None:
                        img = self.set_label_mapping(img, gt_file_idx)
                    image_id = img['filename']
                    original_gt[img['filename']] = copy.deepcopy(img)
                    gt_img_ids.add(image_id)
                    gts['image_num'] += 1
                    for idx, instance in enumerate(img.get('instances', [])):
                        instance['detected'] = False
                        # remember the original index within an image of annoated format so
                        # we can recover from distributed format into original format
                        is_ignore = instance.get('is_ignored', False)
                        instance['local_index'] = idx
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
                        instance['detected'] = False
                        instance['detected_score'] = -1

    def load_dts(self, res_file, res=None):
        dts = {}
        if res is None:
            logger.info(f'loading res from {res_file}')
            with open(res_file, 'r') as f:
                for line in f:
                    dt = json.loads(line)
                    dt_by_label = dts.setdefault(dt['label'], [])
                    dt_by_label.append(dt)
        else:
            for device_res in res:
                for lines in device_res:
                    for line in lines:
                        dt_by_label = dts.setdefault(line['label'], [])
                        dt_by_label.append(line)
        return dts

    def calIoU(self, b1, b2):
        area1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
        area2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
        inter_xmin = np.maximum(b1[:, 0].reshape(-1, 1), b2[:, 0].reshape(1, -1))
        inter_ymin = np.maximum(b1[:, 1].reshape(-1, 1), b2[:, 1].reshape(1, -1))
        inter_xmax = np.minimum(b1[:, 2].reshape(-1, 1), b2[:, 2].reshape(1, -1))
        inter_ymax = np.minimum(b1[:, 3].reshape(-1, 1), b2[:, 3].reshape(1, -1))
        inter_h = np.maximum(inter_xmax - inter_xmin, 0)
        inter_w = np.maximum(inter_ymax - inter_ymin, 0)
        inter_area = inter_h * inter_w
        union_area1 = area1.reshape(-1, 1) + area2.reshape(1, -1)
        union_area2 = (union_area1 - inter_area)
        return inter_area / np.maximum(union_area2, 1)

    def calIof(self, b, g):
        area1 = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        inter_xmin = np.maximum(b[:, 0].reshape(-1, 1), g[:, 0].reshape(1, -1))
        inter_ymin = np.maximum(b[:, 1].reshape(-1, 1), g[:, 1].reshape(1, -1))
        inter_xmax = np.minimum(b[:, 2].reshape(-1, 1), g[:, 2].reshape(1, -1))
        inter_ymax = np.minimum(b[:, 3].reshape(-1, 1), g[:, 3].reshape(1, -1))
        inter_h = np.maximum(inter_xmax - inter_xmin, 0)
        inter_w = np.maximum(inter_ymax - inter_ymin, 0)
        inter_area = inter_h * inter_w
        return inter_area / np.maximum(area1, 1)

    def get_cur_dts(self, dts_cls, gts_img_id_set):
        """ Only the detecton results of images on which class c is annotated are kept.
        This is necessary for federated datasets evaluation
        """
        cur_gt_dts = []
        for _, dt in enumerate(dts_cls):
            if dt['image_id'] in gts_img_id_set:
                cur_gt_dts.append(dt)
        return cur_gt_dts

    def get_cur_gt(self, gts_cls, gts_img_id_set):
        """ Only the ground truthes of images on which class c is annotated are kept.
        This is necessary for federated datasets evaluation
        """
        cur_gt = {}
        for img_id, gt in gts_cls.items():
            if img_id in gts_img_id_set:
                cur_gt[img_id] = gt
        return cur_gt

    def get_cls_tp_fp(self, dts_cls, gts_cls):
        """
        Arguments:
            dts_cls (list): det results of one specific class.
            gts_cls (dict): ground truthes bboxes of all images for one specific class
        """
        fps, tps = np.zeros((len(dts_cls))), np.zeros((len(dts_cls)))
        for i, dt in enumerate(dts_cls):
            img_id = dt['image_id']
            dt_bbox = dt['bbox']
            m_iou, m_gt, m_iof = -1, -1, -1
            if img_id in gts_cls:
                gts = gts_cls[img_id]
                gt_bboxes = [g['bbox'] for g in gts['gts']]
                ign_bboxes = [g['bbox'] for g in gts['ignores']] if 'ignores' in gts else []
                m_iou, m_gt, m_iof = \
                    self.match(dt_bbox, gt_bboxes, ign_bboxes)
            if m_iou >= self.iou_thresh:
                if not gts['gts'][m_gt]['detected']:
                    tps[i] = 1
                    fps[i] = 0
                    gts['gts'][m_gt]['detected'] = True
                    gts['gts'][m_gt]['detected_score'] = dt['score']
                else:
                    fps[i] = 1
                    tps[i] = 0
            else:
                fps[i] = 1
                tps[i] = 0
            if self.ignore_mode == 0:
                # if we share ignore region between classes, some images may only have ignore region
                # and the m_iof will be set to -1, this is wrong, so we need re calculate m_iof of ignore region

                igs_cls = self.gts.get(-1, {})
                if img_id in igs_cls:
                    igs = igs_cls.get(img_id, {})
                    ign_bboxes = [g['bbox'] for g in igs['ignores']] if 'ignores' in igs else []
                    m_iof = self.match_ig(dt_bbox, ign_bboxes)
            if self.ignore_mode == 2:
                m_iof = -1
            if fps[i] == 1 and m_iof >= self.ign_iou_thresh:
                fps[i] = 0
                tps[i] = 0

        return np.array(tps), np.array(fps)

    def match_ig(self, dt, igns):
        dt = np.array(dt).reshape(-1, 4)
        if len(igns) > 0:
            igns = np.array(igns)
            iofs = self.calIof(dt, igns).reshape(-1)
            maxiof = np.max(iofs)
        else:
            maxiof = 0
        return maxiof

    def match(self, dt, gts, igns):
        dt = np.array(dt).reshape(-1, 4)
        gts = np.array(gts)
        if len(gts) > 0:
            ious = self.calIoU(dt, gts).reshape(-1)
            matched_gt = ious.argmax()
            matched_iou = ious[matched_gt]
        else:
            matched_iou = -1
            matched_gt = -1
        if len(igns) > 0:
            igns = np.array(igns)
            iofs = self.calIof(dt, igns).reshape(-1)
            maxiof = np.max(iofs)
        else:
            maxiof = 0
        return matched_iou, matched_gt, maxiof

    def get_cls_tp_fp_mask(self, class_i, dts_cls, gts_cls):
        fps, tps = np.zeros((len(dts_cls))), np.zeros((len(dts_cls)))
        for i, dt in enumerate(dts_cls):
            img_id = dt['image_id']
            dt_mask = dt['segmentation']
            m_iou, m_gt = -1, -1
            if img_id in gts_cls:
                gts = gts_cls[img_id]
                gt_masks = [g['mask'] for g in gts['gts']]
                m_iou, m_gt = self.match_mask(dt, dt_mask, gt_masks, gts['gts'])
            if m_iou >= self.iou_thresh:
                if not gts['gts'][m_gt]['detected']:
                    tps[i] = 1
                    fps[i] = 0
                    gts['gts'][m_gt]['detected'] = True
                    gts['gts'][m_gt]['detected_score'] = dt['score']
                else:
                    fps[i] = 1
                    tps[i] = 0
            else:
                fps[i] = 1
                tps[i] = 0
        return np.array(tps), np.array(fps)


@EVALUATOR_REGISTRY.register('MR')
class MREvaluator(CustomEvaluator):
    """Calculate mAP&MR@FPPI for custom dataset"""
    def __init__(self,
                 gt_file,
                 num_classes,
                 iou_thresh,
                 class_names=None,
                 fppi=np.array([0.1, 0.5, 1]),
                 metrics_csv='metrics.csv',
                 label_mapping=None,
                 ignore_mode=0,
                 ign_iou_thresh=0.5,
                 iou_types=['bbox'],
                 eval_class_idxs=[]):

        super(MREvaluator, self).__init__(gt_file,
                                          num_classes,
                                          iou_thresh,
                                          metrics_csv=metrics_csv,
                                          label_mapping=label_mapping,
                                          ignore_mode=ignore_mode,
                                          ign_iou_thresh=ign_iou_thresh)

        if len(eval_class_idxs) == 0:
            eval_class_idxs = list(range(1, num_classes))
        self.eval_class_idxs = eval_class_idxs

        self.fppi = np.array(fppi)
        self.class_names = class_names
        self.metrics_csv = metrics_csv
        if self.class_names is None:
            self.class_names = eval_class_idxs
        self.iou_types = iou_types

    def get_miss_rate(self, tp, fp, scores, image_num, gt_num, return_index=False):
        """
        input: accumulated tps & fps
        len(tp) == len(fp) == len(scores) == len(box)
        """
        N = len(self.fppi)
        maxfps = self.fppi * image_num
        mrs = np.zeros(N)
        fppi_scores = np.zeros(N)

        indices = []
        for i, f in enumerate(maxfps):
            idxs = np.where(fp > f)[0]
            if len(idxs) > 0:
                idx = idxs[0]  # the last fp@fppi
            else:
                idx = -1  # no fps, tp[-1]==gt_num
            indices.append(idx)
            mrs[i] = 1 - tp[idx] / gt_num
            fppi_scores[i] = scores[idx]
        if return_index:
            return mrs, fppi_scores, indices
        else:
            return mrs, fppi_scores

    def _np_fmt(self, x):
        s = "["
        for i in list(x):
            s += " {:.4f}".format(i)
        s += " ]"
        return s

    def export(self, output, ap, max_recall, fppi_miss, fppi_scores,
               recalls_at_fppi, precisions_at_fppi):

        def compute_f1_score(recall, precision):
            return 2 * recall * precision / np.maximum(1, recall + precision)

        f1_score_at_fppi = compute_f1_score(recalls_at_fppi, precisions_at_fppi)

        fg_class_names = self.class_names
        if self.num_classes == len(self.class_names):
            fg_class_names = fg_class_names[1:]

        assert len(fg_class_names) == self.num_classes - 1

        csv_metrics = OrderedDict()
        csv_metrics['Class'] = fg_class_names
        csv_metrics['AP'] = ap[1:].tolist()
        csv_metrics['Recall'] = max_recall[1:].tolist()
        for idx, fppi in enumerate(self.fppi):
            csv_metrics['MR@FPPI={:.3f}'.format(fppi)] = fppi_miss[1:, idx].tolist()
        for idx, fppi in enumerate(self.fppi):
            csv_metrics['Score@FPPI={:.3f}'.format(fppi)] = fppi_scores[1:, idx].tolist()
        for idx, fppi in enumerate(self.fppi):
            csv_metrics['Recall@FPPI={:.3f}'.format(fppi)] = recalls_at_fppi[1:, idx].tolist()
        for idx, fppi in enumerate(self.fppi):
            csv_metrics['Precision@FPPI={:.3f}'.format(fppi)] = precisions_at_fppi[1:, idx].tolist()
        for idx, fppi in enumerate(self.fppi):
            csv_metrics['F1-Score@FPPI={:.3f}'.format(fppi)] = f1_score_at_fppi[1:, idx].tolist()

        # Summary
        mAP = np.mean(ap[1:]).tolist()
        m_rec = np.mean(max_recall[1:]).tolist()
        m_fppi_miss = np.mean(fppi_miss[1:], axis=0).tolist()
        m_score_at_fppi = np.mean(fppi_scores[1:], axis=0).tolist()
        m_rec_at_fppi = np.mean(recalls_at_fppi[1:], axis=0).tolist()
        m_prec_at_fppi = np.mean(precisions_at_fppi[1:], axis=0).tolist()
        m_f1_score_at_fppi = np.mean(f1_score_at_fppi[1:], axis=0).tolist()

        csv_metrics['Class'].append('Mean')
        csv_metrics['AP'].append(mAP)
        csv_metrics['Recall'].append(m_rec)
        for fppi, mr, score, recall, precision, f1_score in zip(
                self.fppi, m_fppi_miss, m_score_at_fppi, m_rec_at_fppi, m_prec_at_fppi, m_f1_score_at_fppi):

            csv_metrics['MR@FPPI={:.3f}'.format(fppi)].append(mr)
            csv_metrics['Score@FPPI={:.3f}'.format(fppi)].append(score)
            csv_metrics['Recall@FPPI={:.3f}'.format(fppi)].append(recall)
            csv_metrics['Precision@FPPI={:.3f}'.format(fppi)].append(precision)
            csv_metrics['F1-Score@FPPI={:.3f}'.format(fppi)].append(f1_score)

        csv_metrics_table = pandas.DataFrame(csv_metrics)
        csv_metrics_table.to_csv(output, index=False, float_format='%.4f')
        return output, csv_metrics_table, csv_metrics

    def pretty_print(self, metric_table):
        columns = list(metric_table.columns)
        for col in columns:
            if col.startswith('Recall@') or col.startswith('Precision@') or col.startswith('F1-Score@'):
                del metric_table[col]

        table = PrettyTable()
        table.field_names = list(metric_table.columns)
        table.align = 'l'
        table.float_format = '.4'
        table.title = 'Evaluation Results with FPPI={}'.format(self.fppi)
        for index, row in metric_table.iterrows():
            table.add_row(list(row))
        logger.info('\n{}'.format(table))

    def eval(self, res_file, res=None):
        metric_res = Metric({})
        for itype in self.iou_types:
            num_mean_cls = 0
            if not self.gt_loaded:
                self.gts, original_gt = self.load_gts(self.gt_file)
                self.gt_loaded = True
            else:
                self.reset_detected_flag()
            dts = self.load_dts(res_file, res)
            ap = np.zeros(self.num_classes)
            max_recall = np.zeros(self.num_classes)
            fppi_miss = np.zeros([self.num_classes, len(self.fppi)])
            fppi_scores = np.zeros([self.num_classes, len(self.fppi)])
            recalls_at_fppi = np.zeros([self.num_classes, len(self.fppi)])
            precisions_at_fppi = np.zeros([self.num_classes, len(self.fppi)])
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
                    # ap[class_i] = 0.0
                    # max_recall[class_i] = 0.0
                    continue

                if itype == 'bbox':
                    tps, fps = self.get_cls_tp_fp(results_i, cur_gt)
                drec = tps / max(1, sum_gt)
                tp = np.cumsum(tps)
                fp = np.cumsum(fps)
                rec = tp / sum_gt
                prec = tp / np.maximum(tp + fp, 1)
                for v in range(len(prec) - 2, -1, -1):
                    prec[v] = max(prec[v], prec[v + 1])
                scores = [x['score'] for x in results_i]
                image_num = len(gts_img_id_set)
                mrs, s_fppi, indices = self.get_miss_rate(tp, fp, scores, image_num, sum_gt, return_index=True)
                ap[class_i] = np.sum(drec * prec)
                max_recall[class_i] = np.max(rec)
                fppi_miss[class_i] = mrs
                fppi_scores[class_i] = s_fppi
                recalls_at_fppi[class_i] = rec[np.array(indices)]
                precisions_at_fppi[class_i] = prec[np.array(indices)]

            mAP = np.sum(ap[1:]) / num_mean_cls

            _, metric_table, csv_metrics = self.export(
                self.metrics_csv, ap, max_recall, fppi_miss, fppi_scores, recalls_at_fppi, precisions_at_fppi)
            self.pretty_print(metric_table)

            metric_name = '{}_mAP:{}'.format(itype, self.iou_thresh)
            csv_metrics.update({metric_name: mAP})
            metric_res.update(csv_metrics)
            metric_res.set_cmp_key(metric_name)
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
        subparser.add_argument('--img_root',
                               type=str,
                               default='None',
                               help='directory of images used to evaluate, only used for visualization')
        subparser.add_argument('--bad_case_analyser',
                               type=str,
                               default='manual_0.5',
                               help='choice of criterion for analysing bad case, format:{manual or fppi}_{[score]}')
        subparser.add_argument('--vis_mode',
                               type=str,
                               default=None,
                               choices=['all', 'fp', 'fn', None],
                               help='visualize fase negatives or fase positive or all')
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
    def get_classes(self, gt_file):
        all_classes = set()

        # change to petrel
        with open(gt_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                labels = set([ins['label'] for ins in data['instances'] if ins['label'] > 0])
                all_classes |= labels
        all_classes = [0] + sorted(list(all_classes))
        class_names = [str(_) for _ in all_classes]
        print('class_names:{}'.format(class_names))
        return class_names

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
                   args.class_names,
                   args.iou_thresh,
                   metrics_csv=args.metrics_csv,
                   ignore_mode=args.ignore_mode,
                   eval_class_idxs=args.eval_class_idxs,
                   ign_iou_thresh=args.ign_iou_thresh,
                   iou_types=args.iou_types,)
