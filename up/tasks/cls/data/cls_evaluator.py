import json
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from up.data.metrics.base_evaluator import Metric, Evaluator
from up.utils.general.log_helper import default_logger as logger
from up.utils.general.registry_factory import EVALUATOR_REGISTRY
from up.data.image_reader import build_image_reader
from up.utils.model.accuracy import multilabel_accuracy, precision, recall, f1

__all__ = ['ImageNetEvaluator']


@EVALUATOR_REGISTRY.register('imagenet')
class ImageNetEvaluator(Evaluator):
    def __init__(self,
                 topk=[1, 5],
                 bad_case_analyse=False,
                 analysis_json=None,
                 image_reader=None,
                 class_names=[],
                 eval_class_idxs=[],
                 multilabel=False,
                 cmp_key=None,
                 use_prec_rec_f1=False,
                 prec_rec_f1_avg=None):
        super(ImageNetEvaluator, self).__init__()
        self.topk = topk
        self.multilabel = multilabel
        self.bad_case_analyse = bad_case_analyse
        self.analysis_json = analysis_json
        if image_reader is not None:
            self.image_reader = build_image_reader(image_reader)
        else:
            self.image_reader = None
        self.eval_class_idxs = eval_class_idxs
        self.class_names = class_names
        self.cmp_key = cmp_key
        self.use_prec_rec_f1 = use_prec_rec_f1
        self.prec_rec_f1_avg = prec_rec_f1_avg

    def load_res(self, res_file, res=None):
        res_dict = {}
        if res is not None:
            lines = []
            for device_res in res:
                for items in device_res:
                    if self.multilabel:
                        lines.append(items)
                    else:
                        for line in items:
                            lines.append(line)
        else:
            with open(res_file) as f:
                lines = f.readlines()
        for line in lines:
            if res is None:
                info = json.loads(line)
            else:
                info = line
            if self.multilabel:
                for attr_idx in range(len(info)):
                    for key in info[attr_idx].keys():
                        if key not in res_dict.keys():
                            res_dict[key] = [[] for _ in range(len(info))]
                        res_dict[key][attr_idx].append(info[attr_idx][key])
            else:
                for key in info.keys():
                    if key not in res_dict.keys():
                        res_dict[key] = [info[key]]
                    else:
                        res_dict[key].append(info[key])
        return res_dict

    def get_pred_correct(self, pred, label):
        num = pred.size(0)
        if isinstance(pred, torch.FloatTensor) or isinstance(pred, torch.DoubleTensor):
            maxk = max(self.topk)
            maxk = min(maxk, pred.size(1))
            _, pred = pred.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(label.reshape(1, -1).expand_as(pred))
        return correct, num

    def single_eval(self, pred, label, res_dict):
        correct, num = self.get_pred_correct(pred, label)
        res = {}
        for k in self.topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            acc = correct_k.mul_(100.0 / num)
            res.update({f'top{k}': acc.item()})
        if self.use_prec_rec_f1:
            if self.prec_rec_f1_avg is None:
                res.update({'precision': precision(pred, label, self.prec_rec_f1_avg)[0][0].tolist()})
                res.update({'recall': recall(pred, label, self.prec_rec_f1_avg)[0][0].tolist()})
                res.update({'f1': f1(pred, label, self.prec_rec_f1_avg)[0][0].tolist()})
            else:
                res.update({'precision': precision(pred, label, self.prec_rec_f1_avg)[0].item()})
                res.update({'recall': recall(pred, label, self.prec_rec_f1_avg)[0].item()})
                res.update({'f1': f1(pred, label, self.prec_rec_f1_avg)[0].item()})
        metric = Metric(res)
        metric_name = self.cmp_key if self.cmp_key and metric.get(self.cmp_key, False) else f'top{self.topk[0]}'
        metric.set_cmp_key(metric_name)
        if self.bad_case_analyse:
            self.analyze_bad_case(res_dict)
            assert self.analysis_json is not None
            self.export()
            assert self.image_reader is not None
            self.visualize_bad_case()
        return metric

    def multicls_eval(self, preds, labels):
        attr_num = len(preds)
        res = {}
        if not self.multilabel:
            labels = labels.t()
        attr = []
        for idx in range(attr_num):
            if self.multilabel:
                temp = {}
                temp['acc'] = multilabel_accuracy(preds[idx], labels[idx])[0].item()
            else:
                correct, num = self.get_pred_correct(preds[idx], labels[idx])
                temp = {}
                for k in self.topk:
                    n_k = min(preds[idx].size(1), k)
                    correct_k = correct[:n_k].reshape(-1).float().sum(0, keepdim=True)
                    acc = correct_k.mul_(100.0 / num)
                    temp[f'top{k}'] = acc.item()
                if self.use_prec_rec_f1:
                    if self.prec_rec_f1_avg is None:
                        temp['precision'] = precision(preds[idx], labels[idx], self.prec_rec_f1_avg)[0][0].tolist()
                        temp['recall'] = recall(preds[idx], labels[idx], self.prec_rec_f1_avg)[0][0].tolist()
                        temp['f1'] = f1(preds[idx], labels[idx], self.prec_rec_f1_avg)[0][0].tolist()
                    else:
                        temp['precision'] = precision(preds[idx], labels[idx], self.prec_rec_f1_avg)[0].item()
                        temp['recall'] = recall(preds[idx], labels[idx], self.prec_rec_f1_avg)[0].item()
                        temp['f1'] = f1(preds[idx], labels[idx], self.prec_rec_f1_avg)[0].item()

            attr.append(temp)
        for head_num in range(len(attr)):
            res[f'head{head_num}'] = attr[head_num]

        if self.multilabel:
            res['avg_acc'] = 0
            for head_num in range(len(attr)):
                res['avg_acc'] += res[f'head{head_num}']['acc']
            res['avg_acc'] /= len(attr)
            metric = Metric(res)
            metric_name = self.cmp_key if self.cmp_key and metric.get(self.cmp_key, False) else 'avg_acc'
            metric.set_cmp_key(metric_name)
        else:
            for idx, k in enumerate(self.topk):
                res[f'avg_top{k}'] = 0
                for head_num in range(len(attr)):
                    res[f'avg_top{k}'] += res[f'head{head_num}'][f'top{k}']
                res[f'avg_top{k}'] /= len(attr)
                metric = Metric(res)
                if self.cmp_key and metric.get(self.cmp_key, False):
                    metric_name = self.cmp_key
                else:
                    metric_name = f'avg_top{self.topk[0]}'
                metric.set_cmp_key(metric_name)

            if self.use_prec_rec_f1:
                if self.prec_rec_f1_avg:
                    res['avg_precision'], res['avg_recall'], res['avg_f1'] = 0, 0, 0

                    for head_num in range(len(attr)):
                        res['avg_precision'] += res[f'head{head_num}']['precision']
                        res['avg_recall'] += res[f'head{head_num}']['recall']
                        res['avg_f1'] += res[f'head{head_num}']['f1']

                    res['avg_precision'] /= len(attr)
                    res['avg_recall'] /= len(attr)
                    res['avg_f1'] /= len(attr)

                metric.update(res)

        return metric

    def eval(self, res_file, res=None):
        res_dict = self.load_res(res_file, res)
        if self.multilabel:
            label, pred = [], []
            for attr_idx in range(len(res_dict['label'])):
                label.append(np.concatenate(np.array(res_dict['label'][attr_idx]), axis=0))
                pred.append(np.concatenate(np.array(res_dict['prediction'][attr_idx]), axis=0))
            label = torch.from_numpy(np.array(label))
            pred = torch.from_numpy(np.array(pred))
        else:
            label = torch.from_numpy(np.array(res_dict['label']))
        if 'score' in res_dict.keys():
            score = res_dict['score']
        else:
            score = res_dict['topk_idx']
        if len(label.shape) > 1:
            if not self.multilabel:
                preds = []
                for attr_idx in range(label.shape[1]):
                    temp = []
                    for b_idx in range(label.shape[0]):
                        temp.append(score[b_idx][attr_idx])
                    preds.append(temp)
                pred = [torch.from_numpy(np.array(idx)) for idx in preds]
            metric = self.multicls_eval(pred, label)
        else:
            score = torch.from_numpy(np.array(score))
            metric = self.single_eval(score, label, res_dict)
        return metric

    def analyze_bad_case(self, original_data):
        score = torch.from_numpy(np.array(original_data['score']))
        pred = torch.from_numpy(np.array(original_data['topk_idx']))
        analyze_res = {}
        for idx, k in enumerate(self.topk):
            topk_scores = score[:, :idx + 1]
            topk_pred = pred[:, :idx + 1]
            original_data['score'] = topk_scores.numpy().tolist()
            original_data['prediction'] = topk_pred.numpy().tolist()
            convert_res = []
            for i in range(len(original_data['label'])):
                single_res = {}
                single_res['filename'] = original_data['filename'][i]
                single_res['label'] = original_data['label'][i]
                single_res['prediction'] = original_data['prediction'][i]
                single_res['scores'] = original_data['score'][i]
                single_res['bad_case'] = not (single_res['label'] in single_res['prediction'])
                convert_res.append(single_res)
            analyze_res['top' + str(k)] = convert_res
        self.analyze_res = analyze_res

    def export(self):
        for k in self.topk:
            topk_res = self.analyze_res.get('top' + str(k), [])
            with open(self.analysis_json + '_top' + str(k) + '.json', 'w') as fd:
                for res in iter(topk_res):
                    if res['label'] in self.eval_class_idxs:
                        fd.write(json.dumps(res, ensure_ascii=False) + '\n')

    def visualize_bad_case(self, ext='jpg'):
        for k in self.topk:
            topk_res = self.analyze_res.get('top' + str(k), [])
            mix_root = 'vis_cls_bad_case' + '_top' + str(k)
            if not os.path.exists(mix_root):
                os.mkdir(mix_root)
            for res in iter(topk_res):
                label = res.get('label', [])
                prediction = res.get('prediction', [])
                if label not in self.eval_class_idxs:
                    continue
                if res.get('bad_case') is False:
                    continue
                filename = res.get('filename', [])
                score = res.get('scores', [])
                logger.info('visualzing bad case for image {}'.format(filename))
                img = self.image_reader(filename)
                # vis
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.axis('off')
                fig.add_axes(ax)
                ax.imshow(img)
                ax.text(0,
                        10,
                        'label:' + str(self.class_names[label]),
                        fontsize=10,
                        color='black')
                for i in range(len(prediction)):
                    ax.text(0,
                            10 + (i + 1) * 10,
                            'prediction:' + str(self.class_names[prediction[i]]) + ' ' + '%.3f' % score[i],
                            fontsize=10,
                            color='black')
                im_name = filename.split('.')[0]
                output_name = os.path.basename(im_name) + '.' + ext
                plt.savefig(os.path.join(mix_root, '{}'.format(output_name)))
                plt.close('all')

    @staticmethod
    def add_subparser(name, subparsers):
        subparser = subparsers.add_parser(name, help='subcommand for Imagenet evaluation')
        subparser.add_argument('--res_file', required=True, help='results file of detection')
        return subparser


@EVALUATOR_REGISTRY.register('imagenet_lt')
class ImageNetLTEvaluator(ImageNetEvaluator):
    def __init__(self,
                 topk=[1, 5],
                 bad_case_analyse=False,
                 analysis_json=None,
                 image_reader=None,
                 class_names=None,
                 eval_class_idxs=[],
                 cls_freq='',
                 many_shot_thr=100,
                 low_shot_thr=20):
        super(ImageNetLTEvaluator, self).__init__(topk,
                                                  bad_case_analyse,
                                                  analysis_json,
                                                  image_reader,
                                                  class_names,
                                                  eval_class_idxs)
        with open(cls_freq, 'r') as f:
            self.cls_freq = torch.tensor(np.array(json.load(f)))
        self.many_shot_thr = many_shot_thr
        self.low_shot_thr = low_shot_thr

    def eval(self, res_file, res=None):
        res_dict = self.load_res(res_file, res)
        pred = torch.from_numpy(np.array(res_dict['score']))
        label = torch.from_numpy(np.array(res_dict['label']))
        num = pred.size(0)
        maxk = max(self.topk)
        cls_acc = [[] for _ in range(len(self.topk))]
        for il in torch.unique(label):
            for idx, k in enumerate(self.topk):
                _, cls_pred = pred[label == il].topk(k, 1, True, True)
                cls_correct = (cls_pred == il).sum()
                cls_acc[idx].append(cls_correct * 100.0 / cls_pred.shape[0])
        _, pred = pred.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(label.reshape(1, -1).expand_as(pred))
        res = {}
        for k in self.topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            acc = correct_k.mul_(100.0 / num)
            res.update({f'top{k}': acc.item()})

        for idx, k in enumerate(self.topk):
            acc = torch.tensor(cls_acc[idx])
            many_shot = acc[self.cls_freq > self.many_shot_thr]
            low_shot = acc[self.cls_freq < self.low_shot_thr]
            median_shot = acc[(self.cls_freq >= self.low_shot_thr) & (self.cls_freq <= self.many_shot_thr)]
            res.update({f'many_shot_acc_top{k}': many_shot.mean().item(),
                        f'median_shot_acc_top{k}': median_shot.mean().item(),
                        f'low_shot_acc_top{k}': low_shot.mean().item()})
        metric = Metric(res)
        metric.set_cmp_key(f'top{self.topk[0]}')

        return metric
