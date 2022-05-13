import json
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from up.data.metrics.base_evaluator import Metric, Evaluator
from up.utils.general.log_helper import default_logger as logger
from up.utils.general.registry_factory import EVALUATOR_REGISTRY
from up.data.image_reader import build_image_reader

__all__ = ['ImageNetEvaluator']


@EVALUATOR_REGISTRY.register('imagenet')
class ImageNetEvaluator(Evaluator):
    def __init__(self,
                 topk=[1, 5],
                 bad_case_analyse=False,
                 analysis_json=None,
                 image_reader=None,
                 class_names=[],
                 eval_class_idxs=[]):
        super(ImageNetEvaluator, self).__init__()
        self.topk = topk
        self.bad_case_analyse = bad_case_analyse
        self.analysis_json = analysis_json
        if image_reader is not None:
            self.image_reader = build_image_reader(image_reader)
        else:
            self.image_reader = None
        self.eval_class_idxs = eval_class_idxs
        self.class_names = class_names

    def load_res(self, res_file, res=None):
        res_dict = {}
        if res is not None:
            lines = []
            for device_res in res:
                for items in device_res:
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
            for key in info.keys():
                if key not in res_dict.keys():
                    res_dict[key] = [info[key]]
                else:
                    res_dict[key].append(info[key])
        return res_dict

    def get_pred_correct(self, pred, label):
        num = pred.size(0)
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
        metric = Metric(res)
        metric.set_cmp_key(f'top{self.topk[0]}')
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
        labels = labels.t()
        attr = []
        for idx in range(attr_num):
            correct, num = self.get_pred_correct(preds[idx], labels[idx])
            temp = {}
            for k in self.topk:
                n_k = min(preds[idx].size(1), k)
                correct_k = correct[:n_k].reshape(-1).float().sum(0, keepdim=True)
                acc = correct_k.mul_(100.0 / num)
                temp[f'top{k}'] = acc.item()
            attr.append(temp)
        for head_num in range(len(attr)):
            res[f'head{head_num}'] = attr[head_num]
        for idx, k in enumerate(self.topk):
            res[f'ave_top{k}'] = 0
            for head_num in range(len(attr)):
                res[f'ave_top{k}'] += res[f'head{head_num}'][f'top{k}']
            res[f'ave_top{k}'] /= len(attr)
        metric = Metric(res)
        metric.set_cmp_key(f'ave_top{self.topk[0]}')
        return metric

    def eval(self, res_file, res=None):
        res_dict = self.load_res(res_file, res)
        label = torch.from_numpy(np.array(res_dict['label']))
        if len(label.shape) > 1:
            preds = []
            for attr_idx in range(label.shape[1]):
                temp = []
                for b_idx in range(label.shape[0]):
                    temp.append(res_dict['topk_idx'][b_idx][attr_idx])
                preds.append(temp)
            pred = [torch.from_numpy(np.array(idx)) for idx in preds]
            metric = self.multicls_eval(pred, label)
        else:
            topk_idx = torch.from_numpy(np.array(res_dict['topk_idx']))
            metric = self.single_eval(topk_idx, label, res_dict)
        return metric

    def analyze_bad_case(self, original_data):
        pred = torch.from_numpy(np.array(original_data['score']))
        analyze_res = {}
        for k in self.topk:
            topk_scores, topk_pred = pred.topk(k, 1, True, True)
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
