import json
import torch
import numpy as np

from up.data.metrics.base_evaluator import Metric, Evaluator
from up.utils.general.registry_factory import EVALUATOR_REGISTRY


__all__ = ['ImageNetEvaluator']


@EVALUATOR_REGISTRY.register('imagenet')
class ImageNetEvaluator(Evaluator):
    def __init__(self, topk=[1, 5]):
        super(ImageNetEvaluator, self).__init__()
        self.topk = topk

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
        maxk = max(self.topk)
        maxk = min(maxk, pred.size(1))
        _, pred = pred.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(label.reshape(1, -1).expand_as(pred))
        return correct, num

    def single_eval(self, pred, label):
        correct, num = self.get_pred_correct(pred, label)
        res = {}
        for k in self.topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            acc = correct_k.mul_(100.0 / num)
            res.update({f'top{k}': acc.item()})
        metric = Metric(res)
        metric.set_cmp_key(f'top{self.topk[0]}')
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
                    temp.append(res_dict['score'][b_idx][attr_idx])
                preds.append(temp)
            pred = [torch.from_numpy(np.array(score)) for score in preds]
            metric = self.multicls_eval(pred, label)
        else:
            pred = torch.from_numpy(np.array(res_dict['score']))
            metric = self.single_eval(pred, label)
        return metric

    @staticmethod
    def add_subparser(name, subparsers):
        subparser = subparsers.add_parser(name, help='subcommand for Imagenet evaluation')
        subparser.add_argument('--res_file', required=True, help='results file of detection')
        return subparser
