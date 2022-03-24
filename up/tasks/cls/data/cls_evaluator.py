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
        ave_acc = [0. for i in self.topk]
        attr_num = len(preds)
        res = {}
        for idx in range(attr_num):
            correct, num = self.get_pred_correct(preds[idx], labels[idx])
            for k_idx, k in enumerate(self.topk):
                correct_k = correct[:k].reshape(k, -1).float().sum(0)
                acc = correct_k.mul_(100.0)
                ave_acc[k_idx] += acc
        ave_acc_all = [acc.mean() / attr_num for acc in ave_acc]
        ave_acc_all = [acc.numpy().tolist() for acc in ave_acc_all]
        ave_acc = [acc / attr_num for acc in ave_acc]
        ave_acc = [acc.numpy().tolist() for acc in ave_acc]
        for head_num in range(labels.shape[1]):
            res[f'head{head_num}'] = {}
            for k in range(len(self.topk)):
                topk_id = self.topk[k]
                res[f'head{head_num}'].update({f'top{topk_id}': ave_acc[k][head_num]})
        for idx, k in enumerate(self.topk):
            res[f'ave_top{k}'] = ave_acc_all[idx]
        metric = Metric(res)
        metric.set_cmp_key(f'ave_top{self.topk[0]}')
        return metric

    def eval(self, res_file, res=None):
        res_dict = self.load_res(res_file, res)
        pred = torch.from_numpy(np.array(res_dict['score']))
        label = torch.from_numpy(np.array(res_dict['label']))
        if len(label.shape) > 1:
            metric = self.multicls_eval(pred, label)
        else:
            metric = self.single_eval(pred, label)
        return metric

    @staticmethod
    def add_subparser(name, subparsers):
        subparser = subparsers.add_parser(name, help='subcommand for Imagenet evaluation')
        subparser.add_argument('--res_file', required=True, help='results file of detection')
        return subparser
