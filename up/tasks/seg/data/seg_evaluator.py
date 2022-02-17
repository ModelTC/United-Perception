import numpy as np
from up.data.metrics.base_evaluator import Metric, Evaluator
from up.utils.general.registry_factory import EVALUATOR_REGISTRY


def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))
    area_output, _ = np.histogram(output, bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


@EVALUATOR_REGISTRY.register('seg')
class SegEvaluator(Evaluator):
    def __init__(self, num_classes=19, ignore_label=255):
        self.num_classes = num_classes
        self.ignore_label = ignore_label

    def load_res(self, res_file, res=None):
        res_dict = {}
        if res is not None:
            lines = []
            for device_res in res:
                for items in device_res:
                    for line in items:
                        lines.append(line)
        else:
            pass
        for line in lines:
            info = line
            for key in info.keys():
                if key not in res_dict.keys():
                    res_dict[key] = [info[key]]
                else:
                    res_dict[key].append(info[key])
        return res_dict

    def eval(self, res_file, res=None):
        res_dict = self.load_res(res_file, res)
        inter_sum = 0.0
        union_sum = 0.0
        target_sum = 0.0
        if 'inter' in res_dict:
            image_num = len(res_dict['inter'])
        else:
            image_num = len(res_dict['pred'])
            preds = res_dict['pred']
            targets = res_dict['gt_semantic_seg']
        for idx in range(image_num):
            if 'inter' not in res_dict:
                inter, union, target = intersectionAndUnion(preds[idx],
                                                            targets[idx],
                                                            self.num_classes,
                                                            self.ignore_label)
            else:
                inter = res_dict['inter'][idx]
                union = res_dict['union'][idx]
                target = res_dict['target'][idx]
            inter_sum += inter
            union_sum += union
            target_sum += target
        miou_cls = inter_sum / (union_sum + 1e-10)
        miou = np.mean(miou_cls)
        acc_cls = inter_sum / (target_sum + 1e-10)
        macc = np.mean(acc_cls)
        res = {}
        res['mIoU'] = miou
        res['mAcc'] = macc
        metric = Metric(res)
        metric.set_cmp_key('mIoU')
        return metric

    @staticmethod
    def add_subparser(name, subparsers):
        subparser = subparsers.add_parser(name, help='subcommand for Seg evaluation')
        subparser.add_argument('--res_file', required=True, help='results file of detection')
        return subparser
