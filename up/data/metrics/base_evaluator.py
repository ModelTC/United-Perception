# Standard Library
from collections import OrderedDict
from numbers import Real
import abc
import math
import copy
import pandas
import json

from up.utils.general.log_helper import default_logger as logger
from up.utils.general.registry_factory import EVALUATOR_REGISTRY


__all__ = ['Metric', 'Evaluator', 'EvaluatorResp']


class NumMetric(Real):
    '''
    This class inherit the Real class to make it comparable. An example implementation for Prototype
    is as follows:

    ::

        class ClsMetric(Metric):
            def __init__(self, top1, top5, loss):
                self.top1 = top1
                self.top5 = top5
                self.loss = loss
                super(ClsMetric, self).__init__(top1)

            def __str__(self):
                return f'top1={self.top1} top5={self.top5} loss={self.loss}'

            def __repr__(self):
                return f'top1={self.top1} top5={self.top5} loss={self.loss}'

    '''
    def __init__(self, v):
        super(NumMetric, self).__init__()
        self.v = v

    @abc.abstractmethod
    def __str__(self):
        pass

    @abc.abstractmethod
    def __repr__(self):
        pass

    # Don't rewrite functions below
    def __float__(self):
        return float(self.v)

    def __abs__(self):
        return abs(self.v)

    def __ceil__(self):
        return math.ceil(self.v)

    def __floor__(self):
        return math.floor(self.v)

    def __mod__(self, other):
        if isinstance(other, Metric):
            return self.v % other.v
        else:
            return self.v % other

    def __mul__(self, other):
        if isinstance(other, Metric):
            return self.v * other.v
        else:
            return self.v * other

    def __neg__(self):
        return -(self.v)

    def __pos__(self):
        return self.v

    def __pow__(self, other):
        if isinstance(other, Metric):
            return math.pow(self.v, other.v)
        else:
            return math.pow(self.v, other)

    def __add__(self, other):
        if isinstance(other, Metric):
            return self.v + other.v
        else:
            return self.v + other

    def __eq__(self, other):
        if isinstance(other, Metric):
            return self.v == other.v
        else:
            return self.v == other

    def __floordiv__(self, other):
        if isinstance(other, Metric):
            return self.v // other.v
        else:
            return self.v // other

    def __le__(self, other):
        if isinstance(other, Metric):
            return self.v <= other.v
        else:
            return self.v <= other

    def __lt__(self, other):
        if isinstance(other, Metric):
            return self.v < other.v
        else:
            return self.v < other

    def __radd__(self, other):
        return self.__add__(other)

    def __rfloordiv__(self, other):
        if isinstance(other, Metric):
            return other.v // self.v
        else:
            return other // self.v

    def __rmod__(self, other):
        if isinstance(other, Metric):
            return other.v % self.v
        else:
            return other % self.v

    def __rmul__(self, other):
        return self.__mul__(other)

    def __round__(self):
        return round(self.v)

    def __rpow__(self, other):
        if isinstance(other, Metric):
            return math.pow(self.v, other.v)
        else:
            return math.pow(self.v, other)

    def __rtruediv__(self, other):
        return 1 / self.__truediv__(other)

    def __truediv__(self, other):
        if isinstance(other, Metric):
            return self.v / other.v
        else:
            return self.v / other

    def __trunc__(self):
        return math.trunc(self.v)


class Evaluator(object):
    def __init__(self):
        pass

    def eval(self, res_file):
        """
        This should return a dict with keys of metric names, values of metric values

        Arguments:
            res_file (str): file that holds detection results
        """
        raise NotImplementedError

    @staticmethod
    def add_subparser(self, name, subparsers):
        raise NotImplementedError

    @staticmethod
    def from_args(cls, args):
        raise NotImplementedError


@EVALUATOR_REGISTRY.register('Resp')
class EvaluatorResp(Evaluator):
    def __init__(self, sub_cfg, gt_file, metrics_csv='metrics.csv'):
        super(EvaluatorResp, self).__init__()
        if not isinstance(gt_file, list):
            gt_file = [gt_file]
        if isinstance(sub_cfg, list):
            sub_cfgs = copy.deepcopy(sub_cfg)
        else:
            sub_cfgs = [copy.deepcopy(sub_cfg) for _ in range(len(gt_file))]
        assert len(sub_cfgs) == len(gt_file), 'Number of evaluators should be equal to gt_files'

        self.evaluators = []
        # build evaluator
        for idx in range(len(sub_cfgs)):
            if 'gt_file' in sub_cfgs[idx].get('kwargs', {}):
                continue
            kwargs = sub_cfgs[idx].setdefault('kwargs', {})
            kwargs.setdefault('gt_file', gt_file[idx])
            self.evaluators.append(build_evaluator(sub_cfgs[idx]))
        self.gt_files = gt_file
        self.metrics_csv = metrics_csv

    def eval(self, res_file, res=None):
        metric_res = Metric({})
        all_dts = self.load_dts(res_file, res)
        # metrics = [self.evaluators[idx](res_file=None, res=all_dts[idx]) for idx in range(len(self.gt_files))]
        metrics = []
        for idx in range(len(self.gt_files)):
            logger.info("-------- Metric_{} --------".format(idx))
            metrics.append(self.evaluators[idx].eval(res_file=None, res=all_dts[idx]))

        # export metrics
        csv_metrics = self.export_all(metrics)
        metric_res.update(csv_metrics)
        for key in metric_res:
            if 'AP' in key:
                metric_res.set_cmp_key(key)
        return metric_res

    def export_all(self, metrics):
        csv_metrics = OrderedDict()
        for idx in range(len(metrics)):
            metric = metrics[idx]
            for key in metric:
                csv_metrics.update({"{}_{}".format(key, idx): metric[key]})
        csv_metrics_table = pandas.DataFrame(csv_metrics)
        csv_metrics_table.to_csv(self.metrics_csv, index=False, float_format='%.4f')
        return csv_metrics

    def load_dts(self, res_file, res=None):
        all_res = None
        if res is None:
            # load dts from res_file
            logger.info(f'loading res from {res_file}')
            all_res = [[[]] for _ in range(len(self.gt_files))]
            with open(res_file, 'r') as f:
                items = [[] for _ in range(len(self.gt_files))]
                for line in f:
                    dt = json.loads(line)
                    image_source = dt['image_source']
                    items[image_source].append(dt)
                for f_ix in range(len(self.gt_files)):
                    all_res[f_ix][0].append(items[f_ix])
        else:
            num_device = len(res)
            all_res = [[[] for _ in range(num_device)] for _ in range(len(self.gt_files))]
            for d_ix in range(num_device):
                res_device = res[d_ix]
                for res_batch in res_device:
                    items = [[] for _ in range(len(self.gt_files))]
                    for res_item in res_batch:
                        items[res_item['image_source']].append(res_item)
                    for f_ix in range(len(self.gt_files)):
                        all_res[f_ix][d_ix].append(items[f_ix])
        return all_res

    @staticmethod
    def add_subparser(name, subparsers):
        subparser = subparsers.add_parser(name, help='subcommand for RespDataset of multi-datasets mAP metric')
        subparser.add_argument('--sub_cfg', required=True, help='sub evaluator config')
        subparser.add_argument('--gt_file', required=True, help='annotation file')
        subparser.add_argument('--metrics_csv',
                               type=str,
                               default='metrics.csv',
                               help='file to save evaluation metrics')
        return subparser

    @staticmethod
    def from_args(cls, args):
        return cls(args.sub_cfg,
                   args.gt_file,
                   metrics_csv=args.metrics_csv)


class Metric(NumMetric, OrderedDict):
    def __init__(self, *args, **kwargs):
        OrderedDict.__init__(self, *args, **kwargs)
        self.cmp_key = kwargs.get('cmp_key', 'AP')
        v = self.get(self.cmp_key, -1)
        NumMetric.__init__(self, v)

    def __str__(self):
        return OrderedDict.__str__(dict(self))

    def __repr__(self):
        return OrderedDict.__str__(dict(self))

    def set_cmp_key(self, key):
        self.cmp_key = key
        if isinstance(key, list):
            self.cmp_key = key[0]
        self.v = self[self.cmp_key]


def build_evaluator(cfg):
    return EVALUATOR_REGISTRY.build(cfg)
