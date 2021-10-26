# Standard Library
from collections import OrderedDict
from numbers import Real
import abc
import math
from eod.utils.general.registry_factory import EVALUATOR_REGISTRY


__all__ = ['Metric', 'Evaluator']


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
