from __future__ import division

# Standard Library
import logging
import sys
from collections import defaultdict, deque
import os
import time
from multiprocessing import Value

import numpy as np
import torch

# Import from local
from ..env.dist_helper import env, MASTER_RANK

logs = set()

# LOGGER
BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)
RESET_SEQ = "\033[0m"
COLOR_SEQ = "\033[1;%dm"

COLORS = {
    'WARNING': YELLOW,
    'INFO': WHITE,
    'DEBUG': BLUE,
    'CRITICAL': YELLOW,
    'ERROR': RED
}


def basicConfig(*args, **kwargs):
    return


# To prevent duplicate logs, we mask this baseConfig setting
logging.basicConfig = basicConfig
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('default').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)


class ColoredFormatter(logging.Formatter):
    def __init__(self, msg, use_color=True):
        logging.Formatter.__init__(self, msg)
        self.use_color = use_color

    def format(self, record):
        msg = record.msg
        levelname = record.levelname
        if self.use_color and levelname in COLORS and COLORS[levelname] != WHITE:
            if isinstance(msg, str):
                msg_color = COLOR_SEQ % (30 + COLORS[levelname]) + msg + RESET_SEQ
                record.msg = msg_color
            levelname_color = COLOR_SEQ % (30 + COLORS[levelname]) + levelname + RESET_SEQ
            record.levelname = levelname_color
        return logging.Formatter.format(self, record)


class SmoothedValue(object):
    """
    Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """
    window_size = 20
    skip_first_k = 0
    precision = 4

    def __init__(self):
        self.deque = deque(maxlen=self.window_size)
        self.total = 0.0
        self.count = 0

    def __iadd__(self, value):
        if (self.skip_first_k > 0):
            self.count = 1
            self.total = value
            self.deque.clear()
            self.deque.append(value)
            self.skip_fist_k -= 1
        else:
            self.deque.append(value)
            self.count += 1
            self.total += value
        return self

    def reset_window_size(self, new_window_size):
        valid_deque = list(deque)[-new_window_size:]
        self.deque = deque(valid_deque, maxlen=new_window_size)

    @property
    def median(self):
        return np.median(self.deque)

    @property
    def avg(self):
        return np.mean(self.deque)

    @property
    def global_avg(self):
        return self.total / max(1, self.count)

    def __str__(self):
        """format:
            latest(global_avg)
        """
        if len(self.deque) == 0:
            return '0(0)'
        else:
            lastest = self.deque[-1]
            global_avg = self.global_avg
            return '{1:.{0}f}({2:.{0}f})'.format(self.precision, lastest, global_avg)


class MetricLogger(object):
    def __init__(self, delimiter="\t", cur_iter=0, start_iter=0):
        self.meters = defaultdict(SmoothedValue)  # no instantiation here
        # self.first_iter_flag = True
        self.first_K_iter_flag = True
        self.skip_iter = 100
        self.start_iter = start_iter
        self.delimiter = delimiter
        self.cur_iter = cur_iter

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        return object.__getattr__(self, attr)

    def set_window_size(self, window_size):
        SmoothedValue.window_size = window_size

    def update(self, cur_iter, detail_time={}, **kwargs):
        # As batch time of the first iter is much longer than normal time, we
        # exclude the first iter for more accurate speed statistics. If the window
        # size is 1, batch time and loss of the first iter will display, but will not
        # contribute to the global average data.
        """
        if self.first_iter_flag and self.start_iter + 1 == self.cur_iter:
            self.first_iter_flag = False
            for name, meter in self.meters.items():
                meter.count -= 1
                meter.total -= meter.deque.pop()
        """
        if self.first_K_iter_flag:
            self.cur_iter = cur_iter
            if self.start_iter == 0:
                self.start_iter = cur_iter
            if self.cur_iter <= self.start_iter + self.skip_iter - 1:
                for name, meter in self.meters.items():
                    if len(meter.deque) != 0:
                        meter.count -= 1
                        meter.total -= meter.deque.pop()
            else:
                self.first_K_iter_flag = False

        kwargs.update(detail_time)
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k] += v

    def __str__(self):
        # str_list = ['{}:{}'.format(k, v) for k, v in self.meters.items()]
        batch_time_str = []
        other_time_str = []
        for name, meter in self.meters.items():
            if name.endswith("_time"):
                if name == "batch_time":
                    batch_time_str.append("{}:{:.4f}({:.4f})".format(name, meter.avg, meter.global_avg))
                else:
                    other_time_str.append("{}:{:.4f}({:.4f}) ".format(name, meter.avg, meter.global_avg))
        # return self.delimiter.join(sorted(str_list))
        batch_time_str = "".join(batch_time_str)
        other_time_str = "{" + "".join(other_time_str).strip() + "}"
        return self.delimiter.join([batch_time_str, other_time_str])


meters = MetricLogger(delimiter=" ")
timer = MetricLogger(delimiter=" ")


def init_log(name='global', level=logging.INFO):
    if (name, level) in logs:
        return
    logs.add((name, level))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(level)
    format_str = f'%(asctime)s-rk{MASTER_RANK}-%(filename)s#%(lineno)d:%(message)s'
    logger.addFilter(lambda record: env.is_master())
    formatter = ColoredFormatter(format_str)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.propagate = False
    return logger


default_logger = init_log('global', logging.INFO)

read_time = 0
count = 0
last_time = time.time()
max_pid = Value('i', 0)


def dataset_timer_log(obj, type):
    cls = obj.__class__
    call = cls.__call__
    data_timer = MetricLogger(delimiter=" ")
    dataset_timer_enabled = float(os.environ.get("DATASET_TIMER_ENABLED", 1))
    if 'BATCH_SIZE' not in os.environ or 'DISPLAY_FREQ' not in os.environ:
        return obj
    freq = int(os.environ['BATCH_SIZE']) * int(os.environ['DISPLAY_FREQ'])
    dataset_time_threshold = float(os.environ.get("DATASET_TIMER_THRESHOLD", 30))
    if dataset_timer_enabled == 0:
        return obj

    init = False
    pid = 0

    def wrapper(*args, **kw):
        global last_time, count, read_time
        nonlocal init, pid
        if not init:
            init = True
            pid = os.getpid()
            with max_pid.get_lock():
                max_pid.value = max(max_pid.value, pid)

        start = time.time()
        result = call(*args, **kw)
        end = time.time()

        if type == "read":
            read_time = end - start
        else:
            data_timer.update(count, read_time=read_time, transform_time=end - start)

        if type != "read" and max_pid.value == pid:
            if count % freq == 0:
                default_logger.info(f"Load a image: {data_timer}")
            elif end - last_time > dataset_time_threshold:
                default_logger.info(f"Load a image exceed {dataset_time_threshold}s: {data_timer}")

        if type != "read":
            count += 1
            last_time = end
        return result

    cls.__call__ = wrapper
    return obj
