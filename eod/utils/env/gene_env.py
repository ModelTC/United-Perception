import random
import numpy as np
import uuid
import torch
from torch.utils.collect_env import get_pretty_env_info
from .dist_helper import env
from collections.abc import Mapping
import re
from ..general.log_helper import default_logger as logger


def worker_init_reset_seed(worker_id):
    seed = uuid.uuid4().int % 2**32
    random.seed(seed)
    torch.set_rng_state(torch.manual_seed(seed).get_state())
    np.random.seed(seed)


def get_env_info():
    logger.info("Collecting env info (might take some time)")
    try:
        logger.info("\n" + get_pretty_env_info())
    except Exception as e:
        logger.warning('Failed to collect env info. This is probably caused by pip3 version problem')
        logger.warning(e)


def set_random_seed(seed, rank_init=True):
    if not rank_init:
        random.seed(seed)
        np.random.seed(seed ** 2)
        torch.manual_seed(seed ** 3)
        torch.cuda.manual_seed(seed ** 4)
    else:
        seed_add = env.rank
        random.seed(seed + seed_add)
        np.random.seed(seed + seed_add)
        torch.manual_seed(seed + seed_add)
        torch.cuda.manual_seed(seed + seed_add)


def to_device(input, device="cuda", dtype=None):
    """Transfer data between devidces"""

    if 'image' in input:
        input['image'] = input['image'].to(dtype=dtype)

    def transfer(x):
        if torch.is_tensor(x):
            return x.to(device=device)
        elif isinstance(x, list):
            return [transfer(_) for _ in x]
        elif isinstance(x, Mapping):
            return type(x)({k: transfer(v) for k, v in x.items()})
        else:
            return x

    return {k: transfer(v) for k, v in input.items()}


def patterns_match(patterns, string):
    """
    Args:
        patterns: "prefix" or "r:regex"
        string: string
    """

    def match(pattern, string):
        if pattern.startswith('r:'):  # regex
            pattern = pattern[2:]
            if len(re.findall(pattern, string)) > 0:
                return True
        elif string.startswith(pattern):  # prefix
            return True
        return False

    if not isinstance(patterns, (tuple, list)):
        patterns = [patterns]

    if string == "":
        return False

    for pattern in patterns:
        if match(pattern, string):
            return True

    return False


def _bold(s):
    return "\033[33;1m%s\033[0m" % s


def _describe(model, lines=None, spaces=0):
    head = " " * spaces
    for name, p in model.named_parameters():
        if '.' in name:
            continue
        if not p.requires_grad:
            name = _bold(name)
        line = "{head}- {name}".format(head=head, name=name)
        lines.append(line)

    for name, m in model.named_children():
        space_num = len(name) + spaces + 1
        if not m.training:
            name = _bold(name)
        line = "{head}.{name} ({type}({extra_repr}))".format(
            head=head,
            name=name,
            type=m.__class__.__name__,
            extra_repr=m.extra_repr())
        lines.append(line)
        _describe(m, lines, space_num)


def print_network(net, name=None):
    """
    Show network, visualize whether parameters/layers need to train or not
    """
    num = 0
    lines = []
    if name is not None:
        lines.append(name)
        num = len(name)
    _describe(net, lines, num)
    return "\n".join(lines)
