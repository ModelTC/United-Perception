import builtins
import contextlib
import copy


@contextlib.contextmanager
def no_print():
    print_ = builtins.print
    builtins.print = lambda *args: args
    yield
    builtins.print = print_


@contextlib.contextmanager
def config(cfg, phase, allow_empty=True):
    if phase not in cfg:
        yield cfg
    else:
        tmp_cfg = copy.deepcopy(cfg)
        tmp_cfg.update(tmp_cfg.pop(phase))
        yield tmp_cfg
