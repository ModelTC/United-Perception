import builtins
import contextlib


@contextlib.contextmanager
def no_print():
    print_ = builtins.print
    builtins.print = lambda *args: args
    yield
    builtins.print = print_
