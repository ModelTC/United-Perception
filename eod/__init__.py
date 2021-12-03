# flake8: noqa F401
import os
import sys
import importlib

"""Set matplotlib up."""
# Import from third library
import matplotlib

__version__ = "0.0.1"  # Available for other modules to import

# import for register

from .commands import *
from .runner import * 
from .data import *
from .models import * 
from .utils import *
from .tasks import *
from .apis import *

matplotlib.use('Agg')  # Use a non-interactive backend


def import_plugin():
    _PPPATH = 'PLUGINPATH'

    if _PPPATH not in os.environ:
        return
    path_list = os.environ[_PPPATH].split(':')
    for path in path_list:
        if path.find('/') >= 0:
            base, module = path.rsplit('/', 1)
            sys.path.insert(0, base)
            importlib.import_module(module)
        else:
            importlib.import_module(path)


# import other auxiliary packages
import_plugin()
