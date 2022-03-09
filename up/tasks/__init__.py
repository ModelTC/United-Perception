import importlib
import os


def is_package(dirname):
    return os.path.exists(os.path.join(dirname, '__init__.py'))


pwd = os.path.dirname(os.path.realpath(__file__))
tasks_names = os.environ.get("DEFAULT_TASKS", os.listdir(pwd))
if ':' in tasks_names:
    tasks_names = tasks_names.split(':')
for fp in tasks_names:
    realpath = os.path.join(pwd, fp)
    if os.path.isdir(realpath) and is_package(realpath):
        globals()[fp] = importlib.import_module('.' + fp, __package__)
