import importlib
import os


def is_package(dirname):
    return os.path.exists(os.path.join(dirname, '__init__.py'))


pwd = os.path.dirname(os.path.realpath(__file__))
tasks_names = os.environ.get("DEFAULT_TASKS", os.listdir(pwd))
exclude_tasks = os.environ.get("EXCLUDE_TASKS", '').split(":")
if not isinstance(exclude_tasks, list):
    exclude_tasks = [exclude_tasks]
exclude_tasks = set(exclude_tasks)
if ':' in tasks_names:
    tasks_names = tasks_names.split(':')
if not isinstance(tasks_names, list):
    tasks_names = [tasks_names]
for fp in tasks_names:
    if fp in exclude_tasks:
        continue
    realpath = os.path.join(pwd, fp)
    if os.path.isdir(realpath) and is_package(realpath):
        globals()[fp] = importlib.import_module('.' + fp, __package__)
