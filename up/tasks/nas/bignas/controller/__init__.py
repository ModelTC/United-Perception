from .base_controller import BaseController # noqa

from up.tasks.nas.bignas.utils.registry_factory import CONTROLLER_REGISTRY

imported_vars = list(globals().items())

for var_name, var in imported_vars:
    if callable(var):
        CONTROLLER_REGISTRY.register(var_name, var)
