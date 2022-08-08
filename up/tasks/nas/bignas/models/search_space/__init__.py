
from .base_bignas_searchspace import BignasSearchSpace # noqa

from up.utils.general.registry_factory import MODULE_ZOO_REGISTRY

imported_vars = list(globals().items())

for var_name, var in imported_vars:
    if callable(var):
        MODULE_ZOO_REGISTRY.register(var_name, var)
