from .resnet import (resnet101,  # noqa
                     resnet152,
                     resnet18,
                     resnet34,
                     resnet50,
                     resnet_custom)
from eod.utils.general.registry_factory import MODULE_ZOO_REGISTRY

imported_vars = list(globals().items())

for var_name, var in imported_vars:
    if callable(var):
        MODULE_ZOO_REGISTRY.register(var_name, var)
