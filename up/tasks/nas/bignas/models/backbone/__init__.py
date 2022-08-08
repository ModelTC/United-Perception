from .big_resnet_basic import (  # noqa: F401
     big_resnet_basic, bignas_resnet_basic
)
from .big_regnet import (  # noqa
     big_regnet, bignas_regnet
)

from up.utils.general.registry_factory import MODULE_ZOO_REGISTRY

imported_vars = list(globals().items())

for var_name, var in imported_vars:
    if callable(var):
        MODULE_ZOO_REGISTRY.register(var_name, var)
