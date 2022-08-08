# Import from local
from .big_fpn import (  # noqa: F401
    big_fpn, bignas_fpn
)


from up.utils.general.registry_factory import MODULE_ZOO_REGISTRY

imported_vars = list(globals().items())

for var_name, var in imported_vars:
    if callable(var):
        MODULE_ZOO_REGISTRY.register(var_name, var)
