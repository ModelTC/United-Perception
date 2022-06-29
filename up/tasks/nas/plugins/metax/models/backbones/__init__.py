from .xmnet import xmnet  # noqa
from .xmnet_search import xmnet_search  # noqa
from .seg_xmnet import seg_xmnet  # noqa
from up.utils.general.registry_factory import MODULE_ZOO_REGISTRY  # noqa

imported_vars = list(globals().items())

for var_name, var in imported_vars:
    if callable(var):
        MODULE_ZOO_REGISTRY.register(var_name, var)
