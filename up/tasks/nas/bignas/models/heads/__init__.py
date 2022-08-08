from up.utils.general.registry_factory import MODULE_ZOO_REGISTRY


from .big_retinanetwithbn import (  # noqa: F401
    BigRetinaHeadWithBN, BignasRetinaHeadWithBN
)

from .big_clshead import (  # noqa: F401
    BigClsHead
)

from .big_roi_head import (  # noqa: F401
    big_roi_head, bignas_roi_head
)

imported_vars = list(globals().items())

for var_name, var in imported_vars:
    if callable(var):
        MODULE_ZOO_REGISTRY.register(var_name, var)
