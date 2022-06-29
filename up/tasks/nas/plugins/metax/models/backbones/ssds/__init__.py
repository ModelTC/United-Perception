from .base_ssd import BaseSSD  # noqa
from .xmnet_ssd import xmnetSSD  # noqa


def ssd_entry(ssd_type):
    return globals()[ssd_type]()
