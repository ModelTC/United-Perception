from eod.utils.general.registry_factory import LOSSES_REGISTRY


def build_loss(loss_cfg):
    return LOSSES_REGISTRY.build(loss_cfg)
