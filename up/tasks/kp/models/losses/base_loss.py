import torch
from up.utils.general.registry_factory import LOSSES_REGISTRY


LOSSES_REGISTRY.register('kp_bce', torch.nn.BCEWithLogitsLoss)
LOSSES_REGISTRY.register('kp_mse', torch.nn.MSELoss)
