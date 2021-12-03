import torch.nn as nn
from eod.utils.general.registry_factory import MODULE_ZOO_REGISTRY
from eod.utils.model.initializer import init_weights_msra

__all__ = ['BaseClsHead']


@MODULE_ZOO_REGISTRY.register('base_cls_head')
class BaseClsHead(nn.Module):
    def __init__(self, num_classes, in_plane, input_feature_idx=-1):
        super(BaseClsHead, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.num_classes = num_classes
        self.in_plane = in_plane
        self.input_feature_idx = input_feature_idx
        self.classifier = nn.Linear(self.in_plane, self.num_classes)

        self.prefix = self.__class__.__name__
        self._init_weights()

    def _init_weights(self):
        init_weights_msra(self.classifier)

    def forward_net(self, x):
        if isinstance(x, dict):
            x = x['features'][self.input_feature_idx]
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)
        return {'logits': logits}

    def forward(self, input):
        return self.forward_net(input)
