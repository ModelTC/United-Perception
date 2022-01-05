import torch.nn as nn
from eod.utils.general.registry_factory import MODULE_ZOO_REGISTRY
from eod.utils.model.initializer import init_weights_msra

__all__ = ['BaseClsHead']


@MODULE_ZOO_REGISTRY.register('base_cls_head')
class BaseClsHead(nn.Module):
    def __init__(self, num_classes, in_plane, input_feature_idx=-1, dropout=None):
        super(BaseClsHead, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.num_classes = num_classes
        self.in_plane = in_plane
        self.input_feature_idx = input_feature_idx
        self.classifier = nn.Linear(self.in_plane, self.num_classes)
        self.drop = nn.Dropout(p=dropout) if dropout else None

        self.prefix = self.__class__.__name__
        self._init_weights()

    def _init_weights(self):
        #init_weights_msra(self.classifier)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 1.0/float(n))
                m.bias.data.zero_()

    def forward_net(self, x):
        if isinstance(x, dict):
            x = x['features'][self.input_feature_idx]
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        if self.drop is not None:
            x = self.drop(x)
        logits = self.classifier(x)
        return {'logits': logits}

    def forward(self, input):
        return self.forward_net(input)

