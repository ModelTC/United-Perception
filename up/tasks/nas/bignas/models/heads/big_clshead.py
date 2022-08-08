import torch.nn as nn

from up.utils.model.initializer import trunc_normal_
from up.tasks.nas.bignas.models.ops.dynamic_blocks import DynamicLinearBlock


class BigClsHead(nn.Module):
    def __init__(self, num_classes, in_plane, input_feature_idx=-1, use_pool=True, dropout=0):
        super(BigClsHead, self).__init__()
        self.num_classes = num_classes
        self.in_plane = in_plane
        self.input_feature_idx = input_feature_idx
        self.prefix = self.__class__.__name__
        self.use_pool = use_pool
        self.dropout = dropout
        self.build_classifier(in_plane)
        self._init_weights()
        if self.use_pool:
            self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def build_classifier(self, in_plane):
        if isinstance(self.num_classes, list) or isinstance(self.num_classes, tuple):
            self.classifier = nn.ModuleList()
            for cls in self.num_classes:
                self.classifier.append(DynamicLinearBlock(in_plane, cls, bias=True, dropout_rate=self.dropout))
            self.multicls = True
        else:
            self.multicls = False
            self.classifier = DynamicLinearBlock(in_plane, self.num_classes, bias=True, dropout_rate=self.dropout)

    def get_pool_output(self, x):
        if self.use_pool:
            x = self.pool(x)
            x = x.view(x.size(0), -1)
        return x

    def get_logits(self, x):
        if self.multicls:
            logits = [self.classifier[idx](x) for idx, _ in enumerate(self.num_classes)]
        else:
            logits = self.classifier(x)
        return logits

    def forward_net(self, x):
        x = x['features'][self.input_feature_idx]
        x = self.get_pool_output(x)
        logits = self.get_logits(x)
        return {'logits': logits}

    def forward(self, input):
        return self.forward_net(input)
