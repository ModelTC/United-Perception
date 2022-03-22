import torch
import torch.nn as nn
from up.utils.general.registry_factory import MODULE_ZOO_REGISTRY
from up.utils.model.initializer import trunc_normal_
# from up.utils.model.initializer import init_weights_msra

__all__ = ['BaseClsHead', 'ConvNeXtHead', 'ViTHead']


@MODULE_ZOO_REGISTRY.register('base_cls_head')
class BaseClsHead(nn.Module):
    def __init__(self, num_classes, in_plane, input_feature_idx=-1, dropout=None):
        super(BaseClsHead, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.num_classes = num_classes
        self.in_plane = in_plane
        self.input_feature_idx = input_feature_idx
        self.multicls = False
        if isinstance(self.num_classes, list) or isinstance(self.num_classes, tuple):
            self.classifier = nn.ModuleList()
            for cls in self.num_classes:
                self.classifier.append(nn.Linear(self.in_plane, cls))
            self.multicls = True
        else:
            self.classifier = nn.Linear(self.in_plane, self.num_classes)
        self.drop = nn.Dropout(p=dropout) if dropout else None

        self.prefix = self.__class__.__name__
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # n = m.weight.size(1)
                # m.weight.data.normal_(0, 1.0 / float(n))
                # m.bias.data.zero_()
                trunc_normal_(m.weight, std=.02)
                nn.init.constant_(m.bias, 0)

    def forward_net(self, x):
        if isinstance(x, dict):
            x = x['features'][self.input_feature_idx]
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        if self.drop is not None:
            x = self.drop(x)
        if self.multicls:
            logits = [self.classifier[idx](x) for idx, _ in enumerate(self.num_classes)]
        else:
            logits = self.classifier(x)
        return {'logits': logits}

    def forward(self, input):
        return self.forward_net(input)


@MODULE_ZOO_REGISTRY.register('fpn_cls_head')
class FPNClsHead(BaseClsHead):
    def __init__(self, inplanes, feat_planes, num_classes, sum_feature=True, input_feature_idx=-1, dropout=None):

        super().__init__(num_classes, feat_planes, dropout)  # fake

        self.sum_feature = sum_feature

        if isinstance(inplanes, list):
            if self.sum_feature:
                self.classification_inplanes = inplanes[0]
            else:
                self.classification_inplanes = inplanes[self.input_feature_idx]
        else:
            self.classification_inplanes = inplanes

        self.classification_pool = self.pool

        self.relu = nn.ReLU(inplace=True)

        self.classification_fc_embeding = nn.Linear(self.classification_inplanes, feat_planes)

    def forward_classification_feature(self, features):
        if isinstance(features, list):
            features = features[self.input_feature_idx]
        features = self.classification_pool(features)
        features = features.view(features.size(0), -1)
        features = self.relu(self.classification_fc_embeding(features))
        return features

    def forward_classification_sum_feature(self, features):
        out_feature = None
        for feature in features:
            feature = self.classification_pool(feature)
            feature = feature.view(feature.size(0), -1)
            feature = self.relu(self.classification_fc_embeding(feature))
            if out_feature is None:
                out_feature = feature
            else:
                out_feature = out_feature + feature
        return out_feature

    def forward_net(self, x):
        if isinstance(x, dict):
            x = x['features']
        if self.sum_feature:
            feature = self.forward_classification_sum_feature(x)
        else:
            feature = self.forward_classification_feature(x)

        logits = self.classifier(feature)

        return {'logits': logits}


@MODULE_ZOO_REGISTRY.register('convnext_head')
class ConvNeXtHead(nn.Module):
    def __init__(self, num_classes, in_plane, input_feature_idx=-1, head_init_scale=1.):
        super(ConvNeXtHead, self).__init__()
        self.num_classes = num_classes
        self.in_plane = in_plane
        self.input_feature_idx = input_feature_idx
        self.classifier = nn.Linear(in_plane, num_classes)
        self.prefix = self.__class__.__name__
        self.apply(self._init_weights)
        self.classifier.weight.data.mul_(head_init_scale)
        self.classifier.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_net(self, x):
        if isinstance(x, dict):
            x = x['features'][self.input_feature_idx]
        logits = self.classifier(x)
        return {'logits': logits}

    def forward(self, input):
        return self.forward_net(input)


@MODULE_ZOO_REGISTRY.register('vit_head')
class ViTHead(nn.Module):
    def __init__(self, num_classes, in_plane, input_feature_idx=-1, cls_type='token', representation_size=None):
        super(ViTHead, self).__init__()
        self.num_classes = num_classes
        self.in_plane = in_plane
        self.input_feature_idx = input_feature_idx
        self.cls_type = cls_type
        self.representation_size = representation_size
        # if representation_size is not None
        # a pre-logits layer is inserted before classification head
        # it means transformer will be trained from scratch
        # otherwise the model is finetuned on new task
        if representation_size is not None:
            self.pre_logits = nn.Linear(in_plane, representation_size)
            self.tanh = nn.Tanh()
            self.classifier = nn.Linear(representation_size, num_classes)
        else:
            self.classifier = nn.Linear(in_plane, num_classes)
        self.prefix = self.__class__.__name__
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward_net(self, x):
        if isinstance(x, dict):
            x = x['features'][self.input_feature_idx]
        if self.cls_type == 'token':
            x = x[1]
        else:
            x = x[0]
            x = torch.mean(x, dim=1, keepdim=False)
        if self.representation_size is None:
            # finetune on new task
            logits = self.classifier(x)
        else:
            # train from scratch
            logits = self.classifier(self.tanh(self.pre_logits(x)))
        return {'logits': logits}

    def forward(self, input):
        return self.forward_net(input)
