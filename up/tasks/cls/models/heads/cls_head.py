import torch
import torch.nn as nn
import torch.nn.functional as F
from up.utils.general.registry_factory import MODULE_ZOO_REGISTRY
from up.utils.model.initializer import trunc_normal_

__all__ = ['BaseClsHead', 'ConvNeXtHead', 'ViTHead']


@MODULE_ZOO_REGISTRY.register('norm_linear')
class NormedLinear(nn.Module):
    def __init__(self, in_features, out_features, tempearture=30):
        super(NormedLinear, self).__init__()
        self.tempearture = tempearture
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out * self.tempearture


@MODULE_ZOO_REGISTRY.register('base_cls_head')
class BaseClsHead(nn.Module):
    def __init__(self, num_classes, in_plane, input_feature_idx=-1, use_pool=True, dropout=None, classifier_cfg=None):
        super(BaseClsHead, self).__init__()
        self.num_classes = num_classes
        self.in_plane = in_plane
        self.input_feature_idx = input_feature_idx
        self.prefix = self.__class__.__name__
        self.use_pool = use_pool
        self.dropout = dropout
        self.build_classifier(in_plane, classifier_cfg)
        self._init_weights()
        if self.use_pool:
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        if self.dropout is not None:
            self.drop = nn.Dropout(p=dropout)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def build_single_classifier(self, in_plane, out_plane, cfg):
        if cfg is None:
            return nn.Linear(in_plane, out_plane)
        else:
            if 'in_features' not in cfg['kwargs']:
                cfg['kwargs'].update({'in_features': in_plane})
            if 'out_features' not in cfg['kwargs']:
                cfg['kwargs'].update({'out_features': out_plane})
            if cfg['type'] == 'linear':
                return nn.Linear(**cfg['kwargs'])
            else:
                linear_ins = MODULE_ZOO_REGISTRY[cfg['type']]
                return linear_ins(**cfg['kwargs'])

    def build_classifier(self, in_plane, cfgs):
        if isinstance(self.num_classes, list) or isinstance(self.num_classes, tuple):
            self.classifier = nn.ModuleList()
            for i, cls in enumerate(self.num_classes):
                if isinstance(cfgs, list):
                    self.classifier.append(self.build_single_classifier(in_plane, cls, cfgs[i]))
                else:
                    self.classifier.append(self.build_single_classifier(in_plane, cls, cfgs))
            self.multicls = True
        else:
            self.multicls = False
            self.classifier = self.build_single_classifier(in_plane, self.num_classes, cfgs)

    def get_pool_output(self, x):
        if self.use_pool:
            x = self.pool(x)
            x = x.flatten(1, -1)
        return x

    def get_logits(self, x):
        if self.multicls:
            logits = [self.classifier[idx](x) for idx, _ in enumerate(self.num_classes)]
        else:
            logits = self.classifier(x)
        return logits

    def get_dropout(self, x):
        if self.dropout is not None:
            x = self.drop(x)
        return x

    def forward_net(self, x):
        x = x['features'][self.input_feature_idx]
        x = self.get_pool_output(x)
        x = self.get_dropout(x)
        logits = self.get_logits(x)
        return {'logits': logits, 'deploy_output_node': logits}

    def forward(self, input):
        return self.forward_net(input)


@MODULE_ZOO_REGISTRY.register('fpn_cls_head')
class FPNClsHead(BaseClsHead):
    def __init__(self,
                 inplanes,
                 feat_planes,
                 num_classes,
                 sum_feature=True,
                 input_feature_idx=-1,
                 use_pool=True,
                 dropout=None):
        if isinstance(inplanes, list):
            if self.sum_feature:
                self.classification_inplanes = inplanes[0]
            else:
                self.classification_inplanes = inplanes[self.input_feature_idx]
        else:
            self.classification_inplanes = inplanes
        super().__init__(num_classes, self.classification_inplanes, input_feature_idx, use_pool, dropout)
        self.sum_feature = sum_feature
        self.classification_pool = self.pool
        self.relu = nn.ReLU(inplace=True)
        self.build_classifier(feat_planes)
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
class ConvNeXtHead(BaseClsHead):
    def __init__(self,
                 num_classes,
                 in_plane,
                 input_feature_idx=-1,
                 head_init_scale=1.,
                 use_pool=True,
                 dropout=None):
        super(ConvNeXtHead, self).__init__(num_classes, in_plane, input_feature_idx, use_pool, dropout)
        if self.multicls:
            for m in self.classifier:
                m.weight.data.mul_(head_init_scale)
                m.bias.data.mul_(head_init_scale)
        else:
            self.classifier.weight.data.mul_(head_init_scale)
            self.classifier.bias.data.mul_(head_init_scale)
        self.layer_norm = nn.LayerNorm(in_plane, eps=1e-6)

    def forward_net(self, x):
        if isinstance(x, dict):
            x = x['features'][self.input_feature_idx]
        x = self.get_pool_output(x)
        x = self.layer_norm(x)
        x = self.get_dropout(x)
        logits = self.get_logits(x)
        return {'logits': logits}

    def forward(self, input):
        return self.forward_net(input)


@MODULE_ZOO_REGISTRY.register('vit_head')
class ViTHead(BaseClsHead):
    def __init__(self,
                 num_classes,
                 in_plane,
                 input_feature_idx=-1,
                 use_pool=False,
                 dropout=None,
                 cls_type='token',
                 representation_size=None,
                 bn=None):
        super(ViTHead, self).__init__(num_classes, in_plane, input_feature_idx, use_pool, dropout)
        self.cls_type = cls_type
        self.representation_size = representation_size
        self.bn = bn
        if self.bn is not None:
            self.bn = torch.nn.BatchNorm1d(in_plane, affine=False, eps=1e-6)
        # if representation_size is not None
        # a pre-logits layer is inserted before classification head
        # it means transformer will be trained from scratch
        # otherwise the model is finetuned on new task
        if representation_size is not None:
            self.pre_logits = nn.Linear(in_plane, representation_size)
            self.tanh = nn.Tanh()
            self.build_classifier(representation_size)
        self._init_weights()

    def get_pool_output(self, x):
        x = x.reshape(x.shape[0], x.shape[1], -1).mean(dim=-1)
        return x

    def forward_net(self, x):
        if isinstance(x, dict):
            x = x['features'][self.input_feature_idx]
        if self.cls_type == 'token':
            x = x[1]
        else:
            x = x[0]
            x = self.get_pool_output(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.representation_size is not None:
            x = self.tanh(self.pre_logits(x))
        logits = self.get_logits(x)
        return {'logits': logits}

    def forward(self, input):
        return self.forward_net(input)
