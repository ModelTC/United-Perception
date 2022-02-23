import torch.nn as nn

from up.tasks.det.models.heads.roi_head.retina_head import RetinaHeadWithBN
from up.utils.general.registry_factory import MODULE_ZOO_REGISTRY

from .utils import get_inplane


@MODULE_ZOO_REGISTRY.register('UnionRetinaHeadWithBN')
class UnionRetinaHeadWithBN(RetinaHeadWithBN):
    def __init__(self,
                 inplanes,
                 feat_planes,
                 num_classes,
                 normalize=None,
                 initializer=None,
                 num_conv=4,
                 num_level=None,
                 class_activation='sigmoid',
                 init_prior=0.01,
                 num_anchors=9,
                 share_subnet=False,
                 # cls
                 input_feature_idx=-1,
                 # classification_inplanes=512,
                 classification_num_classes=None,
                 classification_feat_planes=None,
                 sum_feature=False,
                 share_cls_subnet=False):
        super(UnionRetinaHeadWithBN, self).__init__(inplanes,
                                                    feat_planes,
                                                    num_classes,
                                                    normalize=normalize,
                                                    initializer=initializer,
                                                    num_conv=num_conv,
                                                    num_level=num_level,
                                                    class_activation=class_activation,
                                                    init_prior=init_prior,
                                                    num_anchors=num_anchors,
                                                    share_subnet=share_subnet)
        self.input_feature_idx = input_feature_idx
        self.classification_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.share_cls_subnet = share_cls_subnet
        if self.share_cls_subnet:
            classification_inplanes = feat_planes  # subnet channel
        else:
            if sum_feature:
                classification_inplanes = get_inplane(inplanes, 0)
            else:
                classification_inplanes = get_inplane(inplanes, input_feature_idx)
        self.classification_inplanes = classification_inplanes
        if classification_feat_planes is None:
            classification_feat_planes = feat_planes
        self.classification_fc_embeding = nn.Linear(self.classification_inplanes, classification_feat_planes)
        self.classification_num_classes = classification_num_classes
        self.relu = nn.ReLU(inplace=True)
        self.classification_cls_fc = nn.Linear(classification_feat_planes, self.classification_num_classes)
        self.sum_feature = sum_feature

    def forward(self, input):
        if 'gt_bboxes' not in input:
            if self.sum_feature:
                features = self.forward_classification_sum_feature(input)
            else:
                features = self.forward_classification_feature(input)
            cls_pred = self.classification_cls_fc(features)
            return {'logits': cls_pred}
        else:
            return super().forward(input)

    def forward_classification_feature(self, input):
        features = input['features']
        if isinstance(features, list):
            features = features[self.input_feature_idx]
        if self.share_cls_subnet:
            features = self.cls_subnet[self.input_feature_idx](features)
        features = self.classification_pool(features)
        features = features.view(features.size(0), -1)
        features = self.relu(self.classification_fc_embeding(features))
        return features

    def forward_classification_sum_feature(self, input):
        features = input['features']
        out_feature = None
        for idx, feature in enumerate(features):
            if self.share_cls_subnet:
                feature = self.cls_subnet[idx](feature)
            feature = self.classification_pool(feature)
            feature = feature.view(feature.size(0), -1)
            feature = self.relu(self.classification_fc_embeding(feature))
            if out_feature is None:
                out_feature = feature
            else:
                out_feature = out_feature + feature
        return out_feature
