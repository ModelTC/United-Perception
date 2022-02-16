import torch.nn as nn

from up.utils.general.registry_factory import MODULE_ZOO_REGISTRY
from up.tasks.det.models.heads.bbox_head.bbox_head import FC

from .utils import get_inplane


@MODULE_ZOO_REGISTRY.register('UnionFC')
class UnionFC(FC):
    def __init__(
        self,
        inplanes,
        feat_planes,
        num_classes,
        cfg,
        initializer=None,
        with_drp=False,
        # cls
        input_feature_idx=-1,
        classification_num_classes=None,
        classification_feat_planes=None,
        sum_feature=False,
        share_fc=False,
        share_classifier=False,
    ):
        super().__init__(inplanes,
                         feat_planes,
                         num_classes,
                         cfg,
                         initializer,
                         with_drp=with_drp)
        self.input_feature_idx = input_feature_idx
        self.classification_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.share_fc = share_fc
        self.share_classifier = share_classifier
        self.sum_feature = sum_feature

        if classification_feat_planes is None:
            classification_feat_planes = feat_planes

        # calc inplanes
        assert not share_fc or not sum_feature, \
            "use share_fc and sum_feature in the same time is not supported!"
        if share_fc:
            classification_feat_planes = feat_planes
        else:
            if sum_feature:
                classification_inplanes = get_inplane(inplanes, 0)
            else:
                classification_inplanes = get_inplane(inplanes,
                                                      input_feature_idx)

            self.classification_fc_embeding = nn.Linear(
                classification_inplanes, classification_feat_planes)

        self.classification_feat_planes = classification_feat_planes

        if not share_classifier:
            self.classification_classifier = nn.Linear(
                classification_feat_planes, classification_num_classes)
        else:
            assert num_classes == classification_num_classes, \
                f"share classifier must have same classes, {num_classes} vs {classification_num_classes}"

    def get_logits(self, input):
        if self.sum_feature:
            features = self.forward_classification_sum_feature(input)
        else:
            features = self.forward_classification_feature(input)

        if self.share_classifier:
            cls_pred = self.fc_rcnn_cls(features)
        else:
            cls_pred = self.classification_classifier(features)
        return {'logits': cls_pred, 'cls_features': features}

    def _share_fc_forward(self, x):
        x = self.relu(self.fc6(x))
        x = self.relu(self.fc7(x))
        return x

    def forward_classification_feature(self, input):
        features = input['features']
        if isinstance(features, list):
            features = features[self.input_feature_idx]
        if not self.share_fc:
            features = self.classification_pool(features)
        features = features.view(features.size(0), -1)
        if self.share_fc:
            features = self._share_fc_forward(features)
        else:
            features = self.relu(self.classification_fc_embeding(features))
        return features

    def forward_classification_sum_feature(self, input):
        features = input['features']
        out_feature = None

        for feature in features:
            # if not self.share_fc:
            feature = self.classification_pool(feature)
            feature = feature.view(feature.size(0), -1)
            if self.share_fc:
                feature = self._share_fc_forward(feature)
            else:
                feature = self.relu(self.classification_fc_embeding(feature))

            if out_feature is not None:
                out_feature = out_feature + feature
            else:
                out_feature = feature
        return out_feature

    def forward(self, input):
        if 'gt_bboxes' in input:
            return super().forward(input)
        else:
            return self.get_logits(input)
