from up.utils.general.registry_factory import MODULE_ZOO_REGISTRY
from up.tasks.cls.models.heads import BaseClsHead, ConvNeXtHead

__all__ = ['SparseBaseClsHead', 'SparseConvNeXtHead']


@MODULE_ZOO_REGISTRY.register('sparse_base_cls_head')
class SparseBaseClsHead(BaseClsHead):
    def __init__(self, num_classes, in_plane, input_feature_idx=-1, use_pool=True, dropout=None):
        super(SparseBaseClsHead, self).__init__(num_classes, in_plane, input_feature_idx=-1,
                                                use_pool=True, dropout=None)

    def forward_net(self, x):
        x = x['features'][self.input_feature_idx]
        x = self.get_pool_output(x)
        x = self.get_dropout(x)
        logits = self.get_logits(x)
        return {'logits': logits}


@MODULE_ZOO_REGISTRY.register('sparse_convnext_head')
class SparseConvNeXtHead(ConvNeXtHead):
    def __init__(self,
                 num_classes,
                 in_plane,
                 input_feature_idx=-1,
                 head_init_scale=1.,
                 use_pool=True,
                 dropout=None):
        super(SparseConvNeXtHead, self).__init__(num_classes, in_plane, input_feature_idx, use_pool, dropout)

    def forward_net(self, x):
        x = x['features'][self.input_feature_idx]
        x = self.get_pool_output(x)
        x = self.layer_norm(x)
        x = self.get_dropout(x)
        logits = self.get_logits(x)
        return {'logits': logits}
