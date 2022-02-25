import torch.nn as nn
from up.utils.general.registry_factory import MODULE_ZOO_REGISTRY


@MODULE_ZOO_REGISTRY.register('kp_head')
class BaseKpHead(nn.Module):
    def __init__(self, num_classes, has_bg, input_feature_idx=-1):
        super(BaseKpHead, self).__init__()
        self.has_bg = has_bg
        self.num_classes = num_classes
        self.input_feature_idx = input_feature_idx

    def forward(self, input):
        output = {}
        output['pred'] = self.forward_net(input['features'])
        output['num_classes'] = self.num_classes
        output['has_bg'] = self.has_bg
        return output

    def forward_net(self, inupt):
        if self.training:
            return inupt
        else:
            return inupt[self.input_feature_idx]
