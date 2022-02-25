import torch.nn as nn
from up.utils.general.registry_factory import MODULE_ZOO_REGISTRY

__all__ = ['KeypointFPNHead']


@MODULE_ZOO_REGISTRY.register('kp_fpn')
class KeypointFPNHead(nn.Module):
    def __init__(self,
                 inplanes,
                 num_classes,
                 has_bg,
                 num_level=4):
        super(KeypointFPNHead, self).__init__()
        self.num_classes = num_classes

        for lvl in range(num_level):
            self.add_module(self.get_predict_name(lvl),
                            nn.Conv2d(
                            inplanes,
                            num_classes + has_bg,
                            kernel_size=3,
                            stride=1,
                            padding=1))

    def get_predict_name(self, idx):
        return 'predict{}'.format(idx + 1)

    def get_predict(self, idx):
        return getattr(self, self.get_predict_name(idx))

    def forward(self, input):
        output = {}
        output['pred'] = self.forward_net(input)
        output['num_classes'] = self.num_classes
        output['has_bg'] = self.has_bg
        return output

    def forward_net(self, input):
        out = []
        for idx, x in enumerate(input['features']):
            out.append(self.get_predict(idx)(x))
        return out
