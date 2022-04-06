import torch.nn as nn
from up.utils.general.registry_factory import MODULE_ZOO_REGISTRY


@MODULE_ZOO_REGISTRY.register('ADAP')
class ADAP(nn.Module):
    def __init__(self,
                 inplanes,
                 outplanes,
                 out_layers=[0, 1, 2, 3, 4],
                 kernel_size=3,
                 with_relu=False,
                 do_init_weights=False,
                 out_name='adapt_neck_features'):
        super(ADAP, self).__init__()
        self.num_level = len(out_layers)
        self.out_layers = out_layers

        assert out_name in ['adapt_neck_features', 'adapt_bb_features'], 'only support \
                adapt_neck_features and adapt_bb_features'
        self.out_name = out_name

        if not isinstance(inplanes, list):
            self.inplanes = [inplanes] * self.num_level
        else:
            self.inplanes = [p for idx, p in enumerate(inplanes) if idx in out_layers]
        if not isinstance(outplanes, list):
            self.outplanes = [outplanes] * self.num_level
        else:
            self.outplanes = outplanes

        self.conv = nn.ModuleList()
        self.with_relu = with_relu
        for idx in range(self.num_level):
            if kernel_size == 3:
                if with_relu:
                    self.conv.append(nn.Sequential(
                        nn.Conv2d(self.inplanes[idx], self.outplanes[idx], 3, padding=(1, 1)),
                        nn.ReLU(),
                    ))
                else:
                    self.conv.append(nn.Sequential(
                        nn.Conv2d(self.inplanes[idx], self.outplanes[idx], 3, padding=(1, 1)),
                    ))
            elif kernel_size == 1:
                if with_relu:
                    self.conv.append(nn.Sequential(
                        nn.Conv2d(self.inplanes[idx], self.outplanes[idx], 1),
                        nn.ReLU(),
                    ))
                else:
                    self.conv.append(nn.Sequential(
                        nn.Conv2d(self.inplanes[idx], self.outplanes[idx], 1),
                    ))
            else:
                raise ValueError("other kernel size not completed")
        if do_init_weights:
            self._init_weights()

    def _init_weights(self):
        for m in self.conv:
            m[0].weight.data.normal_().fmod_(2).mul_(0.0001).add_(0)

    def forward(self, input):
        features = input['features']
        adapt_features = []
        for i, odx in enumerate(self.out_layers):
            adapt_features.append(self.conv[i](features[odx]))
        return {self.out_name: adapt_features}
