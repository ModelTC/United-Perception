import torch.nn as nn


class AdjustFeature(nn.Module):
    def __init__(self, s_feature_size, t_feature_size, s_adjust=True, t_adjust=False, adjust_type='bn'):
        super(AdjustFeature, self).__init__()
        self._register_adjust_layer_map()
        self.s_adjust = self.adjust_map[adjust_type](s_feature_size, t_feature_size, s_adjust)
        self.t_adjust = self.adjust_map[adjust_type](t_feature_size, s_feature_size, t_adjust)

    def _register_adjust_layer_map(self):
        def _bn_layer(in_c, out_c, adjust=False):
            if adjust:
                return nn.Sequential(
                    nn.Conv2d(in_c, out_c, kernel_size=1),
                    nn.BatchNorm2d(out_c)
                )
            else:
                return lambda x: x

        def _relu_layer(in_c, out_c, adjust=False):
            if adjust:
                return nn.Sequential(
                    nn.Conv2d(in_c, out_c, kernel_size=1),
                    nn.BatchNorm2d(out_c),
                    nn.ReLU(inplace=True)
                )
            else:
                return lambda x: x

        def _multi_conv_layer(in_c, out_c, adjust=False):
            if adjust:
                return nn.Sequential(
                    nn.Conv2d(in_c, out_c, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_c, out_c, kernel_size=1)
                )
            else:
                return lambda x: x

        self.adjust_map = {
            "bn": _bn_layer,
            "relu": _relu_layer,
            "multi": _multi_conv_layer
        }

    def forward(self, s_feature, t_feature):
        s_feature = self.s_adjust(s_feature)
        t_feature = self.t_adjust(t_feature)
        return s_feature, t_feature


class AdjustFeatures(nn.Module):
    def __init__(self, s_features, t_features, adjust_config=None):
        super(AdjustFeatures, self).__init__()
        self.adjust_layers = []
        self.adjust_config = []
        s_features = list(self._expand_list(s_features))
        t_features = list(self._expand_list(t_features))

        if isinstance(adjust_config, list):
            self.adjust_config = list(self._expand_list(adjust_config))
            if len(self.adjust_config) != len(s_features):
                raise Exception("adjust config unmatched. feature num = {} VS adjust config num = {}".format(
                    len(s_features), len(self.adjust_config)))
        elif isinstance(adjust_config, dict):
            self.adjust_config = [adjust_config for _ in s_features]
        else:
            self.adjust_config = [{} for _ in s_features]

        for i in range(len(s_features)):
            s_feature_c = s_features[i].shape[1]
            t_feature_c = t_features[i].shape[1]
            if s_feature_c == t_feature_c:
                self.adjust_config[i]['s_adjust'] = False
                self.adjust_config[i]['t_adjust'] = False
            self.adjust_layers.append(AdjustFeature(s_feature_c, t_feature_c, **self.adjust_config[i]).cuda())

    def _expand_list(self, nested_list):
        for item in nested_list:
            if not isinstance(item, list):
                yield item
            else:
                yield from self._expand_list(item)

    def forward(self, s_features, t_features):
        for i in range(len(s_features)):
            s_features[i], t_features[i] = self.adjust_layers[i](s_features[i], t_features[i])
        return s_features, t_features
