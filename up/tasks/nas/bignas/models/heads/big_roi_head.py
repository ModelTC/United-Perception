# Standard Library
import torch
import torch.nn as nn
from functools import partial
import copy

# Import from local library
from up.tasks.nas.bignas.models.search_space import BignasSearchSpace
from up.tasks.nas.bignas.models.ops.dynamic_blocks import DynamicConvBlock
from up.tasks.nas.bignas.models.ops.dynamic_ops import build_dynamic_norm_layer
from up.utils.general.log_helper import default_logger as logger
from up.tasks.nas.bignas.models.ops.normal_blocks import ConvBlock
# from up.tasks.nas.bignas.models.ops.dynamic_utils import build_norm_layer
from up.utils.model.normalize import build_norm_layer

try:
    import spring.linklink as link
except:  # noqa
    from up.utils.general.fake_linklink import link


big_roi_head_kwargs = {
    'out_channel': {
        'space': {
            'min': [128],
            'max': [256],
            'stride': 16},
        'sample_strategy': 'stage_wise'},
    'kernel_size': {
        'space': {
            'min': [1],
            'max': [3],
            'stride': 2},
        'sample_strategy': 'stage_wise'},
    'depth': {
        'space': {
            'min': [2],
            'max': [4],
            'stride': 1},
        'sample_strategy': 'stage_wise_depth'},
}


class BigRoiHead(BignasSearchSpace):
    """
    Head for the first stage detection task

    .. note::

        0 is always for the background class.
    """

    def __init__(self, inplanes, num_levels, num_conv, normalize={'type': 'dynamic_solo_bn'}, **kwargs):
        """
        Arguments:
            - inplanes (:obj:`list` or :obj:`int`): input channel
            - num_levels (:obj:`int`): num of level
            - num_conv (:obj:`int`): num of conv
        """
        super(BigRoiHead, self).__init__(**kwargs)

        self.num_levels = num_levels
        self.inplanes = inplanes

        self.mlvl_heads = nn.ModuleList()
        norm_layer = build_dynamic_norm_layer(normalize)
        if 'switch' in normalize['type']:
            norm_layer = partial(norm_layer, width_list=self.normal_settings['out_channel'][0])

        self.num_conv = num_conv
        assert num_conv == sum(self.normal_settings['depth'])

        for lvl in range(num_levels):
            layers = []
            inplanes = self.inplanes
            feat_planes = self.normal_settings['out_channel'][0]
            if isinstance(feat_planes, list):
                feat_planes = max(feat_planes)
            kernel_size = self.normal_settings['kernel_size'][0]
            for conv_idx in range(self.num_conv):
                if lvl == 0:
                    layers.append(nn.Sequential(
                        DynamicConvBlock(inplanes, feat_planes, kernel_size_list=kernel_size, stride=1,
                                         use_bn=False, act_func='',
                                         normalize=normalize),
                        norm_layer(feat_planes),
                        nn.ReLU(inplace=True)))
                    inplanes = feat_planes
                else:
                    layers.append(nn.Sequential(
                        self.mlvl_heads[-1][conv_idx][0],
                        norm_layer(feat_planes),
                        nn.ReLU(inplace=True)))
                    inplanes = feat_planes
            self.mlvl_heads.append(nn.Sequential(*layers))

        # self.gen_dynamic_module_list(dynamic_types=[DynamicConvBlock])

    def forward(self, x):
        for conv_idx in range(self.num_conv):
            x = self.mlvl_heads[0][conv_idx](x)
        return x

    def get_outplanes(self):
        return self.curr_subnet_settings['out_channel'][-1]

    def get_fake_input(self, input_shape=None):
        if not input_shape:
            input_shape = [1, self.inplanes, 64, 64]
        else:
            assert len(input_shape) == 4
        input = torch.zeros(*input_shape)
        device = next(self.parameters(), torch.tensor([])).device
        input = input.to(device)
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) and m.weight.dtype == torch.float16:
                input = input.half()
        return input

    @property
    def module_str(self):
        _str = ''
        for lvl in range(self.num_levels):
            for conv_idx in range(self.num_conv):
                _str += self.mlvl_heads[lvl][conv_idx][0].module_str + '\n'
        return _str

    def build_active_subnet(self, subnet_settings, inplanes=None):
        if inplanes is None:
            inplanes = self.inplanes
        subnet = bignas_roi_head(inplanes=inplanes,
                                 num_levels=self.num_levels, num_conv=self.num_conv,
                                 depth=subnet_settings['depth'], kernel_size=subnet_settings['kernel_size'],
                                 out_channel=subnet_settings['out_channel'])
        return subnet

    def sample_active_subnet_weights(self, subnet_settings=None, inplanes=None):
        subnet = super().sample_active_subnet_weights(subnet_settings, inplanes)

        logger.info('Warning: Hot fix for BatchNorm as they are not in the block')
        logger.info('It is only right with subnet and supernet has the same depth')

        for module, super_module in zip(subnet.modules(), self.modules()):
            if isinstance(super_module, nn.BatchNorm2d) or isinstance(super_module, link.nn.SyncBatchNorm2d):
                logger.info('special copy for individual BatchNorm')
                # assert len(module.state_dict().keys()) == len(super_module.state_dict().keys()), \
                #    'module {} and super module {} has different number of keys'.format(module.state_dict.keys(),
                #                                                                        super_module.state_dict.keys())
                for (k1, v1), (k2, v2) in zip(module.state_dict().items(), super_module.state_dict().items()):
                    assert v1.dim() == v2.dim()
                    if v1.dim() == 0:
                        v1.copy_(copy.deepcopy(v2.data))
                    elif v1.dim() == 1:
                        v1.copy_(copy.deepcopy(v2[:v1.size(0)].data))
        return subnet


def big_roi_head(*args, **kwargs):
    return BigRoiHead(*args, **kwargs)


def bignas_roi_head(**kwargs):
    return BignasRoiHead(**kwargs)


class BignasRoiHead(nn.Module):
    """
    Head for the first stage detection task

    .. note::

        0 is always for the background class.
    """

    def __init__(self, inplanes, num_levels=5, num_conv=4, normalize={'type': 'dynamic_solo_bn'},
                 depth=[1, 1, 1, 1], kernel_size=[3, 3, 3, 3], out_channel=[256, 256, 256, 256]):
        """
        Arguments:
            - inplanes (:obj:`list` or :obj:`int`): input channel
            - num_level (:obj:`int`): number of levels
            - depth (:obj:`list`): number of conv
            - kernel_size (:obj:`list` of `int`): kernel size settings
            - out_channel (:obj:`list` of `int`): out_channel settings
        """
        super(BignasRoiHead, self).__init__()

        global NormLayer

        assert isinstance(normalize, dict) and 'type' in normalize
        norm_split = normalize['type'].split('_', 1)
        if norm_split[0] == 'dynamic':
            normalize['type'] = norm_split[-1]
        NormLayer = build_norm_layer(num_features=None, cfg=normalize, layer_instance=False)

        self.num_levels = num_levels
        self.inplanes = inplanes

        self.mlvl_heads = nn.ModuleList()

        self.num_conv = num_conv
        assert num_conv == sum(depth)
        self.out_channel = out_channel
        self.kernel_size = kernel_size

        feat_planes = self.out_channel[0]
        kernel_size = self.kernel_size[0]
        for lvl in range(num_levels):
            layers = []
            inplanes = self.inplanes
            for conv_idx in range(self.num_conv):
                if lvl == 0:
                    layers.append(nn.Sequential(
                        ConvBlock(inplanes, feat_planes, kernel_size=kernel_size, stride=1,
                                  use_bn=False, act_func=''),
                        NormLayer(feat_planes),
                        nn.ReLU(inplace=True)))
                    inplanes = feat_planes
                else:
                    layers.append(nn.Sequential(
                        self.mlvl_heads[-1][conv_idx][0],
                        NormLayer(feat_planes),
                        nn.ReLU(inplace=True)))
                    inplanes = feat_planes
            self.mlvl_heads.append(nn.Sequential(*layers))

    def forward(self, x):
        for conv_idx in range(self.num_conv):
            x = self.mlvl_heads[0][conv_idx](x)
        return x

    def get_outplanes(self):
        return self.out_channel[0]
