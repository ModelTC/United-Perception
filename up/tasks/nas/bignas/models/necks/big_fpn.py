# Import from pod
# Import from third library
import torch
import torch.nn as nn
import torch.nn.functional as F

from up.tasks.nas.bignas.models.ops.dynamic_blocks import DynamicConvBlock
from up.tasks.nas.bignas.models.search_space.base_bignas_searchspace import BignasSearchSpace
from up.utils.model.initializer import initialize_from_cfg
# from up.tasks.nas.bignas.models.ops.dynamic_utils import build_norm_layer
from up.tasks.nas.bignas.models.ops.normal_blocks import ConvBlock, get_same_length
from up.utils.model.normalize import build_norm_layer

__all__ = ['big_fpn']


kwargs = {
    'out_channel': {
        'space': {
            'min': [128, 128],
            'max': [256, 512],
            'stride': [16, 32]},
        'sample_strategy': 'stage_wise'},
    'kernel_size': {
        'space': {
            'min': [1, 3],
            'max': [1, 3],
            'stride': 2},
        'sample_strategy': 'stage_wise'},
    'depth': {
        'space': {
            'min': [3, 5],
            'max': [3, 5],
            'stride': 1},
        'sample_strategy': 'stage_wise_depth'},
}


class BigFPN(BignasSearchSpace):
    """
    Dynamic Feature Pyramid Network

    .. note::

        If num_level is larger than backbone's output feature layers, additional layers will be stacked

    """

    def __init__(self,
                 inplanes,
                 start_level,
                 num_level,
                 out_strides,
                 downsample,
                 upsample,
                 normalize=None,
                 tocaffe_friendly=False,
                 initializer=None,
                 align_corners=True,
                 to_onnx=False,
                 use_p5=False,
                 **kwargs):
        """
        Arguments:
            - inplanes (:obj:`list` of :obj:`int`): input channel
            - start_level (:obj:`int`): start layer of backbone to apply FPN, it's only used for naming convs.
            - num_level (:obj:`int`): number of FPN layers
            - out_strides (:obj:`list` of :obj:`int`): stride of FPN output layers
            - downsample (:obj:`str`): method to downsample, for FPN, it's ``pool``, for RetienaNet, it's ``conv``
            - upsample (:obj:`str`): method to upsample, ``nearest`` or ``bilinear``
            - normalize (:obj:`dict`): config of Normalization Layer
            - initializer (:obj:`dict`): config for model parameter initialization
            - depth (dict or list): all conv layer are divided into 2 stage, all lateral conv share one stage
                                    all pconv and downsample conv share one stage
            - kernel_size (dict or list): stage-wise kernel size search space settings
            - out_channel (dict or list): stage-wise out channel search space settings

        """

        super(BigFPN, self).__init__(**kwargs)

        assert downsample in ['pool', 'conv'], downsample
        assert isinstance(inplanes, list)

        self.normalize = normalize
        self.initializer = initializer
        self.inplanes = inplanes
        self.out_strides = out_strides
        self.start_level = start_level
        self.num_level = num_level
        self.downsample = downsample
        self.upsample = upsample
        self.tocaffe_friendly = tocaffe_friendly
        if upsample == 'nearest':
            align_corners = None
        self.align_corners = align_corners
        self.to_onnx = to_onnx
        self.use_p5 = use_p5
        assert num_level == len(out_strides)
        _inter_planes = self.normal_settings['out_channel'][0]
        _out_planes = self.normal_settings['out_channel'][1]
        _inter_kernel_size = self.normal_settings['kernel_size'][0]
        _out_kernel_size = self.normal_settings['kernel_size'][1]

        assert self.normal_settings['depth'][0] == len(inplanes)

        assert _inter_planes == _out_planes
        max_out_planes = _out_planes
        # max_out_planes = max(_inter_planes, _out_planes)
        if normalize is None:
            use_bn = False
        else:
            use_bn = True

        for lvl_idx in range(num_level):
            if lvl_idx < len(inplanes):
                planes = inplanes[lvl_idx]
                self.add_module(
                    self.get_lateral_name(lvl_idx),
                    DynamicConvBlock(planes, _inter_planes, kernel_size_list=_inter_kernel_size,
                                     stride=1, act_func='',
                                     use_bn=use_bn, normalize=normalize, bias=(normalize is None)))
                self.add_module(
                    self.get_pconv_name(lvl_idx),
                    DynamicConvBlock(_inter_planes, _out_planes, kernel_size_list=_out_kernel_size,
                                     stride=1, act_func='',
                                     use_bn=use_bn, normalize=normalize, bias=(normalize is None)))
            else:
                if self.downsample == 'pool':
                    self.add_module(
                        self.get_downsample_name(lvl_idx),
                        nn.MaxPool2d(kernel_size=1, stride=2, padding=0))  # strange pooling
                else:
                    self.add_module(
                        self.get_downsample_name(lvl_idx),
                        DynamicConvBlock(max_out_planes, max_out_planes, kernel_size_list=_out_kernel_size,
                                         stride=2, use_bn=False, act_func='', normalize=normalize, bias=True))
        initialize_from_cfg(self, initializer)

        # self.gen_dynamic_module_list(dynamic_types=[DynamicConvBlock])

    def get_lateral_name(self, idx):
        return 'c{}_lateral'.format(idx + self.start_level)

    def get_lateral(self, idx):
        return getattr(self, self.get_lateral_name(idx))

    def get_downsample_name(self, idx):
        return 'p{}_{}'.format(idx + self.start_level, self.downsample)

    def get_downsample(self, idx):
        return getattr(self, self.get_downsample_name(idx))

    def get_pconv_name(self, idx):
        return 'p{}_conv'.format(idx + self.start_level)

    def get_pconv(self, idx):
        return getattr(self, self.get_pconv_name(idx))

    def forward(self, input):
        """
        .. attention:

            - all the lateral_convs must be adjacent in the computation graph during forward.
            - so do the pconvs.

        .. note::

            - For faster-rcnn, get P2-P5 from C2-C5, then P6 = pool(P5)
            - For RetinaNet, get P3-P5 from C3-C5, then P6 = Conv(C5), P7 = Conv(P6)

        Arguments:
            - input (:obj:`dict`): output of ``Backbone``

        Returns:
            - out (:obj:`dict`):

        Input example::

            {
                'features': [],
                'strides': []
            }

        Output example::

            {
                'features': [], # list of tenosr
                'strides': []   # list of int
            }
        """
        features = input['features']
        assert len(self.inplanes) == len(features)
        laterals = [self.get_lateral(i)(x) for i, x in enumerate(features)]

        features = []

        # top down pathway
        for lvl_idx in range(len(self.inplanes))[::-1]:
            if lvl_idx < len(self.inplanes) - 1:
                if self.tocaffe_friendly:
                    laterals[lvl_idx] += F.interpolate(laterals[lvl_idx + 1],
                                                       scale_factor=2,
                                                       mode=self.upsample,
                                                       align_corners=self.align_corners)
                else:
                    # nart_tools may not support to interpolate to the size of other feature
                    # you may need to modify upsample or interp layer in prototxt manually.
                    upsize = laterals[lvl_idx].shape[-2:]
                    laterals[lvl_idx] += F.interpolate(laterals[lvl_idx + 1],
                                                       size=upsize,
                                                       mode=self.upsample,
                                                       align_corners=self.align_corners)
            out = self.get_pconv(lvl_idx)(laterals[lvl_idx])
            features.append(out)
        features = features[::-1]

        # bottom up further
        if self.downsample == 'pool' or self.use_p5:
            x = features[-1]  # for faster-rcnn, use P5 to get P6
        else:
            x = laterals[-1]  # for RetinaNet, ues C5 to get P6, P7
        for lvl_idx in range(self.num_level):
            if lvl_idx >= len(self.inplanes):
                x = self.get_downsample(lvl_idx)(x)
                features.append(x)
        return {'features': features, 'strides': self.get_outstrides()}

    def get_outplanes(self):
        """
        Return:
            - outplanes (:obj:`list` of :obj:`int`)
        """
        out_channel = self.curr_subnet_settings['out_channel'][-1]
        if isinstance(out_channel, list):
            return max(out_channel)
        elif isinstance(out_channel, int):
            return out_channel
        else:
            raise NotImplementedError

    def get_outstrides(self):
        return self.out_strides

    def get_fake_input(self, input_shape=None):
        if input_shape is None:
            resolution = 56 if len(self.inplanes) == 4 else 28
            input_shape = []
            for i, channel in enumerate(self.inplanes):
                divide = 2**i
                input_shape.append([1, channel, int(resolution / divide), int(resolution / divide)])
        fake_input = []
        for shape in input_shape:
            input = torch.zeros(*shape)
            device = next(self.parameters(), torch.tensor([])).device
            input = input.to(device)
            for m in self.modules():
                if isinstance(m, torch.nn.Conv2d) and m.weight.dtype == torch.float16:
                    input = input.half()
            fake_input.append(input)
        input = {'features': fake_input}
        return input

    def build_active_subnet(self, subnet_settings, inplanes):
        subnet = bignas_fpn(inplanes=inplanes, start_level=self.start_level, num_level=self.num_level,
                            out_strides=self.out_strides, downsample=self.downsample, upsample=self.upsample,
                            normalize=self.normalize, tocaffe_friendly=self.tocaffe_friendly,
                            initializer=self.initializer, align_corners=self.align_corners, to_onnx=self.to_onnx,
                            use_p5=self.use_p5, out_channel=subnet_settings['out_channel'],
                            kernel_size=subnet_settings['kernel_size'], depth=subnet_settings['depth'])
        return subnet

    @property
    def module_str(self):
        _str = ''
        for lvl_idx in range(self.num_level):
            if lvl_idx < len(self.inplanes):
                _str += self.get_lateral_name(lvl_idx) + '_' + self.get_lateral(lvl_idx).module_str + '\n'
                _str += self.get_pconv_name(lvl_idx) + '_' + self.get_pconv(lvl_idx).module_str + '\n'
            else:
                _str += self.get_downsample_name(lvl_idx) + '_'
                if self.downsample == 'pool':
                    _str += 'MaxPool' + '\n'
                else:
                    _str += self.get_downsample(lvl_idx).module_str + '\n'
        return _str


def big_fpn(*args, **kwargs):
    return BigFPN(*args, **kwargs)


def bignas_fpn(**kwargs):
    return BignasFPN(**kwargs)


class BignasFPN(nn.Module):
    """
    Feature Pyramid Network

    .. note::

        If num_level is larger than backbone's output feature layers, additional layers will be stacked

    """

    def __init__(self,
                 inplanes,
                 start_level,
                 num_level,
                 out_strides,
                 downsample,
                 upsample,
                 normalize=None,
                 tocaffe_friendly=False,
                 initializer=None,
                 align_corners=True,
                 to_onnx=False,
                 use_p5=False,
                 out_channel=[256, 256],
                 kernel_size=[1, 3],
                 depth=[3, 5]):
        """
        Arguments:
            - inplanes (:obj:`list` of :obj:`int`): input channel
            - start_level (:obj:`int`): start layer of backbone to apply FPN, it's only used for naming convs.
            - num_level (:obj:`int`): number of FPN layers
            - out_strides (:obj:`list` of :obj:`int`): stride of FPN output layers
            - downsample (:obj:`str`): method to downsample, for FPN, it's ``pool``, for RetienaNet, it's ``conv``
            - upsample (:obj:`str`): method to upsample, ``nearest`` or ``bilinear``
            - normalize (:obj:`dict`): config of Normalization Layer
            - initializer (:obj:`dict`): config for model parameter initialization
            - depth (list): all conv layer are divided into 2 stage, all lateral conv share one stage
                                    all pconv and downsample conv share one stage
            - kernel_size (list): stage-wise kernel size search space settings
            - out_channel (list): stage-wise out channel search space settings

        """

        super(BignasFPN, self).__init__()

        assert isinstance(normalize, dict) and 'type' in normalize
        norm_split = normalize['type'].split('_', 1)
        if norm_split[0] == 'dynamic':
            normalize['type'] = norm_split[-1]
        global NormLayer

        if normalize is not None:
            NormLayer = build_norm_layer(num_features=None, cfg=normalize, layer_instance=False)
            use_bn = True
        else:
            NormLayer = None
            use_bn = False

        assert downsample in ['pool', 'conv'], downsample
        assert isinstance(inplanes, list)

        self.inplanes = inplanes
        self.outstrides = out_strides
        self.start_level = start_level
        self.num_level = num_level
        self.downsample = downsample
        self.upsample = upsample
        self.tocaffe_friendly = tocaffe_friendly
        if upsample == 'nearest':
            align_corners = None
        self.align_corners = align_corners
        self.to_onnx = to_onnx
        self.use_p5 = use_p5
        assert num_level == len(out_strides)

        _inter_planes = out_channel[0]
        _out_planes = out_channel[1]
        self.kernel_size = get_same_length(kernel_size, depth)
        count_lateral = 0
        count_pconv = depth[0]
        self.outplanes = _out_planes

        for lvl_idx in range(num_level):
            if lvl_idx < len(inplanes):
                _inter_kernel_size = self.kernel_size[count_lateral]
                _out_kernel_size = self.kernel_size[count_pconv]
                count_lateral += 1
                count_pconv += 1
                planes = inplanes[lvl_idx]
                self.add_module(
                    self.get_lateral_name(lvl_idx),
                    ConvBlock(planes, _inter_planes, kernel_size=_inter_kernel_size,
                              stride=1, act_func='',
                              use_bn=use_bn, NormLayer=NormLayer, bias=(normalize is None)))
                self.add_module(
                    self.get_pconv_name(lvl_idx),
                    ConvBlock(_inter_planes, _out_planes, kernel_size=_out_kernel_size,
                              stride=1, act_func='',
                              use_bn=use_bn, NormLayer=NormLayer, bias=(normalize is None)))
            else:
                if lvl_idx == len(inplanes):
                    _in_planes = _inter_planes
                else:
                    _in_planes = _out_planes
                _out_kernel_size = self.kernel_size[count_pconv]
                count_pconv += 1
                if self.downsample == 'pool':
                    self.add_module(
                        self.get_downsample_name(lvl_idx),
                        nn.MaxPool2d(kernel_size=1, stride=2, padding=0))  # strange pooling
                else:
                    self.add_module(
                        self.get_downsample_name(lvl_idx),
                        ConvBlock(_in_planes, _out_planes, kernel_size=_out_kernel_size,
                                  stride=2, act_func='',
                                  use_bn=False, NormLayer=NormLayer, bias=True))
        initialize_from_cfg(self, initializer)

    def get_lateral_name(self, idx):
        return 'c{}_lateral'.format(idx + self.start_level)

    def get_lateral(self, idx):
        return getattr(self, self.get_lateral_name(idx))

    def get_downsample_name(self, idx):
        return 'p{}_{}'.format(idx + self.start_level, self.downsample)

    def get_downsample(self, idx):
        return getattr(self, self.get_downsample_name(idx))

    def get_pconv_name(self, idx):
        return 'p{}_conv'.format(idx + self.start_level)

    def get_pconv(self, idx):
        return getattr(self, self.get_pconv_name(idx))

    def forward(self, input):
        """
        .. attention:

            - all the lateral_convs must be adjacent in the computation graph during forward.
            - so do the pconvs.

        .. note::

            - For faster-rcnn, get P2-P5 from C2-C5, then P6 = pool(P5)
            - For RetinaNet, get P3-P5 from C3-C5, then P6 = Conv(C5), P7 = Conv(P6)

        Arguments:
            - input (:obj:`dict`): output of ``Backbone``

        Returns:
            - out (:obj:`dict`):

        Input example::

            {
                'features': [],
                'strides': []
            }

        Output example::

            {
                'features': [], # list of tenosr
                'strides': []   # list of int
            }
        """
        features = input['features']
        assert len(self.inplanes) == len(features)
        laterals = [self.get_lateral(i)(x) for i, x in enumerate(features)]

        features = []

        # top down pathway
        for lvl_idx in range(len(self.inplanes))[::-1]:
            if lvl_idx < len(self.inplanes) - 1:
                if self.tocaffe_friendly:
                    laterals[lvl_idx] += F.interpolate(laterals[lvl_idx + 1],
                                                       scale_factor=2,
                                                       mode=self.upsample,
                                                       align_corners=self.align_corners)
                else:
                    # nart_tools may not support to interpolate to the size of other feature
                    # you may need to modify upsample or interp layer in prototxt manually.
                    upsize = laterals[lvl_idx].shape[-2:]
                    laterals[lvl_idx] += F.interpolate(laterals[lvl_idx + 1],
                                                       size=upsize,
                                                       mode=self.upsample,
                                                       align_corners=self.align_corners)
            out = self.get_pconv(lvl_idx)(laterals[lvl_idx])
            features.append(out)
        features = features[::-1]

        # bottom up further
        if self.downsample == 'pool' or self.use_p5:
            x = features[-1]  # for faster-rcnn, use P5 to get P6
        else:
            x = laterals[-1]  # for RetinaNet, ues C5 to get P6, P7
        for lvl_idx in range(self.num_level):
            if lvl_idx >= len(self.inplanes):
                x = self.get_downsample(lvl_idx)(x)
                features.append(x)
        return {'features': features, 'strides': self.get_outstrides()}

    def get_outplanes(self):
        """
        Return:
            - outplanes (:obj:`list` of :obj:`int`)
        """
        return self.outplanes

    def get_outstrides(self):
        return self.outstrides

    def get_fake_input(self):
        fake_input = [self.get_lateral(0).conv.weight.new_zeros([1, c, res, res]) for
                      (c, res) in zip(self.inplanes, [28, 14, 7])]
        input = {'features': fake_input}
        return input

    def build_active_subnet(self, subnet_settings):
        subnet = None
        return subnet
