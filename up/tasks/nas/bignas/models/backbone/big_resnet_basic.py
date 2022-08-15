import math
import torch.nn as nn

from up.tasks.nas.bignas.models.ops.dynamic_blocks import DynamicConvBlock, DynamicLinearBlock, DynamicBasicBlock, \
    DynamicStemBlock
from up.tasks.nas.bignas.models.search_space.base_bignas_searchspace import BignasSearchSpace
from up.utils.model.initializer import initialize_from_cfg
from up.tasks.nas.bignas.models.ops.normal_blocks import ConvBlock, LinearBlock, BasicBlock, \
    get_same_length, StemBlock
# from up.tasks.nas.bignas.models.ops.dynamic_utils import build_norm_layer
from up.utils.model.normalize import build_norm_layer

big_resnet_basic_kwargs = {
    'out_channel': {
        'space': {
            'min': [32, 32, 64, 128, 256],
            'max': [64, 96, 192, 384, 768],
            'stride': [16, 16, 32, 32, 64]},
        'sample_strategy': 'ordered_stage_wise'},
    'kernel_size': {
        'space': {
            'min': [3, 3, 3, 3, 3],
            'max': [7, 3, 3, 3, 3],
            'stride': 2},
        'sample_strategy': 'stage_wise'},
    'expand_ratio': [0, 1, 1, 1, 1],
    'depth': {
        'space': {
            'min': [1, 1, 1, 1, 1],
            'max': [1, 3, 3, 3, 3],
            'stride': 1},
        'sample_strategy': 'stage_wise_depth'},
}


class BigResNetBasic(BignasSearchSpace):

    def __init__(self,
                 # other settings
                 act_stages=['relu', 'relu', 'relu', 'relu', 'relu'],
                 stride_stages=[2, 1, 2, 2, 2],
                 use_maxpool=True,
                 dropout_rate=0,
                 divisor=8,
                 normalize={'type': 'dynamic_solo_bn'},
                 # clarify settings for detection task
                 gray=False,
                 frozen_layers=[],
                 out_layers=[],
                 out_strides=[],
                 task='classification',
                 num_classes=1000,
                 # search settings
                 kernel_transform=False,
                 zero_last_gamma=True,
                 deep_stem=False,
                 # search space
                 **kwargs):
        super(BigResNetBasic, self).__init__(**kwargs)
        r"""
        Arguments:
        - act_stages(:obj:`list` of 5 (stages+1) ints): activation list for blocks
        - stride_stages (:obj:`list` of 5 (stages+1) ints): stride list for stages
        - use_maxpool (:obj `bool`): use maxpooling to downsample or not
        - dropout_rate (:obj:`float`): dropout rate
        - divisor(:obj:`int`): divisor for channels
        - frozen_layers (:obj:`list` of :obj:`int`): Index of frozen layers, [0,4]
        - out_layers (:obj:`list` of :obj:`int`): Index of output layers, [0,4]
        - out_strides (:obj:`list` of :obj:`int`): Stride of outputs
        - task (:obj:`str`): Type of task, 'classification' or object 'detection'
        - num_classes (:obj:`int`): number of classification classes
        - kernel_transform (:obj:`bool`): use kernel tranformation matrix or not
        - zero_last_gamma (:obj:`bool`): zero the last gamma of BatchNorm in the main path or not
        - out_channel (:obj:`list` or `dict`): stage-wise search settings for out channel
        - depth (:obj:`list` or `dict`): stage-wise search settings for depth
        - kernel_size (:obj:`list` or `dict`): stage-wise search settings for kernel size
        - expand_ratio (:obj:`list` or `dict`): stage-wise search settings for expand ratio
        """

        self.act_stages = act_stages
        self.stride_stages = stride_stages
        self.use_maxpool = use_maxpool
        self.dropout_rate = dropout_rate
        self.divisor = divisor

        self.frozen_layers = frozen_layers
        self.out_layers = out_layers
        self.out_strides = out_strides
        if len(self.out_layers) != len(self.out_strides):
            self.out_layers = [int(math.log2(i)) - 1 for i in self.out_strides]

        self.task = task
        self.gray = gray
        self.num_classes = num_classes
        self.normalize = normalize
        self.deep_stem = deep_stem

        self.out_channel = self.normal_settings['out_channel']
        self.depth = self.normal_settings['depth']
        self.kernel_size = self.normal_settings['kernel_size']
        self.expand_ratio = self.normal_settings['expand_ratio']

        out_channel = [max(i) if isinstance(i, list) else i
                       for i in self.out_channel]
        self.out_planes = [out_channel[i] for i in self.out_layers]

        in_channel = 1 if gray else 3
        # first conv layer
        if self.deep_stem:
            self.first_conv = DynamicStemBlock(
                in_channel_list=in_channel, out_channel_list=self.out_channel[0],
                kernel_size_list=self.kernel_size[0], expand_ratio_list=self.expand_ratio[0],
                stride=self.stride_stages[0], act_func=self.act_stages[0],
                KERNEL_TRANSFORM_MODE=kernel_transform,
                normalize=normalize)
        else:
            self.first_conv = DynamicConvBlock(
                in_channel_list=in_channel, out_channel_list=self.out_channel[0],
                kernel_size_list=self.kernel_size[0],
                stride=self.stride_stages[0], act_func=self.act_stages[0],
                KERNEL_TRANSFORM_MODE=kernel_transform,
                normalize=normalize)

        if self.use_maxpool:
            self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)
            self.stride_stages[1] = 1
        else:
            self.stride_stages[1] = 2

        blocks = []
        input_channel = self.out_channel[0]

        stage_num = 1
        _block_index = 1
        self.stage_out_idx = []
        for s, act_func, n_block in zip(self.stride_stages[1:], self.act_stages[1:], self.depth[1:]):
            ks_list = self.kernel_size[stage_num]
            expand_ratio = self.expand_ratio[stage_num]
            output_channel = self.out_channel[stage_num]
            stage_num += 1
            for i in range(n_block):
                if i == 0:
                    stride = s
                else:
                    stride = 1
                mobile_inverted_conv = DynamicBasicBlock(
                    in_channel_list=input_channel, out_channel_list=output_channel, kernel_size_list=ks_list,
                    expand_ratio_list=expand_ratio, stride=stride, act_func=act_func,
                    KERNEL_TRANSFORM_MODE=kernel_transform, divisor=self.divisor, IS_STAGE_BLOCK=(i == 0),
                    normalize=normalize)
                blocks.append(mobile_inverted_conv)
                input_channel = output_channel
                _block_index += 1
            self.stage_out_idx.append(_block_index - 2)

        self.blocks = nn.ModuleList(blocks)
        if self.task == 'classification':
            self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
            self.classifier = DynamicLinearBlock(
                in_features_list=input_channel, out_features_list=num_classes, bias=True,
                dropout_rate=dropout_rate)

        # self.gen_dynamic_module_list(dynamic_types=[DynamicConvBlock, DynamicBasicBlock, DynamicLinearBlock])
        self.init_model()
        # zero_last gamma is used in default
        if zero_last_gamma:
            self.zero_last_gamma()

        # It's IMPORTANT when you want to freeze part of your backbone.
        # ALWAYS remember freeze layers in __init__ to avoid passing freezed params to optimizer
        if self.task == 'detection':
            self.freeze_layer()

    @staticmethod
    def name():
        return 'BigResNetBasic'

    @property
    def module_str(self):
        _str = self.first_conv.module_str + '\n'
        if self.use_maxpool:
            _str += 'max_pool' + '\n'

        for idx, block in enumerate(self.blocks):
            if block.active_block:
                _str += block.module_str + '\n'

        if self.task == 'classification':
            _str += 'avg_pool' + '\n'
            _str += self.classifier.module_str + '\n'
        return _str

    def forward(self, input):
        x = input['image'] if isinstance(input, dict) else input
        outs = []
        # first conv
        x = self.first_conv(x)
        outs.append(x)
        if self.use_maxpool:
            x = self.max_pool(x)

        # blocks
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.stage_out_idx:
                outs.append(x)

        if self.task == 'classification':
            x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x

        features = [outs[i] for i in self.out_layers]
        return {'features': features, 'strides': self.get_outstrides()}

    def freeze_layer(self):
        """
        Freezing layers during training.
        """
        layers = [self.first_conv]
        start_idx = 0
        for stage_out_idx in self.stage_out_idx:
            end_idx = stage_out_idx + 1
            stage = [self.blocks[i] for i in range(start_idx, end_idx)]
            layers.append(nn.Sequential(*stage))
            start_idx = end_idx

        for layer_idx in self.frozen_layers:
            layer = layers[layer_idx]
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        """
        Sets the module in training mode.
        This has any effect only on modules such as Dropout or BatchNorm.

        Returns:
            - module: self
        """
        self.training = mode
        for module in self.children():
            module.train(mode)
        self.freeze_layer()
        return self

    def get_outplanes(self):
        """
        Get dimensions of the output tensors.

        Returns:
            - out (:obj:`list` of :obj:`int`)
        """
        out_channel = [max(i) if isinstance(i, list) else i
                       for i in self.curr_subnet_settings['out_channel']]
        self.out_planes = [out_channel[i] for i in self.out_layers]
        return self.out_planes

    def get_outstrides(self):
        """
        Get strides of output tensors w.r.t inputs.

        Returns:
            - out (:obj:`list` of :obj:`int`)
        """
        return self.out_strides

    def build_active_subnet(self, subnet_settings):
        subnet = bignas_resnet_basic(act_stages=self.act_stages, stride_stages=self.stride_stages,
                                     use_maxpool=self.use_maxpool, dropout_rate=self.dropout_rate, divisor=self.divisor,
                                     frozen_layers=self.frozen_layers, out_layers=self.out_layers,
                                     deep_stem=self.deep_stem,
                                     normalize=self.normalize,
                                     out_strides=self.out_strides,
                                     gray=self.gray,
                                     task=self.task, num_classes=self.num_classes,
                                     out_channel=subnet_settings['out_channel'],
                                     depth=subnet_settings['depth'],
                                     kernel_size=subnet_settings['kernel_size'],
                                     expand_ratio=subnet_settings['expand_ratio'])
        return subnet


def big_resnet_basic(*args, **kwargs):
    return BigResNetBasic(*args, **kwargs)


def bignas_resnet_basic(**kwargs):
    return BignasResNetBasic(**kwargs)


class BignasResNetBasic(nn.Module):

    def __init__(self,
                 # search settings
                 out_channel=[64, 64, 128, 256, 512],
                 depth=[1, 2, 2, 2, 2],
                 kernel_size=[7, 3, 3, 3, 3],
                 expand_ratio=[0, 1, 1, 1, 1],
                 # other settings
                 gray=False,
                 stride_stages=[2, 1, 2, 2, 2],
                 act_stages=['relu', 'relu', 'relu', 'relu', 'relu'],
                 dropout_rate=0.,
                 divisor=8,
                 use_maxpool=True,
                 # bn and initializer
                 normalize={'type': 'dynamic_solo_bn'},
                 initializer={'method': 'msra'},
                 # configuration for task
                 frozen_layers=[],
                 out_layers=[],
                 out_strides=[],
                 deep_stem=False,
                 task='classification',
                 num_classes=1000):
        r"""
        Arguments:

        - out_channel (:obj:`list` of 5 (stages+1) ints): channel list
        - depth (:obj:`list` of 5 (stages+1) ints): depth list for stages
        - stride_stages (:obj:`list` of 5 (stages+1) ints): stride list for stages
        - kernel_size (:obj:`list` of 8 (blocks+1) ints): kernel size list for blocks
        - expand_ratio (:obj:`list` of 8 (blocks+1) ints): expand ratio list for blocks
        - act_stages(:obj:`list` of 8 (blocks+1) ints): activation list for blocks
        - dropout_rate (:obj:`float`): dropout rate
        - divisor(:obj:`int`): divisor for channels
        - use_maxpool(: obj:`bool`): use max_pooling to downsample or not
        - normalize (:obj:`dict`): configurations for normalize
        - initializer (:obj:`dict`): initializer method
        - frozen_layers (:obj:`list` of :obj:`int`): Index of frozen layers, [0,4]
        - out_layers (:obj:`list` of :obj:`int`): Index of output layers, [0,4]
        - out_strides (:obj:`list` of :obj:`int`): Stride of outputs
        - task (:obj:`str`): Type of task, 'classification' or object 'detection'
        - num_classes (:obj:`int`): number of classification classes
        """

        super(BignasResNetBasic, self).__init__()
        global NormLayer
        assert isinstance(normalize, dict) and 'type' in normalize
        norm_split = normalize['type'].split('_', 1)
        if norm_split[0] == 'dynamic':
            normalize['type'] = norm_split[-1]
        NormLayer = build_norm_layer(num_features=None, cfg=normalize, layer_instance=False)

        input_channel = 1 if gray else 3
        self.depth = depth
        self.out_channel = out_channel
        self.kernel_size = get_same_length(kernel_size, self.depth)
        self.expand_ratio = get_same_length(expand_ratio, self.depth)

        self.act_stages = act_stages
        self.stride_stages = stride_stages
        self.dropout_rate = dropout_rate
        self.use_maxpool = use_maxpool
        self.divisor = divisor

        self.frozen_layers = frozen_layers
        self.out_layers = out_layers
        self.out_strides = out_strides

        if len(self.out_layers) != len(self.out_strides):
            self.out_layers = [int(math.log2(i)) - 1 for i in self.out_strides]

        self.task = task
        self.num_classes = num_classes
        self.deep_stem = deep_stem
        self.out_planes = [int(self.out_channel[i]) for i in self.out_layers]

        # first conv layer
        if self.deep_stem:
            self.first_conv = StemBlock(
                in_channel=input_channel, out_channel=self.out_channel[0], kernel_size=self.kernel_size[0],
                expand_ratio=self.expand_ratio[0],
                stride=stride_stages[0], act_func=act_stages[0], NormLayer=NormLayer)
        else:
            self.first_conv = ConvBlock(
                in_channel=input_channel, out_channel=self.out_channel[0], kernel_size=self.kernel_size[0],
                stride=stride_stages[0], act_func=act_stages[0], NormLayer=NormLayer)

        blocks = []
        input_channel = self.out_channel[0]

        if self.use_maxpool:
            self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)
            self.stride_stages[1] = 1
        else:
            self.stride_stages[1] = 2

        _block_index = 1
        self.stage_out_idx = []
        for s, act_func, n_block, output_channel in zip(self.stride_stages[1:], self.act_stages[1:], self.depth[1:],
                                                        self.out_channel[1:]):
            for i in range(n_block):
                kernel_size = self.kernel_size[_block_index]
                expand_ratio = self.expand_ratio[_block_index]
                _block_index += 1
                if i == 0:
                    stride = s
                else:
                    stride = 1
                basic_block = BasicBlock(
                    in_channel=input_channel, out_channel=output_channel, kernel_size=kernel_size,
                    expand_ratio=expand_ratio, stride=stride, act_func=act_func, divisor=self.divisor,
                    NormLayer=NormLayer)
                blocks.append(basic_block)
                input_channel = output_channel
            self.stage_out_idx.append(_block_index - 2)

        self.blocks = nn.ModuleList(blocks)

        if self.task == 'classification':
            self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
            self.classifier = LinearBlock(
                in_features=self.out_channel[-1], out_features=num_classes, bias=True, dropout_rate=dropout_rate)

        if initializer is not None:
            initialize_from_cfg(self, initializer)

        # It's IMPORTANT when you want to freeze part of your backbone.
        # ALWAYS remember freeze layers in __init__ to avoid passing freezed params to optimizer
        self.freeze_layer()

    def forward(self, input):
        x = input['image'] if isinstance(input, dict) else input
        outs = []
        # first conv
        x = self.first_conv(x)
        outs.append(x)
        if self.use_maxpool:
            x = self.max_pool(x)

        # blocks
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.stage_out_idx:
                outs.append(x)

        if self.task == 'classification':
            x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x

        features = [outs[i] for i in self.out_layers]
        return {'features': features, 'strides': self.get_outstrides()}

    def freeze_layer(self):
        """
        Freezing layers during training.
        """
        layers = [self.first_conv]
        start_idx = 0
        for stage_out_idx in self.stage_out_idx:
            end_idx = stage_out_idx + 1
            stage = [self.blocks[i] for i in range(start_idx, end_idx)]
            layers.append(nn.Sequential(*stage))
            start_idx = end_idx

        for layer_idx in self.frozen_layers:
            layer = layers[layer_idx]
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        """
        Sets the module in training mode.
        This has any effect only on modules such as Dropout or BatchNorm.

        Returns:
            - module: self
        """
        self.training = mode
        for module in self.children():
            module.train(mode)
        self.freeze_layer()
        return self

    def get_outplanes(self):
        """
        Get dimensions of the output tensors.

        Returns:
            - out (:obj:`list` of :obj:`int`)
        """
        return self.out_planes

    def get_outstrides(self):
        """
        Get strides of output tensors w.r.t inputs.

        Returns:
            - out (:obj:`list` of :obj:`int`)
        """
        return self.out_strides
