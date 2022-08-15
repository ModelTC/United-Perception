import math
import torch.nn as nn

from up.tasks.nas.bignas.models.ops.dynamic_blocks import DynamicConvBlock, DynamicLinearBlock, \
    DynamicRegBottleneckBlock
from up.tasks.nas.bignas.models.search_space.base_bignas_searchspace import BignasSearchSpace
from up.utils.model.initializer import initialize_from_cfg
from up.tasks.nas.bignas.models.ops.normal_blocks import ConvBlock, LinearBlock, RegBottleneckBlock, \
    get_same_length
# from up.tasks.nas.bignas.models.ops.dynamic_utils import build_norm_layer
from up.utils.model.normalize import build_norm_layer


big_regnet_kwargs = {
    'out_channel': {
        'space': {
            'min': [16, 16, 32, 128, 256],
            'max': [48, 48, 96, 256, 480],
            'stride': [16, 16, 16, 16, 32]},
        'sample_strategy': 'ordered_stage_wise'},
    'kernel_size': {
        'space': {
            'min': [3, 3, 3, 3, 3],
            'max': [3, 3, 3, 3, 3],
            'stride': 2},
        'sample_strategy': 'stage_wise'},
    'expand_ratio': [0, 1, 1, 1, 1],
    'depth': {
        'space': {
            'min': [1, 1, 1, 2, 5],
            'max': [1, 3, 3, 5, 8],
            'stride': 1},
        'sample_strategy': 'stage_wise_depth'},
    'group_width': {
        'space': {
            'min': [4, 4, 4, 4, 4],
            'max': [16, 16, 16, 16, 16],
            'dynamic_range': [[4, 8, 16], [4, 8, 16], [4, 8, 16], [4, 8, 16], [4, 8, 16]]},
        'sample_strategy': 'net_wise'},
}


class BigRegNet(BignasSearchSpace):

    def __init__(self,
                 # stage wise config
                 act_stages=['relu', 'relu', 'relu', 'relu', 'relu'],
                 se_stages=[False, False, False, False, False],
                 stride_stages=[2, 2, 2, 2, 2],
                 se_act_func1='relu', se_act_func2='sigmoid',
                 dropout_rate=0.1, divisor=8,
                 # clarify settings for detection task
                 normalize={'type': 'dynamic_solo_bn'},
                 frozen_layers=[],
                 out_layers=[],
                 out_strides=[],
                 task='classification',
                 num_classes=1000,
                 # search setting
                 kernel_transform=False,
                 zero_last_gamma=True,
                 # search space
                 **kwargs):
        super(BigRegNet, self).__init__(**kwargs)

        r"""
        Arguments:
        - act_stages(:obj:`list` of 5 (stages+1) ints): activation list for blocks
        - stride_stages (:obj:`list` of 5 (stages+1) ints): stride list for stages
        - se_stages (:obj:`list` of 5 (stages+1) ints): se list for stages
        - se_act_func1(:obj:`str`: first activation function for se block)
        - se_act_func1(:obj:`str`: second activation function for se block)
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
        - group_width (:obj:`list` or `dict`): stage-wise search settings for group width
        """

        self.act_stages = act_stages
        self.stride_stages = stride_stages
        self.se_stages = se_stages
        self.se_act_func1 = se_act_func1
        self.se_act_func2 = se_act_func2
        self.dropout_rate = dropout_rate
        self.divisor = divisor

        self.frozen_layers = frozen_layers
        self.out_layers = out_layers
        self.out_strides = out_strides
        if len(self.out_layers) != len(self.out_strides):
            self.out_layers = [int(math.log2(i)) - 1 for i in self.out_strides]

        self.task = task
        self.num_classes = num_classes
        self.normalize = normalize

        self.kernel_size = self.normal_settings['kernel_size']
        self.out_channel = self.normal_settings['out_channel']
        self.depth = self.normal_settings['depth']
        self.expand_ratio = self.normal_settings['expand_ratio']
        self.group_width = self.normal_settings['group_width']

        # first conv layer
        self.first_conv = DynamicConvBlock(
            in_channel_list=3, out_channel_list=self.out_channel[0], kernel_size_list=self.kernel_size[0],
            stride=self.stride_stages[0], act_func=self.act_stages[0], KERNEL_TRANSFORM_MODE=kernel_transform,
            normalize=normalize)

        # inverted residual blocks
        input_channel = self.out_channel[0]

        blocks = []
        stage_num = 1
        self.stage_out_idx = []
        _block_idx = 0
        for s, act_func, use_se, n_block in zip(self.stride_stages[1:], self.act_stages[1:],
                                                self.se_stages[1:], self.depth[1:]):
            kernel_size_list = self.kernel_size[stage_num]
            expand_ratio_list = self.expand_ratio[stage_num]
            out_channel_list = self.out_channel[stage_num]
            group_width_list = self.group_width[stage_num]
            stage_num += 1
            for i in range(n_block):
                if i == 0:
                    stride = s
                else:
                    stride = 1
                mobile_inverted_conv = DynamicRegBottleneckBlock(
                    input_channel, out_channel_list, kernel_size_list,
                    expand_ratio_list=expand_ratio_list, group_width_list=group_width_list,
                    stride=stride, act_func=act_func, act_func1=self.se_act_func1,
                    act_func2=self.se_act_func2, use_se=use_se, KERNEL_TRANSFORM_MODE=kernel_transform,
                    divisor=self.divisor, normalize=normalize)
                blocks.append(mobile_inverted_conv)
                input_channel = out_channel_list
                _block_idx += 1
            self.stage_out_idx.append(_block_idx - 1)

        out_channel = [max(i) if isinstance(i, list) else i
                       for i in self.out_channel]
        self.out_planes = [out_channel[i] for i in self.out_layers]

        self.blocks = nn.ModuleList(blocks)
        if self.task == 'classification':
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = DynamicLinearBlock(input_channel, num_classes, bias=True, dropout_rate=dropout_rate)

        # self.gen_dynamic_module_list(dynamic_types=[DynamicConvBlock, DynamicRegBottleneckBlock, DynamicLinearBlock])
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
        return 'BigRegNet'

    def forward(self, input):
        x = input['image'] if isinstance(input, dict) else input
        outs = []
        x = self.first_conv(x)
        outs.append(x)

        # blocks
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.stage_out_idx:
                outs.append(x)

        if self.task == 'classification':
            # classifier
            x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x
        features = [outs[i] for i in self.out_layers]
        return {'features': features, 'strides': self.get_outstrides()}

    @property
    def module_str(self):
        _str = self.first_conv.module_str + '\n'

        for block in self.blocks:
            if block.active_block:
                _str += block.module_str + '\n'

        if self.task == 'classification':
            _str += 'avg_pool' + '\n'
            _str += self.classifier.module_str + '\n'
        return _str

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
        subnet = bignas_regnet(act_stages=self.act_stages, stride_stages=self.stride_stages,
                               se_stages=self.se_stages, se_act_func1=self.se_act_func1,
                               se_act_func2=self.se_act_func2,
                               dropout_rate=self.dropout_rate, divisor=self.divisor,
                               frozen_layers=self.frozen_layers, out_layers=self.out_layers,
                               out_strides=self.out_strides,
                               normalize=self.normalize,
                               task=self.task, num_classes=self.num_classes,
                               out_channel=subnet_settings['out_channel'],
                               depth=subnet_settings['depth'],
                               kernel_size=subnet_settings['kernel_size'],
                               expand_ratio=subnet_settings['expand_ratio'],
                               group_width=subnet_settings['group_width'])

        return subnet


def big_regnet(*args, **kwargs):
    return BigRegNet(*args, **kwargs)


def bignas_regnet(**kwargs):
    return BignasRegNet(**kwargs)


class BignasRegNet(nn.Module):
    def __init__(self,
                 num_classes=1000,
                 # search settings
                 out_channel=[8, 8, 16, 48, 224],
                 depth=[1, 2, 2, 1, 2],
                 kernel_size=[7, 3, 3, 3, 3],
                 expand_ratio=[0, 1, 1, 1, 1],
                 group_width=[8, 8, 8, 8, 8],
                 # other settings
                 stride_stages=[2, 2, 2, 2, 2],
                 se_stages=[False, False, False, False, False],
                 act_stages=['relu', 'relu', 'relu', 'relu', 'relu'],
                 dropout_rate=0.1,
                 divisor=8,
                 se_act_func1='relu',
                 se_act_func2='sigmoid',
                 # bn and initializer
                 normalize={'type': 'dynamic_solo_bn'},
                 initializer={'method': 'msra'},
                 # configuration for task
                 frozen_layers=[],
                 out_layers=[],
                 out_strides=[],
                 task='classification'):
        super(BignasRegNet, self).__init__()

        r"""
        Arguments:
        - out_channel (:obj:`list` of 5 (stages+1) ints): channel list
        - depth (:obj:`list` of 5 (stages+1) ints): depth list for stages
        - kernel_size (:obj:`list` of 5 (stages+1) or 8 (blocks+1) ints): kernel size list for blocks
        - expand_ratio (:obj:`list` of 5 (stages+1) or 8 (blocks+1) ints): expand ratio list for blocks
        - group_width (:obj:`list` of 5 (stages+1) or 8 (blocks+1) ints): group width list for blocks
        - act_stages(:obj:`list` of 5 (stages+1) ints): activation list for blocks
        - stride_stages (:obj:`list` of 5 (stages+1) ints): stride list for stages
        - se_stages (:obj:`list` of 5 (stages+1) ints): se list for stages
        - se_act_func1(:obj:`str`: first activation function for se block)
        - se_act_func1(:obj:`str`: second activation function for se block)
        - dropout_rate (:obj:`float`): dropout rate
        - divisor(:obj:`int`): divisor for channels
        - normalize (:obj:`dict`): configurations for normalize
        - initializer (:obj:`dict`): initializer method
        - frozen_layers (:obj:`list` of :obj:`int`): Index of frozen layers, [0,4]
        - out_layers (:obj:`list` of :obj:`int`): Index of output layers, [0,4]
        - out_strides (:obj:`list` of :obj:`int`): Stride of outputs
        - task (:obj:`str`): Type of task, 'classification' or object 'detection'
        - num_classes (:obj:`int`): number of classification classes
        """

        self.frozen_layers = frozen_layers
        self.out_layers = out_layers
        self.out_strides = out_strides
        self.task = task

        global NormLayer

        assert isinstance(normalize, dict) and 'type' in normalize
        norm_split = normalize['type'].split('_', 1)
        if norm_split[0] == 'dynamic':
            normalize['type'] = norm_split[-1]
        NormLayer = build_norm_layer(num_features=None, cfg=normalize, layer_instance=False)

        self.depth = depth
        self.out_channel = out_channel
        self.kernel_size = get_same_length(kernel_size, depth)
        self.expand_ratio = get_same_length(expand_ratio, depth)
        self.group_width = get_same_length(group_width, depth)

        self.se_stages = se_stages
        self.stride_stages = stride_stages
        self.act_stages = act_stages
        self.se_act_func1 = se_act_func1
        self.se_act_func2 = se_act_func2
        self.divisor = divisor

        # first conv layer
        self.first_conv = ConvBlock(
            in_channel=3, out_channel=self.out_channel[0], kernel_size=self.kernel_size[0],
            stride=self.stride_stages[0], act_func=self.act_stages[0], NormLayer=NormLayer)

        # inverted residual blocks
        input_channel = self.out_channel[0]

        blocks = []
        stage_num = 1
        self.stage_out_idx = []
        _block_index = 1
        for s, act_func, use_se, n_block in zip(self.stride_stages[1:], self.act_stages[1:],
                                                self.se_stages[1:], self.depth[1:]):
            out_channel = self.out_channel[stage_num]
            stage_num += 1
            for i in range(n_block):
                kernel_size = self.kernel_size[_block_index]
                expand_ratio = self.expand_ratio[_block_index]
                group_width = self.group_width[stage_num]
                _block_index += 1
                if i == 0:
                    stride = s
                else:
                    stride = 1
                mobile_inverted_conv = RegBottleneckBlock(
                    input_channel, out_channel, kernel_size, expand_ratio=expand_ratio, group_width=group_width,
                    stride=stride, act_func=act_func, act_func1=self.se_act_func1,
                    act_func2=self.se_act_func2, use_se=use_se, divisor=self.divisor, NormLayer=NormLayer)
                blocks.append(mobile_inverted_conv)
                input_channel = out_channel
            self.stage_out_idx.append(_block_index - 2)

        self.out_planes = [self.out_channel[i] for i in self.out_layers]

        self.blocks = nn.ModuleList(blocks)
        if self.task == 'classification':
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = LinearBlock(input_channel, num_classes, bias=True, dropout_rate=dropout_rate)

        if initializer is not None:
            initialize_from_cfg(self, initializer)

        # It's IMPORTANT when you want to freeze part of your backbone.
        # ALWAYS remember freeze layers in __init__ to avoid passing freezed params to optimizer
        self.freeze_layer()

    def forward(self, input):
        x = input['image'] if isinstance(input, dict) else input
        outs = []
        x = self.first_conv(x)
        outs.append(x)

        # blocks
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.stage_out_idx:
                outs.append(x)

        if self.task == 'classification':
            # classifier
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


def bignas_regnetx_200m(**kwargs):
    net_kwargs = {
        'out_channel': [32, 48, 96, 240, 528],
        'depth': [1, 1, 3, 5, 8],
        'group_width': [24, 24, 24, 24, 24]
    }
    net_kwargs.update(**kwargs)
    return BignasRegNet(**net_kwargs)


def bignas_regnetx_400m(**kwargs):
    net_kwargs = {
        'out_channel': [32, 32, 64, 160, 384],
        'depth': [1, 1, 2, 7, 12],
        'group_width': [16, 16, 16, 16, 16]
    }
    net_kwargs.update(**kwargs)
    return BignasRegNet(**net_kwargs)


def bignas_regnetx_600m(**kwargs):
    net_kwargs = {
        'out_channel': [32, 48, 96, 240, 528],
        'depth': [1, 1, 3, 5, 8],
        'group_width': [24, 24, 24, 24, 24]
    }
    net_kwargs.update(**kwargs)
    return BignasRegNet(**net_kwargs)


def bignas_regnetx_800m(**kwargs):
    net_kwargs = {
        'out_channel': [32, 64, 128, 288, 672],
        'depth': [1, 1, 3, 7, 5],
        'group_width': [16, 16, 16, 16, 16]
    }
    net_kwargs.update(**kwargs)
    return BignasRegNet(**net_kwargs)
