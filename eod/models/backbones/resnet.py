# Standard Library
from collections.abc import Iterable

# Import from third library
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from eod.utils.model.initializer import initialize_from_cfg
from eod.utils.model.normalize import build_norm_layer

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnet_custom']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 normalize={'type': 'solo_bn'},
                 stride_in_1x1=False):
        super(BasicBlock, self).__init__()

        self.norm1_name, norm1 = build_norm_layer(planes, normalize, 1)
        self.norm2_name, norm2 = build_norm_layer(planes, normalize, 2)

        self.conv1 = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False)

        self.add_module(self.norm1_name, norm1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.add_module(self.norm2_name, norm2)
        self.downsample = downsample
        self.stride = stride

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 normalize={'type': 'solo_bn'},
                 stride_in_1x1=False):
        super(Bottleneck, self).__init__()

        self.norm1_name, norm1 = build_norm_layer(planes, normalize, 1)
        self.norm2_name, norm2 = build_norm_layer(planes, normalize, 2)
        self.norm3_name, norm3 = build_norm_layer(planes * self.expansion, normalize, 3)

        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)

        self.conv1 = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=1,
            stride=stride_1x1,
            bias=False
        )
        self.add_module(self.norm1_name, norm1)

        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride_3x3,
            padding=dilation,
            bias=False,
            dilation=dilation
        )

        self.add_module(self.norm2_name, norm2)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.add_module(self.norm3_name, norm3)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        return getattr(self, self.norm3_name)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.norm3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def make_layer4(inplanes,
                block,
                planes,
                blocks,
                stride=1,
                dilation=1,
                normalize={'type': 'solo_bn'},
                stride_in_1x1=False):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(
                inplanes,
                planes * block.expansion,
                kernel_size=1,
                stride=stride,
                bias=False),
            build_norm_layer(planes * block.expansion, normalize)[1]
        )

    layers = []
    layers.append(block(inplanes, planes, stride, dilation, downsample, normalize, stride_in_1x1))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(block(inplanes, planes, stride=1, dilation=dilation,
                            downsample=None, normalize=normalize, stride_in_1x1=stride_in_1x1))

    return nn.Sequential(*layers)


class ResNet(nn.Module):
    """
        layer0 <-> Conv1, ..., layer4 <-> Conv5

        You can configure output layer and its related strides.
    """
    def __init__(self,
                 block,
                 layers,
                 out_layers,
                 out_strides,
                 style='pytorch',
                 frozen_layers=[],
                 layer_deform=[0, 0, 0, 0, 0],
                 normalize={'type': 'freeze_bn'},
                 multiplier=1,
                 checkpoint=False,
                 initializer=None,
                 fa=None,
                 nnie=False,
                 deep_stem=False,
                 input_channel=3):
        r"""
        Arguments:
            - frozen_layers (:obj:`list` of :obj:`int`): Index of frozen layers, [0,4]
            - out_layers (:obj:`list` of :obj:`int`): Index of output layers, [0,4]
            - out_strides (:obj:`list` of :obj:`int`): Stride of outputs
            - style (:obj:`str`): ResNet style (``caffe`` or ``pytorch``), default is ``pytorch`` style.
            - layer_deform (:obj:`list` of (:obj:`str` or None): ``DCN`` setting for each layer. See note below
            - normalize (:obj:`dict`): Config of Normalization Layer (see Configuration#Normalization).
            - checkpoint (:obj:`list` or :obj:`bool`): segments to checkpoint for each layer.
              ``False`` or ``0`` for no checkpoint.
              For more details, refer to `Checkpoint <https://pytorch.org/docs/stable/checkpoint.html>`_
            - initializer (:obj:`dict`): Config of initilizaiton
            - fa (:obj:`dict`): Configurations of `FactorizedAttentionBlock <https://arxiv.org/pdf/1812.01243.pdf>`_.
              `Example <http://gitlab.bj.sensetime.com/project-spring/pytorch-object-detection/
              blob/master/configs/usages/retinanet-R50-GN-FA.yaml#L67-68>`_

        `ResNet example <http://gitlab.bj.sensetime.com/project-spring/pytorch-object-detection/blob/
        master/configs/baselines/faster-rcnn-R50-FPN-1x.yaml#L61-66>`_

        .. note::

            We support two style ResNet implementation: ``pytorch`` & ``caffe`` style ResNet. The differences between
            them are as follows:

                1. Architecture: ``pytorch`` style ResNet strides in conv3x3 while ``caffe`` style ResNet
                   strides in conv1x1.
                2. Data preprocess: they use different ``pixel_mean`` & ``pixel_std`` in data preprocessing.
                3. Thus, their pretrained weights are different.
                4. ``caffe`` style ResNet support ``caffe_freeze_bn`` only, since the ``running_mean`` and
                   ``running_var`` are absorbed in ``weight`` and ``bias`` in the pretrained checkpoint.

            Here are the configurations needed to be modified when migrating from pytorch style to caffe style ResNet

            .. code-block:: yaml

                dataset:
                    # ...
                    preprocess_style: caffe
                    pixel_mean: [102.9801, 115.9465, 122.7717]  # caffe-style pretrained statistics
                    pixel_std: [1, 1, 1]

                saver:
                    # ...
                    pretrain_model: /mnt/lustre/share/DSK/model_zoo/pytorch/imagenet/R-50-caffe-style.pkl

                net:
                    backbone:
                        # ...
                        normalize:
                            type: caffe_freeze_bn
                        style: caffe
                    # ...
                # ...

        .. seealso::

            Checkpoint purposes on saving GPU memory. It is mplemented by
            rerunning a forward-pass segment for each checkpointed segment during backward.
            Some modules need to adjust their paramters because of checkpointing, such as BN.
            You need change BN momentum to :math:`momentum_{new} = 1 - \sqrt{1 - momentum_{old}}`
            if checkpoint enabled.
            For mote details, refer to `checkpoint <https://pytorch.org/docs/stable/checkpoint.html>`_


        .. note::

            Elements of layer_deform support three types: **False**, **last**, **all**, **int**

            * last: only last conv3x3 use Deformable Conv for this layer
            * all: all conv3x3 use Deformable Conv for this layer
            * False: disable Deformable Conv for this layer
            * int(n): last n conv3x3 use Deformable Conv for this layer

            .. code-block:: python

                layer_deform = [False, False, 'last', 1, 'all']

        """

        def check_range(x):
            if x:             # Add conditional operation to avoid error when no frozen layer is provided
                assert min(x) >= 0 and max(x) <= 4, x

        check_range(frozen_layers)
        check_range(out_layers)
        assert len(out_layers) == len(out_strides)

        if style == 'caffe':
            stride_in_1x1 = True
            # assert normalize['type'] == 'caffe_freeze_bn', "Caffe style pretrained only supports 'caffe_freeze_bn'"
        else:
            stride_in_1x1 = False
        self.multiplier = float(multiplier)
        self.frozen_layers = frozen_layers
        self.out_layers = out_layers
        self.out_strides = out_strides
        layer_out_planes = [64 * self.multiplier] + [i * block.expansion * self.multiplier for i in [64, 128, 256, 512]]
        layer_in_planes = list(map(int, [64 * self.multiplier] + [i * self.multiplier for i in [64, 128, 256, 512]]))
        layer_out_planes = list(map(int, layer_out_planes))
        self.out_planes = [layer_out_planes[i] for i in out_layers]
        self.segments = self.get_segments(checkpoint)

        self.inplanes = layer_out_planes[0]
        self.norm1_name, norm1 = build_norm_layer(layer_out_planes[0], normalize, 1)

        super(ResNet, self).__init__()
        if not deep_stem:
            self.conv1 = nn.Conv2d(input_channel, layer_out_planes[0], kernel_size=7, stride=2, padding=3, bias=False)
        else:
            inner_planes = layer_out_planes[0] // 2
            self.conv1 = nn.Sequential(
                nn.Conv2d(input_channel, inner_planes, kernel_size=3, stride=2, padding=1, bias=False),
                build_norm_layer(inner_planes, normalize)[1],
                nn.ReLU(inplace=True),
                nn.Conv2d(inner_planes, inner_planes, kernel_size=3, stride=1, padding=1, bias=False),
                build_norm_layer(inner_planes, normalize)[1],
                nn.ReLU(inplace=True),
                nn.Conv2d(inner_planes, layer_out_planes[0], kernel_size=3, stride=1, padding=1, bias=False),
            )
        self.add_module(self.norm1_name, norm1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        if nnie:
            self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)
        self.layer0 = nn.Sequential(self.conv1, self.norm1, self.relu, self.maxpool)
        self.layer1 = self._make_layer(block, layer_in_planes[1], layers[0], normalize=normalize)
        self.layer2 = self._make_layer(block, layer_in_planes[2], layers[1], stride=2,
                                       normalize=normalize,
                                       stride_in_1x1=stride_in_1x1)

        self.layer3 = self._make_layer(block, layer_in_planes[3], layers[2], stride=2,
                                       normalize=normalize,
                                       stride_in_1x1=stride_in_1x1)
        if 4 in self.out_layers:
            layer4_stride = out_strides[-1] // 16
            self.layer4 = self._make_layer(block, layer_in_planes[4], layers[3],
                                           stride=layer4_stride,
                                           dilation=2 // layer4_stride,
                                           normalize=normalize,
                                           stride_in_1x1=stride_in_1x1)
        else:
            self.layer4 = None

        if initializer is not None:
            initialize_from_cfg(self, initializer)

        # It's IMPORTANT when you want to freeze part of your backbone.
        # ALWAYS remember freeze layers in __init__ to avoid passing freezed params
        # to optimizer
        self.freeze_layer()

    def get_segments(self, checkpoint):
        if isinstance(checkpoint, Iterable):
            segments = [int(x) for x in checkpoint]
        else:
            segments = [int(checkpoint)] * 5
        return segments

    def get_fa_layers(self, num_channels, fa_cfg):
        """Build fa blocks in backbone"""
        fa_modules = [None] * 5
        return fa_modules

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    stride=1,
                    dilation=1,
                    normalize={'type': 'solo_bn'},
                    stride_in_1x1=False):
        block_types = [block] * blocks

        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                build_norm_layer(planes * block.expansion, normalize)[1]
            )

        layers = []
        layers.append(block_types[0](self.inplanes,
                                     planes,
                                     stride,
                                     dilation,
                                     downsample,
                                     normalize,
                                     stride_in_1x1))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block_types[i](self.inplanes,
                                         planes,
                                         stride=1,
                                         dilation=dilation,
                                         downsample=None,
                                         normalize=normalize,
                                         stride_in_1x1=stride_in_1x1))
        return nn.Sequential(*layers)

    def get_outplanes(self):
        """
        get dimension of the output tensor
        """
        return self.out_planes

    def get_outstrides(self):
        """
        Returns:

            - out (:obj:`int`): number of channels of output
        """

        return torch.tensor(self.out_strides, dtype=torch.int)

    def forward(self, input):
        """

        Arguments:
            - input (:obj:`dict`): output of
              :class:`~root.datasets.base_dataset.BaseDataset`

        Returns:
            - out (:obj:`dict`):

        Output example::

            {
                'features': [], # list of tenosr
                'strides': []   # list of int
            }
        """
        x = input['image']
        outs = []
        for layer_idx in range(0, 5):
            layer = getattr(self, f'layer{layer_idx}', None)
            if layer is not None:  # layer4 is None for C4 backbone
                # Use checkpoint for learnable layer
                if self.segments[layer_idx] > 0 and layer_idx not in self.frozen_layers:
                    x = self.checkpoint_fwd(layer, x, self.segments[layer_idx])
                else:
                    x = layer(x)
                outs.append(x)

        features = [outs[i] for i in self.out_layers]
        return {'features': features, 'strides': self.get_outstrides()}

    def freeze_layer(self):
        layers = [
            nn.Sequential(self.conv1, self.norm1, self.relu, self.maxpool),
            self.layer1, self.layer2, self.layer3, self.layer4
        ]
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
            Module: self
        """
        self.training = mode
        for module in self.children():
            module.train(mode)
        self.freeze_layer()
        return self


def resnet18(pretrained=False, **kwargs):
    """
    Constructs a ResNet-18 model.

    Arguments:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """
    Constructs a ResNet-34 model.

    Arguments:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """
    Constructs a ResNet-50 model.

    Arguments:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """
    Constructs a ResNet-101 model.

    Arguments:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """
    Constructs a ResNet-152 model.

    Arguments:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


def resnet_custom(**kwargs):
    assert 'block' in kwargs and 'layers' in kwargs, 'Require block and layers'
    block = kwargs.pop('block')
    layers = kwargs.pop('layers')
    if block == 'basic':
        block = BasicBlock
    elif block == 'bottleneck':
        block = Bottleneck
    else:
        raise Exception('Unsupported block type.')
    model = ResNet(block, layers, **kwargs)
    return model
