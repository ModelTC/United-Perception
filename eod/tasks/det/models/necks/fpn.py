# Import from third library
import torch
import torch.nn as nn
import torch.nn.functional as F

from eod.utils.model.normalize import build_conv_norm
from eod.utils.model.initializer import initialize_from_cfg
from eod.utils.general.registry_factory import MODULE_ZOO_REGISTRY

__all__ = ['FPN']


@MODULE_ZOO_REGISTRY.register('FPN')
class FPN(nn.Module):
    """
    Feature Pyramid Network

    .. note::

        If num_level is larger than backbone's output feature layers, additional layers will be stacked

    """

    def __init__(self,
                 inplanes,
                 outplanes,
                 start_level,
                 num_level,
                 out_strides,
                 downsample,
                 upsample,
                 normalize=None,
                 tocaffe_friendly=False,
                 initializer=None,
                 align_corners=True,
                 use_p5=False,
                 skip=False):
        """
        Arguments:
            - inplanes (:obj:`list` of :obj:`int`): input channel
            - outplanes (:obj:`list` of :obj:`int`): output channel, all layers are the same
            - start_level (:obj:`int`): start layer of backbone to apply FPN, it's only used for naming convs.
            - num_level (:obj:`int`): number of FPN layers
            - out_strides (:obj:`list` of :obj:`int`): stride of FPN output layers
            - downsample (:obj:`str`): method to downsample, for FPN, it's ``pool``, for RetienaNet, it's ``conv``
            - upsample (:obj:`str`): method to upsample, ``nearest`` or ``bilinear``
            - normalize (:obj:`dict`): config of Normalization Layer
            - initializer (:obj:`dict`): config for model parameter initialization

        `FPN example <http://gitlab.bj.sensetime.com/project-spring/pytorch-object-detection/blob/
        master/configs/baselines/faster-rcnn-R50-FPN-1x.yaml#L75-82>`_
        """

        super(FPN, self).__init__()

        assert downsample in ['pool', 'conv'], downsample
        assert isinstance(inplanes, list)
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.outstrides = out_strides
        self.start_level = start_level
        self.num_level = num_level
        self.downsample = downsample
        self.upsample = upsample
        self.tocaffe_friendly = tocaffe_friendly
        if upsample == 'nearest':
            align_corners = None
        self.align_corners = align_corners
        self.use_p5 = use_p5
        self.skip = skip
        assert num_level == len(out_strides)

        for lvl_idx in range(num_level):
            if lvl_idx < len(inplanes):
                planes = inplanes[lvl_idx]
                self.add_module(
                    self.get_lateral_name(lvl_idx),
                    build_conv_norm(planes, outplanes, 1, normalize=normalize))
                self.add_module(
                    self.get_pconv_name(lvl_idx),
                    build_conv_norm(outplanes, outplanes, kernel_size=3,
                                    stride=1, padding=1, normalize=normalize))
            else:
                if self.downsample == 'pool':
                    self.add_module(
                        self.get_downsample_name(lvl_idx),
                        nn.MaxPool2d(kernel_size=1, stride=2, padding=0))  # strange pooling
                else:
                    self.add_module(
                        self.get_downsample_name(lvl_idx),
                        nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=2, padding=1))
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
        laterals = [self.get_lateral(i)(features[i]) for i in range(len(self.inplanes))]
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
        return torch.tensor(self.outstrides, dtype=torch.int)
