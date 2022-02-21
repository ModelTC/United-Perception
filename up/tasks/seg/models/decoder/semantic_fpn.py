import torch
import torch.nn as nn
from torch.nn import functional as F
import warnings
import math

from up.utils.model.normalize import build_conv_norm
from up.utils.model.initializer import initialize_from_cfg
from up.utils.general.registry_factory import MODULE_ZOO_REGISTRY
from up.models.losses import build_loss


class SemanticNet(nn.Module):
    def __init__(self, inplanes, num_classes):
        super().__init__()
        self.inplanes = inplanes
        self.num_classes = num_classes
        self.tocaffe = False
        self.prefix = self.__class__.__name__

    def predict(self, input):
        raise NotImplementedError

    def to_caffe_export(self, pred):
        if torch.__version__ >= '1.5.0':
            warnings.warn(
                f'using torch {torch.__version__}, softmax will import permute'
            )
            pred = pred.permute(0, 2, 3, 1).contiguous()
            pred = F.softmax(pred, dim=-1)
            pred = pred.permute(0, 3, 1, 2).contiguous()
            return pred
        else:
            return F.softmax(pred, dim=1)

    def interp_to_origin_image(self, pred, input):
        image = input['image']
        if pred.size()[-2:] != image.size()[-2:]:
            warnings.warn(
                f'use default interp to origin image! {pred.size()} vs {image.size()}'
            )
            pred = F.interpolate(pred,
                                 image.size()[-2:],
                                 mode='bilinear',
                                 align_corners=True)
        return pred

    def forward(self, input):
        output = self.predict(input)
        if not self.training and self.tocaffe:
            output[self.prefix + '.blobs.seg'] = self.to_caffe_export(
                output['blob_pred'])

        output['blob_pred'] = self.interp_to_origin_image(
            output['blob_pred'], input)
        return output


@MODULE_ZOO_REGISTRY.register('base_seg_postprocess')
class BaseSegPostProcess(nn.Module):
    def __init__(self, loss, use_preds=False):
        super().__init__()
        self.loss = build_loss(loss)
        self.prefix = self.__class__.__name__
        self.use_preds = use_preds

    def get_loss(self, inputs):
        if self.use_preds:
            preds = inputs['blob_preds']  # aux
        else:
            preds = inputs['blob_pred']
        gt_semantic_seg = inputs['gt_semantic_seg']

        loss = self.loss(preds, gt_semantic_seg.long())
        return {self.prefix + '.loss': loss}

    def forward(self, input):
        return self.get_loss(input)


class UpsampleBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 upsample_factor=2,
                 upsample_method='bilinear',
                 normalize=None):
        super(UpsampleBlock, self).__init__()
        self.kernel_size = kernel_size
        self.upsample_factor = upsample_factor

        self.conv = build_conv_norm(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            normalize=normalize
        )  # initialised, and with relu as activation function

        self.upsample_method = upsample_method
        # if self.upsample_method == 'carafe':
        #     self.upsample = NaiveCarafe(input_channel=out_channels, enlarge_rate=2, normalize=normalize)

    def forward(self, x):
        x = self.conv(x)
        if self.upsample_method == 'none':
            pass
        elif self.upsample_method == 'bilinear':
            x = F.interpolate(x,
                              scale_factor=2,
                              mode='bilinear',
                              align_corners=True)
        # elif self.upsample_method == 'carafe':
        #     x = self.upsample(x)
        else:
            raise NotImplementedError
        return x


@MODULE_ZOO_REGISTRY.register('semantic_fpn')
class SemanticFPN(SemanticNet):
    """
    Panoptic FPN
    """
    def __init__(self,
                 inplanes,
                 num_classes,
                 fpn_strides,
                 fpn_levels=None,
                 upsample_out_channels=256,
                 upsample_method='bilinear',
                 embedding_out_channels=256,
                 normalize=None,
                 initializer=None,
                 require_feat=False):
        super().__init__(inplanes, num_classes)
        if isinstance(inplanes, list):
            assert len(
                inplanes
            ) == 1, 'single input is expected, but found:{} '.format(inplanes)
            inplanes = inplanes[0]
        assert isinstance(inplanes, int)

        self.upsample_out_channels = upsample_out_channels
        self.fpn_levels = fpn_levels
        self.featmap_strides = fpn_strides
        self.num_scales = len(self.featmap_strides)
        self.embedding_out_channels = embedding_out_channels
        self.require_feat = require_feat
        self.upsample_method = upsample_method

        self.upsamples = nn.ModuleList()

        self.upsamples.append(
            build_conv_norm(inplanes,
                            self.upsample_out_channels,
                            kernel_size=3,
                            padding=1,
                            normalize=normalize))

        for i in range(1, self.num_scales):
            num_blocks = int(
                math.log((self.featmap_strides[i] // self.featmap_strides[0]),
                         2))

            blocks = []

            for _ in range(0, num_blocks):
                # outplanes = inplanes if _ != num_blocks - 1 else upsample_out_channels
                if _ != num_blocks - 1:
                    upsample = UpsampleBlock(inplanes,
                                             inplanes,
                                             upsample_method=upsample_method,
                                             normalize=normalize)
                else:
                    # last block with no interp
                    upsample = UpsampleBlock(inplanes,
                                             upsample_out_channels,
                                             upsample_method='none',
                                             normalize=normalize)
                blocks.append(upsample)

            self.upsamples.append(nn.Sequential(*blocks))

        self.conv_logits = nn.Conv2d(self.upsample_out_channels, num_classes,
                                     1)

        # In case semantic feature is required by other module
        if self.require_feat:
            self.conv_embedding = nn.Conv2d(self.upsample_out_channels,
                                            self.embedding_out_channels, 1)

        initialize_from_cfg(self, initializer)

    def interp(self, feat, size, mode=None):
        if mode is None:
            mode = self.upsample_method
        return F.interpolate(feat, size, mode=mode, align_corners=True)

    def predict(self, input):
        x = input['features']
        image_size = input['image'].size()[2:]
        if self.fpn_levels is not None:
            x = [x[i] for i in self.fpn_levels]
        assert len(x) == self.num_scales

        upsampled_feats = []
        for i, upsample in enumerate(self.upsamples):
            feat = upsample(x[i])
            if i > 0:
                feat = self.interp(feat, upsampled_feats[0].size()[2:])
            upsampled_feats.append(feat)

        # Merging features from all levels
        for i in range(self.num_scales - 1, 0, -1):
            upsampled_feats[i - 1] = upsampled_feats[i - 1] + upsampled_feats[i]
        semantic_feat = upsampled_feats[0]

        # Generate prediction.
        semantic_pred = self.conv_logits(semantic_feat)
        semantic_pred = self.interp(semantic_pred, image_size, mode='bilinear')

        output = {
            'blob_pred': semantic_pred.float(),
        }

        # For HTC, semantic feature is required, thus, keep semantic feature if needed
        if self.require_feat:
            semantic_feat = self.conv_embedding(semantic_feat)
            output['semantic_feat'] = semantic_feat

        return output
