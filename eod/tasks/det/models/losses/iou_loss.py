# Standard Library
from eod.utils.general.fp16_helper import to_float32
import math

# Import from third library
import torch

# Import from local
from eod.models.losses.loss import BaseLoss, _reduce
from eod.utils.general.registry_factory import LOSSES_REGISTRY
from eod.tasks.det.models.utils.bbox_helper import offset2bbox, unnormalize_offset
from eod.utils.general.global_flag import ALIGNED_FLAG


__all__ = ['IOULoss', 'ComposeLocLoss']


@to_float32
def iou_overlaps(b1, b2, eps=1e-9, return_giou=False, return_diou=False, return_ciou=False, return_eiou=False):
    """
    Arguments:
        - b1: dts, [n, >=4] (x1, y1, x2, y2, ...)
        - b1: gts, [n, >=4] (x1, y1, x2, y2, ...)

    Returns:
        intersection-over-union pair-wise, generalized iou.
    """
    # eps = eps if ALIGNED_FLAG.offset else 0
    area1 = (b1[:, 2] - b1[:, 0] + ALIGNED_FLAG.offset) * (b1[:, 3] - b1[:, 1] + ALIGNED_FLAG.offset)
    area2 = (b2[:, 2] - b2[:, 0] + ALIGNED_FLAG.offset) * (b2[:, 3] - b2[:, 1] + ALIGNED_FLAG.offset)
    iou_dict = {}
    # only for eiou loss
    whb1 = (b1[:, 2:4] - b1[:, :2] + ALIGNED_FLAG.offset).clamp(min=0)
    whb2 = (b2[:, 2:4] - b2[:, :2] + ALIGNED_FLAG.offset).clamp(min=0)
    # only for giou loss
    lt1 = torch.max(b1[:, :2], b2[:, :2])
    rb1 = torch.max(b1[:, 2:4], b2[:, 2:4])
    lt2 = torch.min(b1[:, :2], b2[:, :2])
    rb2 = torch.min(b1[:, 2:4], b2[:, 2:4])
    # en = (lt1 < rb2).type(lt1.type()).prod(dim=1)
    # inter_area = inter_area * en
    wh1 = (rb2 - lt1 + ALIGNED_FLAG.offset).clamp(min=0)
    wh2 = (rb1 - lt2 + ALIGNED_FLAG.offset).clamp(min=0)
    inter_area = wh1[:, 0] * wh1[:, 1]
    union_area = area1 + area2 - inter_area

    iou = inter_area / torch.clamp(union_area, min=1)
    iou_dict['iou'] = iou
    if return_giou:
        # origin esp is 1e-7
        ac_union = wh2[:, 0] * wh2[:, 1] + eps
        giou = iou - (ac_union - union_area) / ac_union
        iou_dict['giou'] = giou
    if return_diou or return_ciou:
        c2 = wh2[:, 0] ** 2 + wh2[:, 1] ** 2 + eps  # convex diagonal squared
        rho2 = ((b2[:, 0] + b2[:, 2] - b1[:, 0] - b1[:, 2]) ** 2 + (b2[:, 1] + b2[:, 3] - b1[:, 1] - b1[:, 3]) ** 2) / 4
        # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
        if return_diou:
            diou = iou - rho2 / c2
            iou_dict['diou'] = diou
        # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
        if return_ciou:
            w1, h1 = b1[:, 2] - b1[:, 0] + ALIGNED_FLAG.offset, b1[:, 3] - b1[:, 1] + ALIGNED_FLAG.offset + eps
            w2, h2 = b2[:, 2] - b2[:, 0] + ALIGNED_FLAG.offset, b2[:, 3] - b2[:, 1] + ALIGNED_FLAG.offset + eps
            v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
            with torch.no_grad():
                alpha = v / ((1 + eps) - iou + v)
            ciou = iou - (rho2 / c2 + v * alpha)
            iou_dict['ciou'] = ciou
    if return_eiou:
        # diou penalty
        c2 = wh2[:, 0] ** 2 + wh2[:, 1] ** 2 + eps  # convex diagonal squared
        rho2 = ((b2[:, 0] + b2[:, 2] - b1[:, 0] - b1[:, 2]) ** 2 + (b2[:, 1] + b2[:, 3] - b1[:, 1] - b1[:, 3]) ** 2) / 4
        eiou = iou - rho2 / c2
        # eiou penalty
        cw2 = wh2[:, 0] ** 2 + eps
        ch2 = wh2[:, 1] ** 2 + eps
        wo2 = (whb2[:, 0] - whb1[:, 0]) ** 2
        ho2 = (whb2[:, 1] - whb1[:, 1]) ** 2
        eiou -= (wo2 / cw2 + ho2 / ch2)
        # imbalance
        iou_dict['eiou'] = eiou * iou
    return iou_dict


def iou_loss(input, target, loss_type='iou', reduction='none', normalizer=None, weights=None):
    """
    Arguments:
      - input (FloatTensor) [N, 4] (x1, y1, x2, y2)
      - target (FloatTensor) [N, 4] (x1, y1, x2, y2)
      - weights (FloatTensor) [N, 1]
    """
    ious = iou_overlaps(input, target, return_giou=True, return_diou=True, return_ciou=True, return_eiou=True)
    if loss_type == 'iou':
        loss = -ious['iou'].log()
    elif loss_type == 'linear_iou':
        loss = 1 - ious['iou']
    elif loss_type == 'square_iou':
        loss = 1 - ious['iou'] ** 2
    else:
        loss = 1 - ious[loss_type]
    if weights is not None:
        loss *= weights
    loss = _reduce(loss, reduction=reduction, normalizer=normalizer)
    return loss


@LOSSES_REGISTRY.register('iou_loss')
class IOULoss(BaseLoss):
    def __init__(self,
                 name='iou_loss',
                 reduction='mean',
                 loss_weight=1.0,
                 loss_type='iou'):
        r"""
        Arguments:
            - name (:obj:`str`): name of the loss function
            - reduction (:obj:`str`): reduction type, choice of mean, none, sum
            - loss_weight (:obj:`float`): loss weight
            - loss_type index (:obj:`str`): choice of 'iou', 'linear_iou', 'giou'
        """
        BaseLoss.__init__(self,
                          name=name,
                          reduction=reduction,
                          loss_weight=loss_weight)
        self.loss_type = loss_type
        self.key_fields = ["weights", "anchor", "bbox_normalize", "normalizer_override"]

    def forward(self, input, target, reduction, normalizer=None, weights=None, anchor=None, bbox_normalize=None):
        """
        Arguments:
            - input (:obj:`FloatTensor`), proposals [x1, y1, x2, y2], note the proposals must
              be in the computation graph.
            - target (:obj:`FloatTensor`), gt bboxes [x1, y2, x2, y2]
            - weights (:obj:`FloatTensor`), iou loss weights
        """
        if anchor is not None:
            if bbox_normalize is not None:
                input = unnormalize_offset(input, bbox_normalize["means"], bbox_normalize["stds"])
                target = unnormalize_offset(target, bbox_normalize['means'], bbox_normalize['stds'])
            input = offset2bbox(anchor, input)
            target = offset2bbox(anchor, target)
        loss = iou_loss(input, target, loss_type=self.loss_type, reduction='none', weights=weights)
        return _reduce(loss, reduction=reduction, normalizer=normalizer)


@LOSSES_REGISTRY.register("compose_loc_loss")
class ComposeLocLoss():
    def __init__(self,
                 name='compose_loc_loss',
                 loss_cfg=[]):
        self.name = name
        self.loss_list = [LOSSES_REGISTRY.build(cfg) for cfg in loss_cfg]
        self.key_fields = set()
        self._parse_key_fields()

    def __call__(self, *args, **kwargs):
        all_loss = 0
        for _loss_func in self.loss_list:
            new_kwargs = {}
            for k in getattr(_loss_func, "key_fields", set()):
                new_kwargs[k] = kwargs.get(k, None)
            all_loss += _loss_func(*args, **new_kwargs)
        return all_loss

    def _parse_key_fields(self):
        for _loss in self.loss_list:
            for item in getattr(_loss, 'key_fields', set()):
                self.key_fields.add(item)
