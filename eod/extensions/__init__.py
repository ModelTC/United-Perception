# flake8: noqa

from .python.roi_align import RoIAlignPool
from .python.psroi_align import *
from .python.psroi_pool import PSRoIPool
from .python.nms import naive_nms


def build_generic_roipool(pool_cfg):
    """
    Construct generic RoIPooling-like class

    Arguments:
        pool_cfg: `method` key to speicify roi_pooling method,
            other keys is configuration for this roipooling `method`
    """
    ptype = pool_cfg['method']
    pool_cls = {
        'roialignpool': RoIAlignPool,
        'psroialign': PSRoIAlign,
        'psroipool': PSRoIPool
    }[ptype]
    # RoIAlignPool
    return pool_cls.from_params(pool_cfg)


def nms(bboxes, cfg, descending=True):
    """
    Arguments:
        bboxes (FloatTensor): [N, 5] (x1,y1,x2,y2,score), sorted by score in descending order
        cfg (dict): config for nms
    """
    bboxes = bboxes.contiguous()
    assert bboxes.numel() > 0
    assert bboxes.shape[1] == 5, "Expected nms input bboxes with 5 channels, but got {}".format(bboxes.shape[1])

    if not descending:
        sorted_scores, order = torch.sort(bboxes[:, 4], descending=True)
        bboxes = bboxes[order]

    if cfg['type'] == 'naive':
        if cfg['nms_iou_thresh'] < 0:
            keep_index = torch.arange(bboxes.shape[0], device=bboxes.device)
            nmsed_bboxes = bboxes
        else:
            keep_index = naive_nms(bboxes, cfg['nms_iou_thresh'])
            nmsed_bboxes = bboxes[keep_index]
    else:
        raise NotImplementedError('NMS method {} not supported'.format(cfg['type']))

    if not descending:
        keep_index = order[keep_index]
    return nmsed_bboxes, keep_index
