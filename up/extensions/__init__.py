# flake8: noqa

from .python.roi_align import RoIAlignPool
from .python.psroi_align import PSRoIAlign 
from .python.psroi_pool import PSRoIPool
from .python.nms import naive_nms,softer_nms
from .python.deformable_conv import DeformableConv
from .python.focal_loss import (
    SigmoidFocalLossFunction,
    SoftmaxFocalLossFunction
)
from .python.cross_focal_loss import CrossSigmoidFocalLossFunction
from .python.iou_overlap import gpu_iou_overlap
import torch

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
    return pool_cls.from_params(pool_cfg)


def nms(bboxes, cfg, descending=True):
    """
    Arguments:
        bboxes (FloatTensor): [N, 5] (x1,y1,x2,y2,score), sorted by score in descending order
        cfg (dict): config for nms
    """
    bboxes = bboxes.contiguous()
    assert bboxes.numel() > 0

    if cfg['type'] == 'softer':
        assert bboxes.shape[1] == 9, "Expected nms input bboxes with 9 channels, but got {}".format(bboxes.shape[1])
    else:
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
    elif cfg['type'] == 'softer':
        sigma = np.float32(cfg.get('softnms_sigma', 0.5))
        iou_sigma = np.float32(cfg.get('softernms_sigma', 0.02))
        iou_thresh = np.float32(cfg['nms_iou_thresh'])
        threshold = np.float32(cfg.get('softnms_bbox_score_thresh', 0.0001))
        iou_method = cfg.get('softnms_method', 'linear')
        iou_methods = {'hard': 0, 'linear': 1, 'gaussian': 2}
        assert iou_method in iou_methods, 'Unknown soft_nms method: {}'.format(iou_method)
        method = cfg.get('softernms_method', 'var_voting')
        methods = {'softer': 1, 'var_voting': 0}
        assert method in methods, 'Unknown softer_nms method: {}'.format(method)
        bboxes_cp = bboxes.clone()
        nmsed_bboxes, keep_index = softer_nms(dets=bboxes_cp, sigma=sigma, iou_thresh=iou_thresh, thresh=threshold,
                                              iou_method=iou_method, iou_sigma=iou_sigma, method=method)
        nmsed_bboxes = nmsed_bboxes[:, [0, 1, 2, 3, 8]]
    else:
        raise NotImplementedError('NMS method {} not supported'.format(cfg['type']))

    if not descending:
        keep_index = order[keep_index]
    return nmsed_bboxes, keep_index
