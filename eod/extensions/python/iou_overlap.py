# Import from local
from .._C import overlaps


def gpu_iou_overlap(b1, b2, mode='IoU'):
    """compute IoU/IoF/IoS between b1 and b2
    Args:
        b1: Tensor, [N, >=4]
        b2: Tensor, [M, >=4]
        mode: str, {IoU, IoF, IoS} Intersection over Union/First/Second area
    """
    if b1.numel() == 0 or b2.numel() == 0:
        return b1.new_zeros((0,))

    flag = {'IoU': 0, 'IoF': 1, 'IoS': 2}[mode]

    assert b1.shape[1] >= 4 and b2.shape[1] >= 4
    assert b1.is_cuda and b2.is_cuda

    b1 = b1[:, :4].contiguous()
    b2 = b2[:, :4].contiguous()
    ious = b1.new_zeros((b1.shape[0], b2.shape[0]))
    overlaps.iou(b1, b2, ious, flag)
    return ious
