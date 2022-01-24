# Import from third library
import numpy as np
import torch
from .._C import naive_nms as nms
from up.utils.general.global_flag import ALIGNED_FLAG


def naive_nms(dets, thresh):
    assert dets.shape[1] == 5
    num_dets = dets.shape[0]
    if torch.is_tensor(num_dets):
        num_dets = num_dets.item()
    keep = torch.LongTensor(num_dets)
    num_out = torch.LongTensor(1)
    if dets.device.type == 'cuda':
        nms.gpu_nms(keep, num_out, dets.float(), thresh, ALIGNED_FLAG.offset)
    else:
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        areas = (x2 - x1 + ALIGNED_FLAG.offset) * (y2 - y1 + ALIGNED_FLAG.offset)
        order = torch.from_numpy(np.arange(dets.shape[0])).long()
        nms.cpu_nms(keep, num_out, dets.float(), order, areas, thresh, ALIGNED_FLAG.offset)
    return keep[:num_out[0]].contiguous().to(device=dets.device)
