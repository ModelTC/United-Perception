# Standard Library
from functools import reduce

# Import from third library
import torch


def binary_accuracy(output, target, thresh=0.5, ignore_index=-1):
    """binary classification accuracy. e.g sigmoid"""
    output = output.view(-1)
    target = target.view(-1)
    keep = torch.nonzero(target != ignore_index).squeeze()
    if keep.dim() <= 0:
        return [torch.cuda.FloatTensor([1]).zero_()]
    assert (keep.dim() == 1)
    target = target[keep]
    output = output[keep]
    binary = (output > thresh).type_as(target)
    tp = (binary == target).float().sum(0, keepdim=True)
    return [tp.mul_(100.0 / target.numel())]


def accuracy(output, target, topk=(1, ), ignore_indices=[-1]):
    """Computes the precision@k for the specified values of k"""
    if output.numel() != target.numel():
        C = output.shape[-1]
        output = output.view(-1, C)
        target = target.view(-1)
    masks = [target != idx for idx in ignore_indices]
    mask = reduce(lambda x, y: x & y, masks)
    keep = torch.nonzero(mask).squeeze()
    if keep.numel() <= 0:
        return [torch.cuda.FloatTensor([1]).zero_()]
    if keep.dim() == 0:
        keep = keep.view(-1)
    assert keep.dim() == 1, keep.dim()
    target = target[keep]
    output = output[keep]
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def accuracy_v2(cls_pred, cls_target, activation_type='softmax'):
    if cls_pred.numel() <= 0:
        return cls_pred.sum()
    cls_pred = cls_pred.reshape(cls_target.numel(), -1)
    cls_target = cls_target.reshape(-1)
    num_channel = cls_pred.shape[1]
    if activation_type == 'softmax':
        acc = accuracy(cls_pred, cls_target)[0]
    elif activation_type == 'sigmoid' and num_channel > 2 or activation_type == 'qfl':
        acc = accuracy(cls_pred, cls_target.long() - 1, ignore_indices=[-1, -2])[0]
    elif activation_type == 'sigmoid' and num_channel <= 2:
        acc = binary_accuracy(cls_pred, cls_target)[0]
    else:
        raise NotImplementedError
    return acc
