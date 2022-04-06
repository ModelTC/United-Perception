import numpy as np
import torch


def rand_bbox(size, lam):

    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def mixup(data, alpha, num_classes):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    images = data['image']
    batch_size = images.size()[0]
    index = torch.randperm(batch_size)
    mixed_x = lam * images + (1 - lam) * images[index, :]

    if len(data['gt'].shape) > 1:
        num_head = data['gt'].shape[1]
        labels = torch.zeros(batch_size, num_head, num_classes)
        labels.scatter_(2, data.gt.reshape(-1, num_head, 1), 1)
    else:
        labels = torch.zeros(batch_size, num_classes)
        labels.scatter_(1, data.gt.reshape(-1, 1), 1)

    y_a, y_b = labels, labels[index]
    mixed_y = lam * y_a + (1 - lam) * y_b

    data['image'] = mixed_x
    data['gt'] = mixed_y
    return data


def cutmix(data, alpha, num_classes):

    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    images = data['image']
    batch_size = images.size()[0]
    index = torch.randperm(batch_size)
    if len(data['gt'].shape) > 1:
        num_head = data['gt'].shape[1]
        labels = torch.zeros(batch_size, num_head, num_classes)
        labels.scatter_(2, data.gt.reshape(-1, num_head, 1), 1)
    else:
        labels = torch.zeros(batch_size, num_classes)
        labels.scatter_(1, data.gt.reshape(-1, 1), 1)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data['image'].shape, lam)
    images[:, :, bbx1:bbx2, bby1:bby2] = images[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
    data.gt = lam * labels + (1 - lam) * labels[index]
    data.image = data.image

    return data
