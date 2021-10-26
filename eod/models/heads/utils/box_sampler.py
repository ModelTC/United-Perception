# Import from third library
import torch

from eod.utils.general.registry_factory import ROI_SAMPLER_REGISTRY


__all__ = [
    'KeepBatchRoiSampler',
    'IouBalacneRoiSampler',
    'KeepAllRoiSampler',
    'KeepRatioRoiSampler',
    'NoRandomRoiSampler']


# sample fixed batch size
@ROI_SAMPLER_REGISTRY.register('naive')
class KeepBatchRoiSampler(object):
    """Sample positives up to pos_percent, if positive num is not enough,
    sample more negatives to make it up to the batch_size
    """
    def __init__(self, batch_size, positive_percent):
        self.batch_size = batch_size
        self.positive_percent = positive_percent

    def sample(self, match_target, **kwargs):
        """
        Arguments:
            - match_target (``LongTensor``): [N] output of :meth:`~root.models.heads.utils.matcher.match` function,
            matched gt index for N RoIs.
        """
        positive = torch.nonzero(match_target >= 0).reshape(-1)
        negative = torch.nonzero(match_target == -1).reshape(-1)

        expected_pos_num = int(self.batch_size * self.positive_percent)
        if positive.numel() > expected_pos_num:
            shuffle_pos = torch.randperm(positive.numel(), device=positive.device)[:expected_pos_num]
            positive = positive[shuffle_pos]

        expected_neg_num = self.batch_size - positive.numel()
        if negative.numel() > expected_neg_num:
            shuffle_neg = torch.randperm(negative.numel(), device=negative.device)[:expected_neg_num]
            negative = negative[shuffle_neg]
        return positive, negative


@ROI_SAMPLER_REGISTRY.register('force_keep_ratio')
class KeepRatioRoiSampler(object):
    """Sample pos_percent positives from `positive` if not empty,
    so does to negative
    """
    def __init__(self, batch_size, positive_percent):
        self.batch_size = batch_size
        self.positive_percent = positive_percent

    def sample(self, match_target, **kwargs):
        """
        Arguments:
            - match_target (``LongTensor``): [N] output of :meth:`~root.models.heads.utils.matcher.match` function,
            matched gt index for N RoIs.
        """
        positive = torch.nonzero(match_target >= 0).reshape(-1)
        negative = torch.nonzero(match_target == -1).reshape(-1)

        expected_pos_num = int(self.batch_size * self.positive_percent)
        expected_neg_num = self.batch_size - expected_pos_num
        if positive.numel() > 0:
            if expected_pos_num > positive.numel():
                shuffle_pos = torch.randperm(expected_pos_num, device=positive.device) % positive.numel()
            else:
                shuffle_pos = torch.randperm(positive.numel(), device=positive.device)[:expected_pos_num]
            positive = positive[shuffle_pos]
        if negative.numel() > 0:
            if expected_neg_num > negative.numel():
                shuffle_neg = torch.randperm(expected_neg_num, device=negative.device) % negative.numel()
            else:
                shuffle_neg = torch.randperm(negative.numel(), device=negative.device)[:expected_neg_num]
            negative = negative[shuffle_neg]
        return positive, negative


@ROI_SAMPLER_REGISTRY.register('balanced')
class IouBalacneRoiSampler(object):
    """IoU-balanced sampling, balanced sampling is only applied to negative examples
    """
    def __init__(self, batch_size, positive_percent, k, negative_iou_thresh):
        self.batch_size = batch_size
        self.positive_percent = positive_percent
        self.num_bin = k
        self.negative_iou_thresh = negative_iou_thresh

    def sample(self, match_target, overlaps, **kwargs):
        """
        Arguments:
            - match_target (``LongTensor``): [N] output of :meth:`~root.models.heads.utils.matcher.match` function,
            matched gt index for N RoIs.
        """
        positive = torch.nonzero(match_target >= 0).reshape(-1)
        negative = torch.nonzero(match_target == -1).reshape(-1)

        expected_pos_num = int(self.batch_size * self.positive_percent)
        if positive.numel() > expected_pos_num:
            shuffle_pos = torch.randperm(positive.numel(), device=positive.device)[:expected_pos_num]
            positive = positive[shuffle_pos]

        expected_neg_num = self.batch_size - positive.numel()
        expected_neg_num_per_bin = expected_neg_num // self.num_bin

        sampled_negative = []
        neg_overlaps = overlaps[negative]
        for i in range(self.num_bin):
            low = i / self.num_bin * self.negative_iou_thresh
            high = (i + 1) / self.num_bin * self.negative_iou_thresh
            bin_neg_inds = torch.nonzero((neg_overlaps >= low) & (neg_overlaps < high)).view(-1)
            bin_sampled_neg = torch.randperm(bin_neg_inds.numel(),
                                             device=negative.device)[:expected_neg_num_per_bin]
            sampled_negative.append(negative[bin_neg_inds[bin_sampled_neg]])
        negative = torch.cat(sampled_negative, dim=0)
        return positive, negative


@ROI_SAMPLER_REGISTRY.register('keep_all')
class KeepAllRoiSampler(object):
    """IoU-balanced sampling, balanced sampling is only applied to negative examples
    """
    def sample(self, match_target, **kwargs):
        positive = torch.nonzero(match_target >= 0).reshape(-1)
        negative = torch.nonzero(match_target == -1).reshape(-1)

        return positive, negative


@ROI_SAMPLER_REGISTRY.register('no_random')
class NoRandomRoiSampler(object):
    """Sample positives up to pos_percent, if positive num is not enough,
    sample more negatives to make it up to the batch_size
    """
    def __init__(self, batch_size, positive_percent):
        self.batch_size = batch_size
        self.positive_percent = positive_percent

    def sample(self, match_target, **kwargs):
        """
        Arguments:
            - match_target (``LongTensor``): [N] output of :meth:`~root.models.heads.utils.matcher.match` function,
            matched gt index for N RoIs.
        """
        positive = torch.nonzero(match_target >= 0).reshape(-1)
        negative = torch.nonzero(match_target == -1).reshape(-1)

        expected_pos_num = int(self.batch_size * self.positive_percent)
        if positive.numel() > expected_pos_num:
            positive = positive[:expected_pos_num]

        expected_neg_num = self.batch_size - positive.numel()
        if negative.numel() > expected_neg_num:
            negative = negative[:expected_neg_num]
        return positive, negative


def build_roi_sampler(cfg):
    return ROI_SAMPLER_REGISTRY.build(cfg)


def sample(match_target, cfg, overlaps=None):
    sampler = build_roi_sampler(cfg)
    return sampler.sample(match_target, overlaps=overlaps)
