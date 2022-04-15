from torch.utils.data import DataLoader
# import torch
from easydict import EasyDict

from up.utils.general.registry_factory import DATALOADER_REGISTRY
from up.data.samplers.batch_sampler import InfiniteBatchSampler
from up.utils.general.registry_factory import BATCHING_REGISTRY


__all__ = ['SegDataLoader']


def expand_seg(seg):
    if seg.dim() == 3:
        assert seg.size(0) == 1, seg.size()
        return seg
    else:
        return seg[None, :, :]


@DATALOADER_REGISTRY.register('seg_base')
class SegDataLoader(DataLoader):

    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None,
                 num_workers=0, pin_memory=False, drop_last=False,
                 alignment=1, pad_type='batch_pad', pad_value=[0, 255]
                 ):
        super(SegDataLoader, self).__init__(
            dataset, batch_size, shuffle, sampler, batch_sampler, num_workers,
            self._collate_fn, pin_memory, drop_last)
        self.alignment = alignment
        self.pad_image = BATCHING_REGISTRY.get(pad_type)(alignment, pad_value[0])
        self.pad_label = BATCHING_REGISTRY.get(pad_type)(alignment, pad_value[1])

    def _collate_fn(self, batch):
        images = [_.image for _ in batch]
        gt_semantic_seg = [_.get('gt_semantic_seg', None) for _ in batch]
        image_info = [_.get('image_info', None) for _ in batch]
        output = EasyDict({
            'image': images,
            'image_info': image_info
        })
        if gt_semantic_seg[0] is not None:
            gt_semantic_seg = gt_semantic_seg
        output.gt_semantic_seg = gt_semantic_seg

        if self.alignment > 0:  # when size not match, directly cat will fail
            output = self.pad_image(output)  # image
            if gt_semantic_seg[0] is not None:
                fake_dict = {'image': [expand_seg(seg) for seg in gt_semantic_seg]}
                output['gt_semantic_seg'] = self.pad_label(fake_dict)['image'][:, 0, :, :]

        return output

    def get_epoch_size(self):
        if isinstance(self.batch_sampler, InfiniteBatchSampler):
            return len(self.batch_sampler.batch_sampler)   # training
        return len(self.batch_sampler)
