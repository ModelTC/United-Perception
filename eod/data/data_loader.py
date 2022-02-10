from __future__ import division

# Import from third library
from easydict import EasyDict
from torch.utils.data import DataLoader

from eod.utils.general.registry_factory import DATALOADER_REGISTRY, BATCHING_REGISTRY
from .samplers.batch_sampler import InfiniteBatchSampler


__all__ = ['BaseDataLoader']


@DATALOADER_REGISTRY.register('base')
class BaseDataLoader(DataLoader):
    def __init__(self,
                 dataset,
                 alignment=1,
                 batch_size=1,
                 shuffle=False,
                 sampler=None,
                 batch_sampler=None,
                 num_workers=0,
                 pin_memory=False,
                 drop_last=False,
                 pad_value=0,
                 pad_type='batch_pad',
                 worker_init=False):
        worker_init_fn = None
        if worker_init:
            from eod.utils.env.gene_env import worker_init_reset_seed
            worker_init_fn = worker_init_reset_seed
        super(BaseDataLoader, self).__init__(
            dataset, batch_size, shuffle, sampler, batch_sampler, num_workers,
            self._collate_fn, pin_memory, drop_last, worker_init_fn=worker_init_fn)
        self.pad = BATCHING_REGISTRY.get(pad_type)(alignment, pad_value)

    def _collate_fn(self, batch):
        """
        Form a mini-batch from list of data of :meth:`~root.datasets.base_dataset.BaseDataset.__getitem__`

        Arguments:
            - batch (:obj:`list` of data): type of data depends on output of Dataset.
              For :class:`~root.datasets.coco_dataset.CocoDataset`,
              :class:`~root.datasets.voc_dataset.VocDataset` and :class:`~root.datasets.custom_dataset.CustomDataset`,
              data is a Dictionary.

        Returns:
            - output (:obj:`dict`)

        Output example::

            {
                # (FloatTensor): [B, 3, max_h, max_w], RGB format
                'image': ..,
                # (list of FloatTensor): [B, 5], (resized_h, resize_w, scale_factor, origin_h, origin_w)
                'image_info': ..,
                # (list of FloatTensor): [B] [num_gts, 5] or None
                'gt_bboxes': ..,
                # (list of FloatTensor): [B] [num_igs, 4] or None
                'ig_bboxes': ..,
                #(list of FloatTensor): [B] [num_gts, k, 3] or None
                'gt_keyps': ..,
                #(list of list of list of ndarray): [B] [num_gts] [polygons] or None
                'gt_masks': ..,
                # (FloatTensor) [B, max_num_gts, num_grid_points(9), 3] or None
                'gt_grids': ..,
                # (FloatTensor) [B, 1, max_h, max_w], semantic mask
                'gt_semantic_seg': ..,
                # filename of images
                'filenames': ..,
                # label of negative sample (default = 0)
                'neg_targets': ..
            }
        """
        images = [_['image'] for _ in batch]
        image_info = [_['image_info'] for _ in batch]
        filenames = [_['filename'] for _ in batch]
        image_ids = [_['image_id'] for _ in batch]
        flipped = [_['flipped'] for _ in batch]
        neg_targets = [_.get('neg_target', 0) for _ in batch]
        image_sources = [_.get('image_source', 0) for _ in batch]
        gt_masks = [_.get('gt_masks', None) for _ in batch]
        gt_bboxes = [_.get('gt_bboxes', None) for _ in batch]
        gt_ignores = [_.get('gt_ignores', None) for _ in batch]
        caches = [_.get('cache', False) for _ in batch]
        gt_semantic_seg = [_.get('gt_semantic_seg', None) for _ in batch]

        output = EasyDict({
            'image': images,
            'image_info': image_info,
            'image_id': image_ids,
            'filenames': filenames,
            'flipped': flipped,
            'neg_targets': neg_targets,
            'image_sources': image_sources,
            'caches': caches
        })
        output['gt_masks'] = gt_masks if gt_masks[0] is not None else None
        output['gt_bboxes'] = gt_bboxes if gt_bboxes[0] is not None else None
        output['gt_ignores'] = gt_ignores if gt_ignores[0] is not None else None
        if gt_semantic_seg[0] is not None:
            fake_dict = {'image': gt_semantic_seg}
            output['gt_semantic_seg'] = self.pad(fake_dict)['image']
        output = self.pad(output)
        return output

    def get_data_size(self):
        return self.get_epoch_size()

    def get_epoch_size(self):
        if isinstance(self.batch_sampler, InfiniteBatchSampler):
            return len(self.batch_sampler.batch_sampler)   # training
        return len(self.batch_sampler)
