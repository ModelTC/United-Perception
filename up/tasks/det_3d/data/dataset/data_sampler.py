from torch.utils.data.sampler import BatchSampler
from up.utils.general.registry_factory import BATCH_SAMPLER_REGISTRY, SAMPLER_REGISTRY

__all__ = ['PointBatchSampler']


@BATCH_SAMPLER_REGISTRY.register('point')
class PointBatchSampler(BatchSampler):
    def __init__(self, sampler, batch_size, drop_last=False):
        super(PointBatchSampler, self).__init__(sampler, batch_size, drop_last)

        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size  # type: ignore
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size  # type: ignore


def build_sampler(cfg_sampler):
    return SAMPLER_REGISTRY.build(cfg_sampler)
