import copy
from eod.utils.general.log_helper import default_logger as logger
from .samplers.batch_sampler import InfiniteBatchSampler

from eod.utils.general.registry_factory import (
    DATASET_REGISTRY,
    BATCH_SAMPLER_REGISTRY,
    DATALOADER_REGISTRY,
    SAMPLER_REGISTRY,
    DATA_BUILDER_REGISTY)


__all__ = ['BaseDataLoaderBuilder']


@DATA_BUILDER_REGISTY.register('base')
class BaseDataLoaderBuilder(object):
    def __init__(self, cfg, data_source, reload_cfg=None):
        self.cfg = cfg
        self.data_source = data_source
        self.reload_cfg = reload_cfg

    def parse_data_source(self):
        """
        Arguments:
            - data_source: `train:train`
        """
        splits = self.data_source.split(':')
        if len(splits) == 1:
            self.phase, self.source = splits[0], splits[0]
        else:
            self.phase, self.source = splits[0], splits[1]

    def build_dataset(self, cfg_dataset):
        if self.reload_cfg is not None and self.phase == 'train':
            cfg_dataset['kwargs']['reload_cfg'] = self.reload_cfg
        return DATASET_REGISTRY.build(cfg_dataset)

    def build_sampler(self, cfg_sampler, dataset):
        """Only works for training. We support dist_test for testing for now
        """
        cfg_sampler = copy.deepcopy(cfg_sampler)
        cfg_sampler['kwargs']['dataset'] = dataset
        if self.phase == 'test':
            test_sampler = {'type': 'dist_test', 'kwargs': {'dataset': dataset}}
            logger.warning('We use dist_test instead of {} for test'.format(cfg_sampler['type']))
            return SAMPLER_REGISTRY.build(test_sampler)
        return SAMPLER_REGISTRY.build(cfg_sampler)

    def build_batch_sampler(self, cfg_batch_sampler, dataset):
        cfg_batch_sampler = copy.deepcopy(cfg_batch_sampler)

        cfg_sampler = cfg_batch_sampler['kwargs']['sampler']
        sampler = self.build_sampler(cfg_sampler, dataset)

        cfg_batch_sampler['kwargs']['sampler'] = sampler
        if self.phase == 'test' and cfg_batch_sampler['type'] == 'aspect_ratio_group':
            logger.warning("'aspect_ratio_group' is not a 'test-friendly' choice, "
                           "cause it may miss some images under some conditions. "
                           "We use 'base' instead")
            cfg_batch_sampler = {
                'type': 'base',
                'kwargs': {
                    'sampler': sampler,
                    'batch_size': cfg_batch_sampler['kwargs']['batch_size']
                }
            }
        batch_sampler = BATCH_SAMPLER_REGISTRY.build(cfg_batch_sampler)
        batch_sampler = InfiniteBatchSampler(batch_sampler)

        return batch_sampler

    def update_cfg(self):
        assert self.source in self.cfg
        cfg = copy.deepcopy(self.cfg)
        cfg_data = cfg.pop(self.source)
        cfg.update(cfg_data)
        return cfg

    def generate_cfg_dataloader(self, cfg, dataset, batch_sampler):
        cfg_dataloader = cfg['dataloader']
        cfg_dataloader['kwargs']['dataset'] = dataset
        cfg_dataloader['kwargs']['batch_sampler'] = batch_sampler
        return cfg_dataloader

    def build_dataloader(self):
        self.parse_data_source()
        cfg = self.update_cfg()
        dataset = self.build_dataset(cfg['dataset'])
        batch_sampler = self.build_batch_sampler(cfg['batch_sampler'], dataset)
        cfg_dataloader = self.generate_cfg_dataloader(cfg, dataset, batch_sampler)
        return DATALOADER_REGISTRY.build(cfg_dataloader)
