import random
from abc import ABCMeta, abstractmethod


__all__ = ["BaseSSD"]


class BaseSSD(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def get_SSD(self):
        '''
        return base config
        '''
        pass

    def get_global_SSD(self):
        global_SSD_list = []
        return global_SSD_list

    def resolve_SSD(self, cell_config, **kwargs):
        raise Exception('resolve_SSD needs to be overrided')

    def check_global_var(self):
        if hasattr(self.SSD, 'global_var') and len(self.SSD.global_var) > 0:
            return True
        else:
            return False

    def resolve_global_var(self, cell_config):
        raise Exception('resolve_global_var needs to be overrided')

    def get_random_samples(self):
        global_ssd_list = self.get_global_SSD()
        ssd_list = self.get_SSD()
        sample = []
        for ssd in global_ssd_list:
            total_choice = len(ssd)
            sample.append(random.randint(0, total_choice - 1))
        for i in range(self.cell_num):
            for ssd in ssd_list:
                total_choice = len(ssd)
                sample.append(random.randint(0, total_choice - 1))
        return sample
