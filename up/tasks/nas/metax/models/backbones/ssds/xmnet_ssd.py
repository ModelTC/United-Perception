from easydict import EasyDict as edict
from .base_ssd import BaseSSD


__all__ = ['xmnetSSD']


class xmnetSSD(BaseSSD):
    def __init__(self):
        super(xmnetSSD, self).__init__()

        self.SSD = edict()
        self.SSD.block_type = ['mbconv', 'fuseconv']
        self.SSD.kernel_size_choice = [3, 5, 7]
        self.SSD.repeat_choice = [-2, -1, 0, 1, 2]
        self.SSD.channel_choice = [0.5, 0.75, 1.0, 1.25, 1.5]
        self.SSD.expansion_choice = [2, 3, 4, 5, 6]

        self.cell_num = 5

    def get_SSD(self):
        SSD_list = []
        SSD_list.append(self.SSD.block_type)
        SSD_list.append(self.SSD.kernel_size_choice)
        SSD_list.append(self.SSD.repeat_choice)
        SSD_list.append(self.SSD.channel_choice)
        SSD_list.append(self.SSD.expansion_choice)
        return SSD_list

    def resolve_SSD(self, cell_config, BlockArgs, **kwargs):
        block_type_index, kernel_size_index, repeats_index, channel_choice_index, expansion_choice_index = cell_config

        # block_type_idx, filter_ratio_idx, layers_idx, module_num_idx = cell_config
        block_type = self.SSD.block_type[block_type_index]
        kernel_size = self.SSD.kernel_size_choice[kernel_size_index]
        repeat = self.SSD.repeat_choice[repeats_index]
        channel = self.SSD.channel_choice[channel_choice_index]
        expansion = self.SSD.expansion_choice[expansion_choice_index]

        return BlockArgs(
            block=block_type,
            kernel_size=kernel_size,
            channel=channel,
            repeat=repeat,
            expansion=expansion
        )
