import math
import random
import numpy as np
import copy
from easydict import EasyDict as edict
import itertools

import torch
import torch.nn as nn

from up.tasks.nas.bignas.utils.registry_factory import DYNAMIC_BLOCKS_REGISTRY
from up.tasks.nas.bignas.models.ops.dynamic_blocks import DynamicLinearBlock, DynamicBasicBlock, Identity


# from up.tasks.nas.bignas.models.ops.dynamic_blocks import DynamicLinearBlock, DynamicBasicBlock, Identity
from up.tasks.nas.bignas.models.ops.dynamic_utils import int2list, copy_module_weights, sub_filter_start_end
from up.tasks.nas.bignas.utils.sample import sample_search_settings
from up.tasks.nas.bignas.utils.registry_factory import STATIC_BLOCKS_REGISTRY
from up.utils.general.log_helper import default_logger as logger

try:
    import spring.linklink as link
except:  # noqa
    from up.utils.general.fake_linklink import link


class BignasSearchSpace(nn.Module):

    def __init__(self,
                 # stage wise config
                 **kwargs):
        super(BignasSearchSpace, self).__init__()
        """
        BigNas SearchSpace base module
        Args:
            out_channel (dict):
                space (dict):
                    min (list): the stage wise config of the minimum output channels of every stage
                    max (list): the stage wise config of the maximum output channels of every stage
                    stride (None or int or list): the stage wise config of channel stride of every stage
                    dynamic_range (None or list of list): support compute by min, max and stride or manually set
                sample_strategy: the sample strategy of channels
            depth (dict):
                space (dict):
                    min (list): the stage wise config of the minimum depth of every stage
                    max (list): the stage wise config of the maximum depth of every stage
                    stride (None or int or list): the stage wise config of depth stride of every stage
                    dynamic_range (None or list of list): support compute by min, max and stride or manually set
                sample_strategy (str): the sample strategy of depth
            kernel_size (dict):
                space (dict):
                    min (list): the stage wise config of the minimum kernel size of every stage
                    max (list): the stage wise config of the maximum kernel size of every stage
                    stride (None or int or list): the stage wise config of kernel size stride of every stage
                    dynamic_range (None or list of list): support compute by min, max and stride or manually set
                sample_strategy (str): the sample strategy of kernel size
            expand_ratio (dict):
                space (dict):
                    min (list): the stage wise config of the minimum expand ratio of every stage
                    max (list): the stage wise config of the maximum expand ratio of every stage
                    stride (None int or list): the stage wise config of expand ratio stride of every stage
                    dynamic_range (None or list of list): support compute by min, max and stride or manually set
                sample_strategy (str): the sample strategy of expand ratio
            **kwargs: (extra search element such as group_width)
            group_width (None or dict):
                space (dict):
                    min (list): the stage wise config of the minimum group width of every stage
                    max (list): the stage wise config of the maximum group width of every stage
                    stride (None int or list): the stage wise config of group width stride of every stage
                    dynamic_range (None or list of list): support compute by min, max and stride or manually set
                sample_strategy (str): the sample strategy of group width
        """
        if 'depth' not in kwargs.keys():
            raise ValueError('you have to include depth in the config')

        self.dynamic_settings, self.normal_settings = self.build_model_dynamic_settings(**kwargs)
        self.subnet_settings = self.build_model_subnet_settings()
        self.curr_subnet_settings = copy.deepcopy(self.normal_settings)

        # in case there is no search for kernel size settings.
        if 'kernel_size' in self.normal_settings.keys():
            kernel_size = [max(i) if isinstance(i, list) else i for i in self.normal_settings['kernel_size']]
            self.curr_subnet_settings['kernel_size'] = kernel_size

        self.dynamic_module_list = None
        self.remove_register_names = []

    def build_model_dynamic_settings(self, **kwargs):
        # 只存储搜索的element
        dynamic_settings = edict({})
        # 以dict形式存储所有的搜索和不搜索的element
        normal_settings = edict({})
        for k, v in kwargs.items():
            processv = self.compute_search_element_settings(v, k)
            if isinstance(processv, list):
                normal_settings[k] = processv
            else:
                if processv.search:
                    dynamic_settings[k] = processv
                    if k == 'kernel_size' or k == 'expand_ratio' or k == 'out_channel':
                        normal_settings[k] = dynamic_settings[k]['space']['dynamic_range']
                    else:
                        normal_settings[k] = dynamic_settings[k]['space']['max']
                else:
                    normal_settings[k] = processv.space.max
        return dynamic_settings, normal_settings

    def compute_search_element_settings(self, element, element_name=None):
        """
        compute the stage_wise dynamic_range and return whether it needs search or not
        Args:
            element (dict or list): (search element)
                if dict:
                    space (dict):
                        min (list): the stage wise config of the minimum range of every stage
                        max (list): the stage wise config of the maximum range of every stage
                        stride (None or int or list): the stage wise config of channel stride of every stage
                        dynamic_range (None or list of list): support compute by min, max and stride or manually set
                    sample_strategy: the sample strategy of channels
                else:
                    it corresponds to list of int
        Return:
            processed element (dict):
        """
        if isinstance(element, list):
            return element
        elif not isinstance(element, dict):
            logger.info('Only supported list and dict value while get {} {}'.format(element, element_name))
            raise ValueError
        element = edict(element)
        if element.space.min == element.space.max:
            element.search = False
            return element
        else:
            element.search = True

        # process stride when stride is int
        stride = element.space.get('stride', None)
        if stride is not None:
            stride = int2list(stride, len(element.space.min))
            element.space.stride = stride

        dynamic_range = element.space.get('dynamic_range', None)

        # if stride is not empty and dynamic_range is not manually set, we compute
        if stride is not None and dynamic_range is None:
            def get_dynamic_stage_settings(min_range, max_range, range_stride):
                dynamic_range = []
                for curr_min, curr_max, stride in zip(min_range, max_range, range_stride):
                    if isinstance(stride, int):
                        extra_adder = 1
                        curr_range = [int(i) for i in np.arange(curr_min, curr_max + extra_adder, stride)]
                    elif isinstance(stride, float):
                        extra_adder = 1e-4
                        curr_range = [float(i) for i in np.arange(curr_min, curr_max + extra_adder, stride)]
                    dynamic_range.append(curr_range)
                return dynamic_range

            dynamic_range = get_dynamic_stage_settings(
                element.space.min, element.space.max, element.space.stride)
            element.space.dynamic_range = dynamic_range
        elif stride is None and dynamic_range is None:
            raise ValueError("stride and dynamic range cannot be both empty.")
        return element

    # 根据最大最小的范围来生成需要重点关注的模型有哪些，放在subnet_settings里面
    def build_model_subnet_settings(self):
        model_settings = {}
        for k, v in self.dynamic_settings.items():
            model_settings[k] = [v.space.min, v.space.max]
            if v.space.get('middle', None):
                model_settings[k] = [v.space.min, v.space.middle, v.space.max]

        subnet_settings = []
        keys = list(model_settings.keys())
        for settings in itertools.product(*model_settings.values()):
            curr_settings = {}
            for k, v in zip(keys, settings):
                curr_settings[k] = v
            if 'depth' not in curr_settings.keys():
                curr_settings['depth'] = self.normal_settings['depth']
            subnet_settings.append(curr_settings)
        return subnet_settings

    def forward(self, x):
        """Required.
        implement the normal forward process
        """
        raise NotImplementedError

    @staticmethod
    def name():
        return 'BignasSearchSpace'

    def build_active_subnet(self):
        """Required.
        implement to get the active subnet weights
        """
        raise NotImplementedError

    def get_active_subnet_module_list(self, subnet):
        module_list = []
        handles = []
        input = self.get_fake_input()
        device = next(self.parameters(), torch.tensor([])).device
        subnet = subnet.to(device)
        block_types = [i for i in STATIC_BLOCKS_REGISTRY.values()]

        def make_hook(module_name):
            def hook(module, data, out):
                module_list.append((module))
                logger.info('register block for {}'.format(module_name))
            return hook

        for n, m in subnet.named_modules():
            if sum(map(lambda x: isinstance(m, x), block_types)):
                handle = m.register_forward_hook(make_hook(n))
                handles.append(handle)
        subnet(input)
        for handle in handles:
            handle.remove()

        return module_list

    def set_default_subnet_weights(self, subnet_settings=None):
        if subnet_settings is None:
            subnet_settings = self.curr_subnet_settings
        if 'expand_ratio' in self.normal_settings.keys() and 'expand_ratio' not in subnet_settings.keys():
            subnet_settings['expand_ratio'] = self.normal_settings['expand_ratio']
        if 'kernel_size' in self.normal_settings.keys() and 'kernel_size' not in subnet_settings.keys():
            subnet_settings['kernel_size'] = [max(i) if isinstance(i, list) else i for i in
                                              self.normal_settings['kernel_size']]
        if 'group_width' in self.normal_settings.keys() and 'group_width' not in subnet_settings.keys():
            subnet_settings['group_width'] = self.normal_settings['group_width']
        if 'out_channel' in self.normal_settings.keys() and 'out_channel' not in subnet_settings.keys():
            subnet_settings['out_channel'] = self.normal_settings['out_channel']
        if 'depth' in self.normal_settings.keys() and 'depth' not in subnet_settings.keys():
            subnet_settings['depth'] = self.normal_settings['depth']
        return subnet_settings

    def sample_active_subnet_weights(self, subnet_settings=None, inplanes=None):
        subnet_settings = self.set_default_subnet_weights(subnet_settings)

        if inplanes is not None:
            subnet = self.build_active_subnet(subnet_settings, inplanes)
        else:
            subnet = self.build_active_subnet(subnet_settings)

        if inplanes is not None and hasattr(self, 'inplanes'):
            self.inplanes = inplanes

        subnet_module_list = self.get_active_subnet_module_list(subnet)
        active_module_list = self.get_active_module_list(subnet_settings['depth'])

        if hasattr(self, 'task') and self.task == 'classification':
            active_module_list += [self.dynamic_module_list[-1]]

        for block, super_block in zip(subnet_module_list, active_module_list):
            copy_module_weights(block, super_block)
        return subnet

    def sample_active_subnet(self, sample_mode='random', subnet_settings=None):
        # This is used in sandwich rule, ['min', 'max', 'random', 'subnet']
        # In the forward pass, we need to clarify the sample mode
        if sample_mode == 'max':
            return self.set_active_subnet(**self.subnet_settings[-1])
        elif sample_mode == 'min':
            return self.set_active_subnet(**self.subnet_settings[0])
        elif sample_mode == 'middle':
            # we need to remove min and max
            # if only min and max is involved, choose the min
            i = min([math.ceil((len(self.subnet_settings) - 1) / 2),
                    math.floor((len(self.subnet_settings) - 1) / 2)])
            return self.set_active_subnet(**self.subnet_settings[i])
        elif sample_mode == 'random_slim':
            # we need to remove min and max
            # if only min and max is involved, choose the min
            if len(self.subnet_settings) == 2:
                return self.set_active_subnet(**self.subnet_settings[0])
            else:
                i = random.choice([i for i in range(1, len(self.subnet_settings) - 1)])
            return self.set_active_subnet(**self.subnet_settings[i])
        elif sample_mode == 'subnet':
            if 'depth' not in subnet_settings.keys():
                subnet_settings['depth'] = self.normal_settings['depth']
            return self.set_active_subnet(**subnet_settings)
        elif sample_mode == 'random':
            subnet_settings = {}
            # we need to compute depth first
            if 'depth' not in self.dynamic_settings.keys():
                curr_depth_settings = self.normal_settings['depth']
            else:
                curr_depth_settings = sample_search_settings(self.dynamic_settings.depth.space.max,
                                                             self.dynamic_settings.depth.space.dynamic_range,
                                                             self.dynamic_settings.depth.sample_strategy)
            subnet_settings['depth'] = curr_depth_settings
            for k, v in self.dynamic_settings.items():
                if v.search and k != 'depth':
                    # For stage wise depth, we choose other elements with repect to their index in depth
                    if v.sample_strategy == 'stage_wise_depth':
                        sample_strategy = {
                            'name': v.sample_strategy,
                            'kwargs': {'depth_dynamic_range': self.dynamic_settings.depth.space.dynamic_range}}
                    else:
                        sample_strategy = v.sample_strategy
                    subnet_settings[k] = sample_search_settings(curr_depth_settings,
                                                                v.space.dynamic_range,
                                                                sample_strategy)

            return self.set_active_subnet(**subnet_settings)
        else:
            raise ValueError('Wrong sample mode with {}'.format(sample_mode))

    def get_active_module_list(self, depth_settings):
        if self.dynamic_module_list is None:
            self.gen_dynamic_module_list()

        # get the max dynamic module idx
        max_depth = self.normal_settings['depth']
        max_idx = [i for i in range(sum(max_depth))]
        max_group_idx = [max_idx[sum(max_depth[:i]):sum(max_depth[:i]) + d] for i, d in enumerate(max_depth)]

        # get the active module list from all the dynamic module list and set the forward action for every block
        active_dynamic_model_group_idx = [max_group_idx[i][:d] for i, d in enumerate(depth_settings)]
        active_dynamic_module_idx = []
        for i in active_dynamic_model_group_idx:
            active_dynamic_module_idx += i
        active_module_list = []

        for i, block in enumerate(self.dynamic_module_list):
            if i in active_dynamic_module_idx:
                active_module_list.append(block)
                block.active_block = True
            else:
                block.active_block = False

        # hot fix the linear block should alway be computed
        if getattr(self, 'task', '') == 'classification' \
                and isinstance(self.dynamic_module_list[-1], DynamicLinearBlock):
            self.dynamic_module_list[-1].active_block = True

        return active_module_list

    # 返回和depth的和的长度等长的element setting，返回block-wise的配置
    # 当输入的setting的长度是stage_wise的，就是len(element) == len(depth)的情况，需要加长
    # 或者输入的setting的长度是和depth的sum的长度等长， 就是len(element) == sum(depth)
    def get_same_length_settings(self, element, depth):
        if len(element) == len(depth):
            element_settings = []
            for i, d in enumerate(depth):
                element_settings += [element[i]] * d
        elif len(element) == sum(depth):
            element_settings = element
        else:
            raise ValueError('only support stage-wise or block-wise config')
        return element_settings

    # 返回和depth长度等长的element，返回stage-wise的配置
    # 当输入的setting的长度是stage_wise的，就是len(element) == len(depth)的情况，直接返回
    # 或者当输入的setting的长度是和depth的sum的长度等长， 就是len(element) == sum(depth)，需要缩短
    def get_same_depth_settings(self, element, depth):
        if len(element) == sum(depth):
            element_settings, index = [], 0
            for i in range(len(depth)):
                element_settings.append(element[index])
                index += depth[i]
        elif len(element) == len(depth):
            element_settings = element
        else:
            raise ValueError('only support stage-wise or block-wise config')
        return element_settings

    def set_active_subnet(self, **kwargs):
        # this is used when every stage in block is the same
        depth_settings = kwargs['depth']
        active_module_list = self.get_active_module_list(depth_settings)

        subnet_settings = {}
        for k, v in kwargs.items():
            if k == 'depth':
                subnet_settings[k] = v
                self.curr_subnet_settings[k] = v
                continue
            elif k == 'out_channel':
                new_processv = self.get_same_depth_settings(v, depth_settings)
                processv = self.get_same_length_settings(v, depth_settings)
                if 'out_channel' in self.dynamic_settings.keys() and \
                        'block' in self.dynamic_settings['out_channel']['sample_strategy']:
                    self.curr_subnet_settings[k] = processv
                else:
                    self.curr_subnet_settings[k] = new_processv
            else:
                processv = self.get_same_length_settings(v, depth_settings)
                self.curr_subnet_settings[k] = processv
            subnet_settings[k] = processv
            for block, value in zip(active_module_list, processv):
                attribute = 'active_' + k
                if hasattr(block, attribute):
                    setattr(block, attribute, value)
        return subnet_settings

    def get_fake_input(self, input_shape=[1, 3, 224, 224]):
        """Required.
        implement to get the fake input, it is used for all backbone.
        For necks with multiple input, rewrite this function
        """
        input = torch.randn(*input_shape)
        device = next(self.parameters(), torch.tensor([])).device
        input = input.to(device)
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) and m.weight.dtype == torch.float16:
                input = input.half()

        if not hasattr(self, 'task'):
            raise ValueError('default only support task with classification or detection')

        if self.task == 'classification':
            return input
        elif self.task == 'detection':
            b, c, height, width = map(int, input.size())
            input = {
                'image_info': [[height, width, 1.0, height, width, False]],
                'image': input,
                'filename': ['Test.jpg'],
                'label': torch.LongTensor([[0]]),
            }
            return input
        else:
            raise ValueError('only support classification & detection')

    def gen_dynamic_module_list(self, dynamic_types=None):
        dynamic_types = [i for i in DYNAMIC_BLOCKS_REGISTRY.values()] if dynamic_types is None else dynamic_types

        dynamic_module_list = {}
        handles = []
        input = self.get_fake_input()
        self.eval()

        def make_hook(module_name):
            def hook(module, data, out):
                remove_module_names = []
                for k, v in dynamic_module_list.items():
                    if module_name + '.' in k:
                        remove_module_names.append(k)
                for k in remove_module_names:
                    logger.info('unregister block for {}'.format(k))
                    del dynamic_module_list[k]
                if module_name not in self.remove_register_names:
                    dynamic_module_list[module_name] = module
                    logger.info('register block for {}'.format(module_name))
                else:
                    logger.info('do not register block for {}'.format(module_name))
            return hook

        for n, m in self.named_modules():
            if sum(map(lambda x: isinstance(m, x), dynamic_types)):
                handle = m.register_forward_hook(make_hook(n))
                handles.append(handle)
        self.forward(input)
        for handle in handles:
            handle.remove()

        self.train()
        logger.info('--------------Register Block----------------')
        for k, v in dynamic_module_list.items():
            logger.info('register block for {} with type {}'.format(k, type(v)))
        self.dynamic_module_list = list(dynamic_module_list.values())
        return

    def init_model(self):
        """ Conv2d, BatchNorm2d, BatchNorm1d, Linear, """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) and m.weight is not None and m.bias is not None:
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, link.nn.SyncBatchNorm2d) and m.weight is not None and m.bias is not None:
                m.weight.data.fill_(1)
                m.bias.data.fill_(1)
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 1.0 / float(n))
                if m.bias is not None:
                    m.bias.data.zero_()

    # 让主路上的BN的alpha等于0
    def zero_last_gamma(self):
        for m in self.modules():
            # if isinstance(m, DynamicMBConvBlock) or isinstance(m, DynamicMBConvBlock_Shortcut):
            #     if isinstance(m.shortcut, Identity):
            #         m.point_linear.bn.weight.data.zero_()
            if isinstance(m, DynamicBasicBlock):
                if isinstance(m.shortcut, Identity):
                    m.normal_conv2.bn.weight.data.zero_()
            # elif (isinstance(m, DynamicBottleneckBlock) or isinstance(m, DynamicRegBottleneckBlock)):
            #     if isinstance(m.shortcut, Identity):
            #         m.point_conv2.bn.weight.data.zero_()

    def modify_state_dict(self, state_dict, strict=True):
        new_state_dict = {}
        for k1, v1 in self.state_dict().items():
            if k1 in state_dict.keys():
                logger.info('keys {} has same names in supernet'.format(k1))
                v2 = state_dict[k1]
                if v1.dim() == v2.dim() and all([i <= j for i, j in zip(v1.size(), v2.size())]):
                    logger.info('provided ckpt larger dimension, we slice {} from {}'.format(v1.size(), v2.size()))
                    if v1.dim() == 0:
                        v1.copy_(copy.deepcopy(v2.data))
                    elif v1.dim() == 1:
                        v1.copy_(copy.deepcopy(v2[:v1.size(0)].data))
                    elif v1.dim() == 2:
                        v1.copy_(copy.deepcopy(v2[:v1.size(0), :v1.size(1)].data))
                    elif v1.dim() == 4:
                        start, end = sub_filter_start_end(v2.size(3), v1.size(3))
                        v1.copy_(copy.deepcopy(v2[:v1.size(0), :v1.size(1), start:end, start:end].data))
                    new_state_dict[k1] = v1
                else:
                    logger.info('provided ckpt has smaller dimension, we skip it {} from {}'.format(
                        v1.size(), v2.size()))
            else:
                logger.info('skip this layer {}'.format(k1))

        self.load_state_dict(new_state_dict, strict)
