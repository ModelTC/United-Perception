import random


def net_wise(stage_wise_length, stage_wise_range):
    """process random sample settings
    len(search_settings) equals sum(stage_wise_length)

    Args:
        stage_wise_length (list of int): the length of each stage
        stage_wise_range (list of list): the range to be sampled of each stage

    Returns:
        search_settings (list of int): the processed settings with one same number
    """
    assert len(stage_wise_length) == len(stage_wise_range)
    for i in stage_wise_range:
        if i != stage_wise_range[0]:
            raise ValueError('With net-wise sample strategy, the stage-wise range should be the same')

    search_settings = [random.choice(stage_wise_range[0])] * sum(stage_wise_length)
    return search_settings


def net_wise_index(stage_wise_length, stage_wise_range):
    """process random sample settings
    len(search_settings) equals sum(stage_wise_length)

    Args:
        stage_wise_length (list of int): the length of each stage
        stage_wise_range (list of list): the range to be sampled of each stage

    Returns:
        search_settings (list of int): the processed settings with one same number
    """
    assert len(stage_wise_length) == len(stage_wise_range)
    for i in stage_wise_range:
        if len(i) == 1:
            continue
        if len(i) != len(stage_wise_range[1]):
            raise ValueError('With net-wise-index sample strategy, the stage-wise length should be the same')

    index = random.choice(range(len(stage_wise_range[0])))
    search_settings = [i[index] if len(i) != 1 else i[0] for i in stage_wise_range]
    return search_settings


def stage_wise(stage_wise_length, stage_wise_range, mode='sum'):
    """process random sample settings
    len(search_settings) equals sum(stage_wise_length)

    Args:
        stage_wise_length (list of int): the length of each stage
        stage_wise_range (list of list): the range to be sampled of each stage
        mode (str): sum means len(search_settings) equals sum(stage_wise_length)
                    same means len(search_settings) equals len(stage_wise_length)

    Returns:
        search_settings (list of int): the processed settings with one same number for every stage
    """
    search_settings = []
    for depth, curr_range in zip(stage_wise_length, stage_wise_range):
        curr = random.choice(curr_range)
        if mode == 'same':
            depth = 1
        search_settings += [curr] * depth
    return search_settings


def get_next_ordered_element(last, curr_range, min_slope=1, max_slope=1e3):
    if max(curr_range) < last * min_slope:
        raise ValueError('wrong range and min_slope')
    while 1:
        next_element = random.choice(curr_range)
        if next_element >= last * min_slope and next_element <= last * max_slope:
            break
    return next_element


def ordered_stage_wise(stage_wise_length, stage_wise_range, mode='sum',
                       start_stage=None, end_stage=None,
                       min_slope=1, max_slope=1e3
                       ):
    """process random sample settings
    len(search_settings) equals sum(stage_wise_length)

    Args:
        stage_wise_length (list of int): the length of each stage
        stage_wise_range (list of list): the range to be sampled of each stage
        start_stage (int or None): the start stage number in ascending order
        end_stage (int or None): the end stage number in ascending order
        min_slope (float): the minimum slope
        max_slope (float): the maximum slope

    Returns:
        search_settings (list of int): the processed settings with one same number for every stage in ascending order
    """
    if start_stage is None:
        start_stage = 0
    if end_stage is None:
        end_stage = len(stage_wise_length)

    search_settings = []
    for stage_num, (depth, curr_range) in enumerate(zip(stage_wise_length, stage_wise_range)):
        if mode == 'same':
            depth = 1
        elif mode == 'sum':
            depth = depth
        else:
            raise ValueError('Sample mode only supports same and sum')
        if search_settings != [] and stage_num > start_stage and stage_num < end_stage + 1:
            curr = get_next_ordered_element(search_settings[-1], curr_range, min_slope, max_slope)
        else:
            curr = random.choice(curr_range)
        search_settings += [curr] * depth
    return search_settings


def stage_wise_depth(stage_wise_length, stage_wise_range, depth_dynamic_range=None):
    """process random sample settings
    len(search_settings) equals len(stage_wise_length)

    Args:
        stage_wise_length (list of int): the length of each stage
        stage_wise_range (list of list): the range to be sampled of each stage

    Returns:
        search_settings (list of int): the processed settings with one same number for every stage
    """
    if depth_dynamic_range is None:
        search_settings = []
        for _, curr_range in zip(stage_wise_length, stage_wise_range):
            curr = random.choice(curr_range)
            search_settings += [curr]
    else:
        # we compute the depth index in depth_dynamic_range
        # then the element is selected to be the corresponding one
        depth_index = [depth_range.index(i) for i, depth_range in zip(stage_wise_length, depth_dynamic_range)]
        search_settings = []
        for (i, element_range) in enumerate(stage_wise_range):
            if len(depth_dynamic_range[i]) != len(element_range):
                cur = random.choice(element_range)
                search_settings.append(cur)
            else:
                cur_index = depth_index[i]
                search_settings += [element_range[cur_index]]

    return search_settings


def ordered_stage_wise_depth(stage_wise_length, stage_wise_range,
                             start_stage=None, end_stage=None,
                             min_slope=1, max_slope=1e3):
    """process random sample settings
    len(search_settings) equals len(stage_wise_length)

    Args:
        stage_wise_length (list of int): the length of each stage
        stage_wise_range (list of list): the range to be sampled of each stage

    Returns:
        search_settings (list of int): the processed settings with one same number for every stage in ascending order
    """
    if start_stage is None:
        start_stage = 0
    if end_stage is None:
        end_stage = len(stage_wise_length)

    search_settings = []
    for stage_num, (_, curr_range) in enumerate(zip(stage_wise_length, stage_wise_range)):
        if search_settings != [] and stage_num > start_stage and stage_num < end_stage + 1:
            curr = get_next_ordered_element(search_settings[-1], curr_range, min_slope, max_slope)
        else:
            curr = random.choice(curr_range)
        search_settings += [curr]
    return search_settings


def block_wise(stage_wise_length, stage_wise_range):
    """process random sample settings
    len(search_settings) equals sum(stage_wise_length)

    Args:
        stage_wise_length (list of int): the length of each stage
        stage_wise_range (list of list): the range to be sampled of each stage

    Returns:
        search_settings (list of int): the processed settings with one same number for every block
    """
    search_settings = []
    for depth, curr_range in zip(stage_wise_length, stage_wise_range):
        for _ in range(depth):
            curr = random.choice(curr_range)
            search_settings.append(curr)
    return search_settings


def ordered_block_wise(stage_wise_length, stage_wise_range):
    """process random sample settings
    len(search_settings) equals sum(stage_wise_length)

    Args:
        stage_wise_length (list of int): the length of each stage
        stage_wise_range (list of list): the range to be sampled of each stage

    Returns:
        search_settings (list of int): the processed settings with one same number for every block in ascending order
    """
    search_settings = []
    for depth, curr_range in zip(stage_wise_length, stage_wise_range):
        for i in range(depth):
            curr = random.choice(curr_range)
            if search_settings:
                while curr < search_settings[-1]:
                    curr = random.choice(curr_range)
                search_settings.append(curr)
            else:
                search_settings.append(curr)
    return search_settings


def sample_search_settings(stage_wise_length, stage_wise_range, sample_strategy):
    """process random search settings

    Args:
        stage_wise_length (list of int): stage wise length
        stage_wise_range (list of list): stage wise range
        sample_strategy (str): sample strategy

    Returns:
        search_settings (list of int): the processed settings
    """
    assert len(stage_wise_length) == len(stage_wise_range)
    if isinstance(sample_strategy, str):
        kwargs = {}
        sample_method = sample_strategy
    elif isinstance(sample_strategy, dict):
        kwargs = sample_strategy['kwargs']
        sample_method = sample_strategy['name']
    else:
        raise ValueError('Only Support Sample Strategy with str or dict')

    if sample_method == 'net_wise':
        return net_wise(stage_wise_length, stage_wise_range, **kwargs)
    elif sample_method == 'net_wise_index':
        return net_wise_index(stage_wise_length, stage_wise_range, **kwargs)
    elif sample_method == 'stage_wise':
        return stage_wise(stage_wise_length, stage_wise_range, **kwargs)
    elif sample_method == 'ordered_stage_wise':
        return ordered_stage_wise(stage_wise_length, stage_wise_range, **kwargs)
    elif sample_method == 'stage_wise_depth':
        return stage_wise_depth(stage_wise_length, stage_wise_range, **kwargs)
    elif sample_method == 'ordered_stage_wise_depth':
        return ordered_stage_wise_depth(stage_wise_length, stage_wise_range, **kwargs)
    elif sample_method == 'block_wise':
        return block_wise(stage_wise_length, stage_wise_range, **kwargs)
    elif sample_method == 'ordered_block_wise':
        return ordered_block_wise(stage_wise_length, stage_wise_range, **kwargs)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    stage_wise_length = [1, 1, 2, 3, 4, 5, 1]
    stage_wise_range = [[i for i in range(32, 64 + 1, 16)]] * 7
    print('net_wise', sample_search_settings(stage_wise_length, stage_wise_range, 'net_wise'))
    stage_wise_range = [[32], [64], [i for i in range(64, 96 + 1, 16)], [i for i in range(128, 192 + 1, 16)],
                        [i for i in range(256, 384 + 1, 16)], [i for i in range(512, 768 + 1, 16)], [1024]]
    for method in ['stage_wise', 'ordered_stage_wise',
                   'stage_wise_depth', 'ordered_stage_wise_depth', 'block_wise', 'ordered_block_wise']:
        print(method, sample_search_settings(stage_wise_length, stage_wise_range, method))

    method = {'name': 'ordered_stage_wise',
              'kwargs': {'start_stage': 2, 'end_stage': 5, 'min_slope': 1.5, 'max_slope': 2.5}}
    print(method, sample_search_settings(stage_wise_length, stage_wise_range, method))
    method = {'name': 'ordered_stage_wise',
              'kwargs': {'mode': 'same', 'start_stage': 2, 'end_stage': 5, 'min_slope': 1.5, 'max_slope': 2.5}}
    print(method, sample_search_settings(stage_wise_length, stage_wise_range, method))
    method = {'name': 'ordered_stage_wise_depth',
              'kwargs': {'start_stage': 2, 'end_stage': 5, 'min_slope': 1.5, 'max_slope': 2.5}}
    print(method, sample_search_settings(stage_wise_length, stage_wise_range, method))

    stage_wise_range = [[32, 64, 128], [32, 64, 128], [32, 64, 128], [32, 64, 128], [32, 64, 128], [32, 64, 128], [128]]
    method = {'name': 'stage_wise_depth',
              'kwargs': {'depth_dynamic_range': [[1], [1], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [1]]}}
    print(method, sample_search_settings(stage_wise_length, stage_wise_range, method))
