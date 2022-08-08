import itertools


def net_wise(stage_wise_range):
    """process random sample settings
    len(search_settings) equals sum(stage_wise_length)

    Args:
        stage_wise_range (list of list): the range to be sampled of each stage

    Returns:
        search_settings (list of int): the processed settings with one same number
    """
    traverse_ranges = []
    # skip element with only one range in the beginning
    element_with_only_one = []
    for i in stage_wise_range:
        if len(i) == 1:
            element_with_only_one.append(i[0])
    start_index = len(element_with_only_one)
    if start_index == len(stage_wise_range):
        return [element_with_only_one]

    for i in range(len(stage_wise_range[start_index])):
        cur = element_with_only_one + [range[i] for range in stage_wise_range[start_index:]]
        traverse_ranges.append(cur)
    return traverse_ranges


def stage_wise(stage_wise_range):
    """process random sample settings

    Args:
        stage_wise_range (list of list): the range to be sampled of each stage

    Returns:
        search_settings (list of int): the processed settings with one same number for every stage
    """
    traverse_range = []
    for i in itertools.product(*stage_wise_range):
        traverse_range.append(list(i))
    return traverse_range


def ordered_stage_wise(stage_wise_range, mode='sum',
                       start_stage=None, end_stage=None,
                       min_slope=1, max_slope=1e3
                       ):
    """process random sample settings

    Args:
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
        end_stage = len(stage_wise_range)

    traverse_ranges = []
    for setting in itertools.product(*stage_wise_range):
        setting = list(setting)
        insert = True
        for i in range(start_stage + 1, end_stage):
            if setting[i] < setting[i - 1] * min_slope or setting[i] > setting[i - 1] * max_slope:
                insert = False
                break
        if insert:
            traverse_ranges.append(setting)

    return traverse_ranges


def traverse_search_range(stage_wise_range, sample_strategy):
    """traver search range and produce all possibilities

    Args:
        stage_wise_range (list of list): stage wise range
        sample_strategy (str): sample strategy

    Returns:
        search_settings (list of int): the processed settings
    """
    if isinstance(sample_strategy, str):
        kwargs = {}
        sample_method = sample_strategy
    elif isinstance(sample_strategy, dict):
        kwargs = sample_strategy['kwargs']
        sample_method = sample_strategy['name']
    else:
        raise ValueError('Only Support Sample Strategy with str or dict')

    if sample_method == 'net_wise':
        return net_wise(stage_wise_range, **kwargs)
    elif sample_method == 'net_wise_index':
        return net_wise(stage_wise_range, **kwargs)
    elif sample_method == 'stage_wise':
        return stage_wise(stage_wise_range, **kwargs)
    elif sample_method == 'ordered_stage_wise':
        return ordered_stage_wise(stage_wise_range, **kwargs)
    elif sample_method == 'stage_wise_depth':
        return stage_wise(stage_wise_range, **kwargs)
    elif sample_method == 'ordered_stage_wise_depth':
        return ordered_stage_wise(stage_wise_range, **kwargs)
        raise NotImplementedError


if __name__ == '__main__':
    stage_wise_range = [[32]] * 2 + [[i for i in range(32, 64 + 1, 16)]] * 3
    for method in ['stage_wise', 'ordered_stage_wise', 'stage_wise_depth', 'ordered_stage_wise_depth', 'net_wise']:
        ranges = traverse_search_range(stage_wise_range, method)
        print(method, len(ranges), ranges[0], ranges)
