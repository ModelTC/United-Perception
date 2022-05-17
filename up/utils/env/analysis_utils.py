import psutil
from .dist_helper import env


def get_sigle_node_memory_info(get_total=False, node=0):
    mem = psutil.virtual_memory()
    memory_info = {}
    prefix = 'node'
    if get_total:
        mem_total = mem.total / 1024 / 1024 / 1024.
        memory_info[f'{prefix}_mem_total'] = round(mem_total, 3)
    mem_used = mem.used / 1024 / 1024 / 1024.
    memory_info[f'{prefix}_mem_used'] = round(mem_used, 3)
    mem_used_per = mem.percent
    memory_info[f'{prefix}_mem_used_percent'] = mem_used_per
    swap_mem = psutil.swap_memory()
    if get_total:
        swap_mem_total = swap_mem.total / 1024 / 1024 / 1024.
        memory_info[f'{prefix}_swap_mem_total'] = round(swap_mem_total, 3)
    swap_mem_per = swap_mem.percent
    memory_info[f'{prefix}_swap_mem_used_percent'] = swap_mem_per
    return memory_info


def get_memory_info(get_total=False, gpu_per_node=8):
    memory_info = {}
    node_list = split_node(gpu_per_node)
    if env.rank in node_list:
        temp_info = get_sigle_node_memory_info(get_total, env.rank // gpu_per_node)
        memory_info.update(temp_info)
    return memory_info, node_list


def split_node(gpu_per_node=8):
    world_size = env.world_size
    if world_size <= gpu_per_node:
        return [0]
    assert world_size % gpu_per_node == 0
    max_node = world_size // gpu_per_node
    return [i * gpu_per_node for i in range(max_node)]
