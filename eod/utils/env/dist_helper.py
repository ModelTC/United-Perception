# Standard Library
import os
import sys
import functools

# Import from third library
import torch
import torch.distributed as dist
import torch.nn as nn
import pickle
from collections import OrderedDict

allreduce = dist.all_reduce
allgather = dist.all_gather
broadcast = dist.broadcast
synchronize = torch.cuda.synchronize
init_process_group = dist.init_process_group


MASTER_RANK = 0
_LOCAL_PROCESS_GROUP = None


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    # if 'SLURM_PROCID' in os.environ:
    #     return int(os.environ.get('SLURM_NTASKS', 1))
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def barrier():
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def get_local_rank():
    rank = get_rank()
    return rank % torch.cuda.device_count()


class DistEnv(object):
    @property
    def rank(self):
        self._rank = get_rank()
        return self._rank

    @property
    def local_rank(self):
        self._local_rank = get_local_rank()
        return self._local_rank

    @property
    def world_size(self):
        self._world_size = get_world_size()
        return self._world_size

    def is_master(self):
        return self.rank == MASTER_RANK

    @property
    def distributed(self):
        return self.world_size > 1


env = DistEnv()


# TODO support free port
def find_free_port():
    import socket
    from contextlib import closing
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        return str(s.getsockname()[1])


def dist_init_slurm(backend='nccl', port='13333'):
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    if '[' in node_list:
        beg = node_list.find('[')
        pos1 = node_list.find('-', beg)
        if pos1 < 0:
            pos1 = 1000
        pos2 = node_list.find(',', beg)
        if pos2 < 0:
            pos2 = 1000
        node_list = node_list[:min(pos1, pos2)].replace('[', '')
    addr = node_list[8:].replace('-', '.')
    os.environ['MASTER_PORT'] = str(port)
    os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['RANK'] = str(proc_id)
    if backend == 'nccl':
        dist.init_process_group(backend='nccl')
    else:
        dist.init_process_group(backend='gloo', rank=proc_id, world_size=ntasks)
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)


def dist_init_mpi(backend='nccl',
                  addr='localhost',
                  port="13333",
                  rank=None,
                  world_size=None):
    r"""
    Overview:
        Init the distributed training setting
    """
    assert backend in ['nccl', 'gloo'], backend
    os.environ['MASTER_ADDR'] = addr or os.environ.get('MASTER_ADDR', addr)
    os.environ['MASTER_PORT'] = str(port) or os.environ.get('MASTER_PORT', str(port))

    if rank is None:
        local_id = os.environ.get('SLURM_LOCALID', os.environ.get('PMI_RANK', None))
        if local_id is None:
            raise RuntimeError("please indicate rank explicitly in dist_init method")
        else:
            rank = int(local_id)
    if world_size is None:
        ntasks = os.environ.get('SLURM_NTASKS', os.environ.get('PMI_SIZE', None))
        if ntasks is None:
            raise RuntimeError("please indicate world_size explicitly in dist_init method")
        else:
            world_size = int(ntasks)

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)


def setup_distributed(port=33334, launch='slurm'):
    if launch == 'slurm':
        device = get_rank() % torch.cuda.device_count()
        torch.cuda.set_device(device)
        dist_init_slurm(port=port)
    else:
        dist_init_mpi(port=port)
        device = get_rank() % torch.cuda.device_count()
        torch.cuda.set_device(device)
    rank = get_rank()
    world_size = get_world_size()
    print('rank:{} world_size(gpus):{}'.format(rank, world_size))
    sys.stdout.flush()
    return rank, world_size


@functools.lru_cache()
def _get_global_gloo_group():
    """
    Return a process group based on gloo backend, containing all the ranks
    The result is cached.
    """
    if dist.get_backend() == "nccl":
        return dist.new_group(backend="gloo")
    else:
        return dist.group.WORLD


def _serialize_to_tensor(data, group):
    backend = dist.get_backend(group)
    assert backend in ["gloo", "nccl"]
    device = torch.device("cpu" if backend == "gloo" else "cuda")

    buffer = pickle.dumps(data)
    if len(buffer) > 1024 ** 3:
        import logging
        logger = logging.getLogger('global')
        logger.warning(
            "Rank {} trying to all-gather {:.2f} GB of data on device {}".format(
                get_rank(), len(buffer) / (1024 ** 3), device
            )
        )
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(device=device)
    return tensor


def _pad_to_largest_tensor(tensor, group):
    """
    Returns:
        list[int]: size of the tensor, on each rank
        Tensor: padded tensor that has the max size
    """
    world_size = dist.get_world_size(group=group)
    assert (
        world_size >= 1
    ), "comm.gather/all_gather must be called from ranks within the given group!"
    local_size = torch.tensor([tensor.numel()], dtype=torch.int64, device=tensor.device)
    size_list = [
        torch.zeros([1], dtype=torch.int64, device=tensor.device)
        for _ in range(world_size)
    ]
    dist.all_gather(size_list, local_size, group=group)
    size_list = [int(size.item()) for size in size_list]

    max_size = max(size_list)

    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    if local_size != max_size:
        padding = torch.zeros(
            (max_size - local_size,), dtype=torch.uint8, device=tensor.device
        )
        tensor = torch.cat((tensor, padding), dim=0)
    return size_list, tensor


def all_gather(data, group=None):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors).
    Args:
        data: any picklable object
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.
    Returns:
        list[data]: list of data gathered from each rank
    """
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()
    if dist.get_world_size(group) == 1:
        return [data]

    tensor = _serialize_to_tensor(data, group)

    size_list, tensor = _pad_to_largest_tensor(tensor, group)
    max_size = max(size_list)

    # receiving Tensor from all ranks
    tensor_list = [
        torch.empty((max_size,), dtype=torch.uint8, device=tensor.device)
        for _ in size_list
    ]
    dist.all_gather(tensor_list, tensor, group=group)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def gather(data, dst=0, group=None):
    """
    Run gather on arbitrary picklable data (not necessarily tensors).
    Args:
        data: any picklable object
        dst (int): destination rank
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.
    Returns:
        list[data]: on dst, a list of data gathered from each rank. Otherwise,
            an empty list.
    """
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()
    if dist.get_world_size(group=group) == 1:
        return [data]
    rank = dist.get_rank(group=group)

    tensor = _serialize_to_tensor(data, group)
    size_list, tensor = _pad_to_largest_tensor(tensor, group)

    # receiving Tensor from all ranks
    if rank == dst:
        max_size = max(size_list)
        tensor_list = [
            torch.empty((max_size,), dtype=torch.uint8, device=tensor.device)
            for _ in size_list
        ]
        dist.gather(tensor, tensor_list, dst=dst, group=group)

        data_list = []
        for size, tensor in zip(size_list, tensor_list):
            buffer = tensor.cpu().numpy().tobytes()[:size]
            data_list.append(pickle.loads(buffer))
        return data_list
    else:
        dist.gather(tensor, [], dst=dst, group=group)
        return []


def dist_finalize():
    r"""
    Overview:
        Finalize distributed training resources
    """
    dist.destroy_process_group()


def pyobj2tensor(pyobj, device="cuda"):
    """serialize picklable python object to tensor"""
    storage = torch.ByteStorage.from_buffer(pickle.dumps(pyobj))
    return torch.ByteTensor(storage).to(device=device)


def tensor2pyobj(tensor):
    """deserialize tensor to picklable python object"""
    return pickle.loads(tensor.cpu().numpy().tobytes())


def all_reduce(py_dict, op="sum", group=None):
    """
    Apply all reduce function for python dict object.
    NOTE: make sure that every py_dict has the same keys and values are in the same shape.

    Args:
        py_dict (dict): dict to apply all reduce op.
        op (str): operator, could be "sum" or "mean".
    """
    world_size = get_world_size()
    if world_size == 1:
        return py_dict
    # all reduce logic across different devices.
    py_key = list(py_dict.keys())
    py_key_tensor = pyobj2tensor(py_key)
    dist.broadcast(py_key_tensor, src=MASTER_RANK)
    py_key = tensor2pyobj(py_key_tensor)

    tensor_shapes = [py_dict[k].shape for k in py_key]
    tensor_numels = [py_dict[k].numel() for k in py_key]

    flatten_tensor = torch.cat([py_dict[k].flatten() for k in py_key])
    dist.all_reduce(flatten_tensor)
    if op == "mean":
        flatten_tensor /= world_size
    split_tensors = [
        x.reshape(shape)
        for x, shape in zip(torch.split(flatten_tensor, tensor_numels), tensor_shapes)
    ]
    return OrderedDict({k: v for k, v in zip(py_key, split_tensors)})


def get_async_norm_states(module):
    async_norm_states = OrderedDict()
    ASYNC_NORM = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
    for name, child in module.named_modules():
        if isinstance(child, ASYNC_NORM):
            for k, v in child.state_dict().items():
                async_norm_states[".".join([name, k])] = v
    return async_norm_states


def all_reduce_norm(module):
    """
    All reduce norm statistics in different devices.
    """
    states = get_async_norm_states(module)
    states = all_reduce(states, op="mean")
    module.load_state_dict(states, strict=False)
