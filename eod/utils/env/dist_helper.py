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
from eod.utils.general.global_flag import DIST_BACKEND
try:
    import spring.linklink as link
except:   # noqa
    link = None

MASTER_RANK = 0
_LOCAL_PROCESS_GROUP = None


def allreduce(*args, **kwargs):
    if DIST_BACKEND.backend == 'linklink':
        return link.allreduce(*args, **kwargs)
    elif DIST_BACKEND.backend == 'dist':
        return dist.all_reduce(*args, **kwargs)
    else:
        raise NotImplementedError


def broadcast(*args, **kwargs):
    if DIST_BACKEND.backend == 'linklink':
        return link.broadcast(*args, **kwargs)
    elif DIST_BACKEND.backend == 'dist':
        return dist.broadcast(*args, **kwargs)
    else:
        raise NotImplementedError


def allgather(*args, **kwargs):
    if DIST_BACKEND.backend == 'linklink':
        return link.allgather(*args, **kwargs)
    elif DIST_BACKEND.backend == 'dist':
        return dist.all_gather(*args, **kwargs)
    else:
        raise NotImplementedError


def gather(*args, **kwargs):
    if DIST_BACKEND.backend == 'linklink':
        return link.gather(*args, **kwargs)
    elif DIST_BACKEND.backend == 'dist':
        return dist.gather(*args, **kwargs)
    else:
        raise NotImplementedError


def get_dist_rank(*args, **kwargs):
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank(*args, **kwargs)


def get_link_rank(*args, **kwargs):
    return link.get_rank(*args, **kwargs)


def get_dist_world_size(*args, **kwargs):
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size(*args, **kwargs)


def get_link_world_size(*args, **kwargs):
    return link.get_world_size(*args, **kwargs)


def get_rank_from_env():
    rank_cands = ['SLURM_PROCID', 'MV2_COMM_WORLD_RANK', 'PMI_RANK']
    for rank_name in rank_cands:
        if rank_name in os.environ:
            return int(os.environ[rank_name])
    return None


def get_world_size_from_env():
    ws_cands = ['SLURM_NTASKS', 'MV2_COMM_WORLD_SIZE', 'PMI_SIZE']
    for ws_name in ws_cands:
        if ws_name in os.environ:
            return int(os.environ[ws_name])
    return None


def dist_barrier(*args, **kwargs):
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier(*args, **kwargs)


def link_barrier(*args, **kwargs):
    return link.barrier(*args, **kwargs)


def barrier(*args, **kwargs):
    if DIST_BACKEND.backend == 'linklink':
        link_barrier(*args, **kwargs)
    elif DIST_BACKEND.backend == 'dist':
        dist_barrier(*args, **kwargs)
    else:
        raise NotImplementedError


def get_rank(*args, **kwargs):
    rank = get_rank_from_env()
    if rank is not None:
        return rank
    if DIST_BACKEND.backend == 'linklink':
        return get_link_rank(*args, **kwargs)
    elif DIST_BACKEND.backend == 'dist':
        return get_dist_rank(*args, **kwargs)
    else:
        raise NotImplementedError


def get_world_size(*args, **kwargs):
    world_size = get_world_size_from_env()
    if world_size is not None:
        return world_size
    if DIST_BACKEND.backend == 'linklink':
        return get_link_world_size(*args, **kwargs)
    elif DIST_BACKEND.backend == 'dist':
        return get_dist_world_size(*args, **kwargs)
    else:
        raise NotImplementedError


def get_local_rank(*args, **kwargs):
    rank = get_rank(*args, **kwargs)
    return rank % torch.cuda.device_count()


def all_reduce(*args, **kwargs):
    if DIST_BACKEND.backend == 'linklink':
        return link.allreduce(*args, **kwargs)
    elif DIST_BACKEND.backend == 'dist':
        return dist.all_reduce(*args, **kwargs)
    else:
        raise NotImplementedError


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


def DistModule(model, sync=True):
    def _register_hooks(self):
        for i, (name, p) in enumerate(self.named_parameters()):
            if p.requires_grad:
                p_tmp = p.expand_as(p)
                grad_acc = p_tmp.grad_fn.next_functions[0][0]
                grad_acc.register_hook(self._make_hook(name, p, i))
                self._grad_accs.append(grad_acc)

    def _make_hook(name, p, i):
        def hook(*ignore):
            link.allreduce_async(name, p.grad.data)
        return hook

    def broadcast_params(model):
        for name, p in model.state_dict().items():
            link.broadcast(p, MASTER_RANK)

    broadcast_params(model)
    if not sync:
        model._grad_accs = []
        model._register_hooks = _register_hooks
        model._make_hook = _make_hook
        model._register_hooks(model)
    return model


def reduce_gradients(model, sync=True, allow_dead_parameter=False):
    if sync:
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                try:
                    link.allreduce(param.grad.data)
                except AttributeError as e:
                    warning = (f"The AttributeError above was probably caused by {name}.grad being None "
                               f"but the gradient w.r.t {name} is required. "
                               f"Please check your model to make sure that {name} is always used in your "
                               "forward pass if it is learnable, otherwise, set it's requrires_grad flag to False. "
                               "Another temporary workaround, you may add 'and param.grad is not None' to "
                               "the conditional above only if you know exactly the grad is needless.")
                    if not allow_dead_parameter:
                        raise AttributeError('This line is not an error, just warning: ' + warning) from e
    else:
        # reduce all grdients asynchronously, faster
        link.synchronize()


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


def setup_distributed_dist(port, launch):
    if launch == 'slurm':
        device = get_rank() % torch.cuda.device_count()
        torch.cuda.set_device(device)
        dist_init_slurm(port=port)
    else:
        dist_init_mpi(port=port)
        device = get_rank() % torch.cuda.device_count()
        torch.cuda.set_device(device)


def setup_distributed_link(port, launch):
    if 'SLURM_PROCID' in os.environ:    # slurm mode
        device = get_rank() % torch.cuda.device_count()
        torch.cuda.set_device(device)
        link.initialize()
    else:
        link.initialize()
        device = get_rank() % torch.cuda.device_count()
        torch.cuda.set_device(device)


def setup_distributed(port=33334, launch='slurm', backend='dist'):
    if backend == 'dist':
        setup_distributed_dist(port, launch)
    else:
        setup_distributed_link(port, launch)
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
    if DIST_BACKEND.backend == 'dist':
        if dist.get_backend() == "nccl":
            return dist.new_group(backend="gloo")
        else:
            return dist.group.WORLD
    elif DIST_BACKEND.backend == 'linklink':
        if link.get_backend() == "nccl":
            return link.new_group(backend="gloo")
        else:
            return link.group.WORLD
    else:
        raise NotImplementedError


def _serialize_to_tensor(data, group):
    if DIST_BACKEND.backend == 'dist':
        backend = dist.get_backend(group)
        assert backend in ["gloo", "nccl"]
        device = torch.device("cpu" if backend == "gloo" else "cuda")
    else:
        device = torch.device("cpu")

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
    if DIST_BACKEND.backend == 'dist':
        world_size = get_world_size(group=group)
    else:
        world_size = get_world_size()
    assert (
        world_size >= 1
    ), "comm.gather/all_gather must be called from ranks within the given group!"
    local_size = torch.tensor([tensor.numel()], dtype=torch.int64, device=tensor.device)
    size_list = [
        torch.zeros([1], dtype=torch.int64, device=tensor.device)
        for _ in range(world_size)
    ]
    if DIST_BACKEND.backend == 'dist':
        allgather(size_list, local_size, group=group)
    else:
        allgather(size_list, local_size)
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
    if DIST_BACKEND.backend == 'dist':
        if group is None:
            group = _get_global_gloo_group()
        if get_world_size(group) == 1:
            return [data]

    tensor = _serialize_to_tensor(data, group)

    size_list, tensor = _pad_to_largest_tensor(tensor, group)
    max_size = max(size_list)

    # receiving Tensor from all ranks
    tensor_list = [
        torch.empty((max_size,), dtype=torch.uint8, device=tensor.device)
        for _ in size_list
    ]
    if DIST_BACKEND.backend == 'dist':
        allgather(tensor_list, tensor, group=group)
    else:
        allgather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def gather_pk(data, dst=0, group=None):
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
    if DIST_BACKEND.backend == 'dist':
        if group is None:
            group = _get_global_gloo_group()
        if get_world_size(group=group) == 1:
            return [data]
        rank = get_rank(group=group)
    else:
        rank = get_rank()

    tensor = _serialize_to_tensor(data, group)
    size_list, tensor = _pad_to_largest_tensor(tensor, group)

    # receiving Tensor from all ranks
    if rank == dst:
        max_size = max(size_list)
        tensor_list = [
            torch.empty((max_size,), dtype=torch.uint8, device=tensor.device)
            for _ in size_list
        ]
        if DIST_BACKEND.backend == 'dist':
            gather(tensor, tensor_list, dst=dst, group=group)
        else:
            gather(tensor, tensor_list, dst=dst)
        data_list = []
        for size, tensor in zip(size_list, tensor_list):
            buffer = tensor.cpu().numpy().tobytes()[:size]
            data_list.append(pickle.loads(buffer))
        return data_list
    else:
        if DIST_BACKEND.backend == 'dist':
            gather(tensor, [], dst=dst, group=group)
        else:
            gather(tensor, [], dst=dst)
        return []


def finalize():
    if DIST_BACKEND.backend == 'dist':
        dist.destroy_process_group()
    elif DIST_BACKEND.backend == 'linklink':
        link.finalize()
    else:
        raise NotImplementedError


def pyobj2tensor(pyobj, device="cuda"):
    """serialize picklable python object to tensor"""
    storage = torch.ByteStorage.from_buffer(pickle.dumps(pyobj))
    return torch.ByteTensor(storage).to(device=device)


def tensor2pyobj(tensor):
    """deserialize tensor to picklable python object"""
    return pickle.loads(tensor.cpu().numpy().tobytes())


def all_reduce_dict(py_dict, op="sum", group=None):
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
    if DIST_BACKEND.backend == 'dist':
        broadcast(py_key_tensor, src=MASTER_RANK)
    else:
        broadcast(py_key_tensor, MASTER_RANK)
    py_key = tensor2pyobj(py_key_tensor)

    tensor_shapes = [py_dict[k].shape for k in py_key]
    tensor_numels = [py_dict[k].numel() for k in py_key]

    flatten_tensor = torch.cat([py_dict[k].flatten() for k in py_key])
    all_reduce(flatten_tensor)
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
    states = all_reduce_dict(states, op="mean")
    module.load_state_dict(states, strict=False)
