#!/usr/bin/env python3
# Code are based on
# https://github.com/facebookresearch/detectron2/blob/master/detectron2/engine/launch.py
# Copyright (c) Facebook, Inc. and its affiliates.

from datetime import timedelta
from ..general.log_helper import default_logger as logger

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from eod.utils.env import dist_helper as dist_helper

__all__ = ["launch"]

DEFAULT_TIMEOUT = timedelta(minutes=30)


def launch(
    main_func,
    num_gpus_per_machine,
    num_machines=1,
    machine_rank=0,
    backend="nccl",
    dist_url='auto',
    args=(),
    timeout=DEFAULT_TIMEOUT,
    start_method='fork'
):
    """
    Args:
        main_func: a function that will be called by `main_func(*args)`
        num_machines (int): the total number of machines
        machine_rank (int): the rank of this machine (one per machine)
        dist_url (str): url to connect to for distributed training, including protocol
                       e.g. "tcp://127.0.0.1:8686".
                       Can be set to auto to automatically select a free port on localhost
        args (tuple): arguments passed to main_func
    """
    world_size = num_machines * num_gpus_per_machine
    if world_size > 1:
        # https://github.com/pytorch/pytorch/pull/14391
        # TODO prctl in spawned processes

        if dist_url == "auto":
            assert (
                num_machines == 1
            ), "dist_url=auto cannot work with distributed training."
            port = dist_helper.find_free_port()
            dist_url = f"tcp://127.0.0.1:{port}"

        mp.start_processes(
            _distributed_worker,
            nprocs=num_gpus_per_machine,
            args=(
                main_func,
                world_size,
                num_gpus_per_machine,
                machine_rank,
                backend,
                dist_url,
                args,
            ),
            daemon=False,
            start_method=start_method,
        )
    else:
        main_func(*(args,))


def _distributed_worker(
    local_rank,
    main_func,
    world_size,
    num_gpus_per_machine,
    machine_rank,
    backend,
    dist_url,
    args,
    timeout=DEFAULT_TIMEOUT,
):
    assert (
        torch.cuda.is_available()
    ), "cuda is not available. Please check your installation."
    global_rank = machine_rank * num_gpus_per_machine + local_rank
    logger.info("Rank {} initialization finished.".format(global_rank))
    try:
        dist.init_process_group(
            backend=backend,
            init_method=dist_url,
            world_size=world_size,
            rank=global_rank,
            timeout=timeout,
        )
    except Exception:
        logger.error("Process group URL: {}".format(dist_url))
        raise

    # Setup the local process group (which contains ranks within the same machine)
    assert dist_helper._LOCAL_PROCESS_GROUP is None
    num_machines = world_size // num_gpus_per_machine
    for i in range(num_machines):
        ranks_on_i = list(
            range(i * num_gpus_per_machine, (i + 1) * num_gpus_per_machine)
        )
        pg = dist.new_group(ranks_on_i)
        if i == machine_rank:
            dist_helper._LOCAL_PROCESS_GROUP = pg

    # synchronize is needed here to prevent a possible timeout after calling init_process_group
    # See: https://github.com/facebookresearch/maskrcnn-benchmark/issues/172

    dist_helper.barrier()

    assert num_gpus_per_machine <= torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    main_func(args)
