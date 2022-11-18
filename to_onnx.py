from __future__ import division

# Standard Library
import argparse
import sys
import os
os.environ['DEFAULT_TASKS']="cls"

# Import from third library
import torch.multiprocessing as mp

from up.utils.env.dist_helper import setup_distributed, finalize, env
from up.utils.general.yaml_loader import load_yaml  # IncludeLoader
from up.utils.env.launch import launch
from up.utils.general.user_analysis_helper import send_info

# Import from local
from up.utils.general.registry_factory import SUBCOMMAND_REGISTRY, RUNNER_REGISTRY
from up.utils.general.global_flag import DIST_BACKEND



def add_arguments(parser):

    parser.add_argument(
        '--fork-method',
        dest='fork_method',
        type=str,
        default='fork',
        choices=['spawn', 'fork'],
        help='method to fork subprocess, especially for dataloader')
    parser.add_argument('--backend',
                            dest='backend',
                            type=str,
                            default='linklink',
                            help='model backend')
    parser.add_argument('--config',
                            dest='config',
                            required=True,
                            help='settings of detection in yaml format')
    parser.add_argument('--save_prefix',
                            dest='save_prefix',
                            required=True,
                            help='prefix of saved files')
    parser.add_argument('--input_size',
                            dest='input_size',
                            type=lambda x: tuple(map(int, x.split('x'))),
                            help='input shape "CxHxW" to network, delimited by "x". e.g. 3x512x512')
    parser.add_argument('--ng', '--num_gpus_per_machine',
                            dest='num_gpus_per_machine',
                            type=int,
                            default=1,
                            help='num_gpus_per_machine')
    parser.add_argument('--nm', '--num_machines',
                            dest='num_machines',
                            type=int,
                            default=1,
                            help='num_machines')
    parser.add_argument('--launch',
                            dest='launch',
                            type=str,
                            default='pytorch',
                            help='pytorch backend')
    parser.add_argument('--port',
                            dest='port',
                            type=int,
                            default=13333,
                            help='dist port')
    parser.add_argument('--cfg_type',
                            dest='cfg_type',
                            type=str,
                            default='up',
                            help='config type (up or pod)')
    parser.add_argument('--opts',
                            help='options to replace yaml config',
                            default=None,
                            nargs=argparse.REMAINDER)

    parser.set_defaults(run=_main)


def main(args):
    cfg = load_yaml(args.config, args.cfg_type)
    cfg['args'] = {
        'ddp': args.backend == 'dist',
        'config_path': args.config,
        'opts': args.opts
    }
    cfg['runtime'] = cfg.setdefault('runtime', {})
    runner_cfg = cfg['runtime'].get('runner', {})
    runner_cfg['type'] = runner_cfg.get('type', 'base')
    runner_cfg['kwargs'] = runner_cfg.get('kwargs', {})
    cfg['runtime']['runner'] = runner_cfg
    training = False
    send_info(cfg, 'to_onnx')
    runner_cfg['kwargs']['training'] = training
    runner = RUNNER_REGISTRY.get(runner_cfg['type'])(cfg, **runner_cfg['kwargs'])
    runner.to_onnx(args.save_prefix, args.input_size)
    if env.world_size > 1:
        finalize()


def _main(args):
    DIST_BACKEND.backend = args.backend
    if args.launch == 'pytorch':
        launch(main, args.num_gpus_per_machine, args.num_machines, args=args, start_method=args.fork_method)
    else:
        mp.set_start_method(args.fork_method, force=True)
        fork_method = mp.get_start_method(allow_none=True)
        assert fork_method == args.fork_method
        sys.stdout.flush()
        setup_distributed(args.port, args.launch, args.backend)
        main(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Onnx Exporting")
    add_arguments(parser)
    args = parser.parse_args()
    args.run(args)
  

