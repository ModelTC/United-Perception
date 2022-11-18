from __future__ import division

# Standard Library
import argparse
import sys
import os
os.environ['DEFAULT_TASKS']="cls"
# Import from third library
import torch.multiprocessing as mp

from up.utils.env.dist_helper import setup_distributed, finalize, gpu_check
from up.utils.general.yaml_loader import load_yaml  # IncludeLoader
from up.utils.env.launch import launch
from up.utils.general.user_analysis_helper import send_info

# Import from local
from up.commands.subcommand import Subcommand
from up.utils.general.registry_factory import SUBCOMMAND_REGISTRY, RUNNER_REGISTRY
from up.utils.general.global_flag import DIST_BACKEND


__all__ = ['Train']


def add_arguments(parser):
    parser.add_argument('-e',
                            '--evaluate',
                            dest='evaluate',
                            action='store_true',
                            help='evaluate model on validation set')
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
                            default='dist',
                            help='model backend')
    parser.add_argument(
        '--nocudnn',
        dest='nocudnn',
        action='store_true',
        help='Whether to use cudnn backend or not. Please disable cudnn when running on V100'
    )
    parser.add_argument(
        '--allow_dead_parameter',
        action='store_true',
        help='dead parameter (defined in model but not used in forward pass) is allowed'
    )
    parser.add_argument('--config',
                            dest='config',
                            required=True,
                            help='settings of detection in yaml format')
    parser.add_argument('--display',
                            dest='display',
                            type=int,
                            default=20,
                            help='display intervel')
    parser.add_argument('--async',
                            dest='asynchronize',
                            action='store_true',
                            help='whether to use asynchronize mode(linklink)')
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
                            help='launch backend')
    parser.add_argument('--port',
                            dest='port',
                            type=int,
                            default=13333,
                            help='dist port')
    parser.add_argument('--no_running_config',
                            action='store_true',
                            help='disable display running config')
    parser.add_argument('--phase', default='train', help="train phase")
    parser.add_argument('--cfg_type',
                            dest='cfg_type',
                            type=str,
                            default='up',
                            help='config type (up or pod)')
    parser.add_argument('--opts',
                            help='options to replace yaml config',
                            default=None,
                            nargs=argparse.REMAINDER)
    parser.add_argument('--test_gpu',
                            dest='test_gpu',
                            action='store_true',
                            help='test if gpus work properly before training')

    parser.set_defaults(run=_main)
    # return parser


def main(args):
    if args.test_gpu:
        gpu_check()
    cfg = load_yaml(args.config, args.cfg_type)
    cfg['args'] = {
        'ddp': args.backend == 'dist',
        'config_path': args.config,
        'asynchronize': args.asynchronize,
        'nocudnn': args.nocudnn,
        'display': args.display,
        'no_running_config': args.no_running_config,
        'allow_dead_parameter': args.allow_dead_parameter,
        'opts': args.opts
    }
    print(cfg)
    train_phase = args.phase
    cfg['runtime'] = cfg.setdefault('runtime', {})
    runner_cfg = cfg['runtime'].get('runner', {})
    runner_cfg['type'] = runner_cfg.get('type', 'base')
    runner_cfg['kwargs'] = runner_cfg.get('kwargs', {})
    cfg['runtime']['runner'] = runner_cfg
    training = True
    if args.evaluate:
        training = False
        train_phase = "eval"
    send_info(cfg, train_phase)
    runner_cfg['kwargs']['training'] = training
    runner = RUNNER_REGISTRY.get(runner_cfg['type'])(cfg, **runner_cfg['kwargs'])
    train_func = {"train": runner.train, "eval": runner.evaluate}
    if runner_cfg['type'] == 'bignas':
        train_func = {
            "train_supnet": runner.train,
            "sample_flops": runner.sample_multiple_subnet_flops,
            "sample_accuracy":
                runner.sample_multiple_subnet_accuracy,
            "evaluate_subnet": runner.evaluate_subnet,
            "finetune_subnet": runner.finetune_subnet,
            "sample_subnet": runner.sample_subnet_weight
        }
        assert train_phase in train_func, f"{train_phase} is not supported"

    train_func[train_phase]()
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
    parser = argparse.ArgumentParser(description="Run Training")
    add_arguments(parser)
    args = parser.parse_args()
    args.run(args)
  

