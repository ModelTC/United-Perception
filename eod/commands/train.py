from __future__ import division

# Standard Library
import argparse
import sys

# Import from third library
import torch.multiprocessing as mp

from eod.utils.env.dist_helper import setup_distributed, finalize, env
from eod.utils.general.yaml_loader import load_yaml  # IncludeLoader
from eod.utils.env.launch import launch

# Import from local
from .subcommand import Subcommand
from eod.utils.general.registry_factory import SUBCOMMAND_REGISTRY, RUNNER_REGISTRY
from eod.utils.general.global_flag import DIST_BACKEND
from eod.utils.general.log_helper import default_logger as logger


__all__ = ['Train']


@SUBCOMMAND_REGISTRY.register('train')
class Train(Subcommand):
    def add_subparser(self, name, parser):
        sub_parser = parser.add_parser(name,
                                       description='subcommand for training',
                                       help='train a model')

        sub_parser.add_argument('-e',
                                '--evaluate',
                                dest='evaluate',
                                action='store_true',
                                help='evaluate model on validation set')
        sub_parser.add_argument(
            '--fork-method',
            dest='fork_method',
            type=str,
            default='fork',
            choices=['spawn', 'fork'],
            help='method to fork subprocess, especially for dataloader')
        sub_parser.add_argument('--backend',
                                dest='backend',
                                type=str,
                                default='dist',
                                help='model backend')
        sub_parser.add_argument(
            '--nocudnn',
            dest='nocudnn',
            action='store_true',
            help='Whether to use cudnn backend or not. Please disable cudnn when running on V100'
        )
        sub_parser.add_argument(
            '--allow_dead_parameter',
            action='store_true',
            help='dead parameter (defined in model but not used in forward pass) is allowed'
        )
        sub_parser.add_argument('--config',
                                dest='config',
                                required=True,
                                help='settings of detection in yaml format')
        sub_parser.add_argument('--display',
                                dest='display',
                                type=int,
                                default=20,
                                help='display intervel')
        sub_parser.add_argument('--async',
                                dest='asynchronize',
                                action='store_true',
                                help='whether to use asynchronize mode(linklink)')
        sub_parser.add_argument('--ng', '--num_gpus_per_machine',
                                dest='num_gpus_per_machine',
                                type=int,
                                default=8,
                                help='num_gpus_per_machine')
        sub_parser.add_argument('--nm', '--num_machines',
                                dest='num_machines',
                                type=int,
                                default=1,
                                help='num_machines')
        sub_parser.add_argument('--launch',
                                dest='launch',
                                type=str,
                                default='slurm',
                                help='launch backend')
        sub_parser.add_argument('--port',
                                dest='port',
                                type=int,
                                default=13333,
                                help='dist port')
        sub_parser.add_argument('--no_running_config',
                                action='store_true',
                                help='disable display running config')
        sub_parser.add_argument('--phase', default='train', help="train phase")
        sub_parser.add_argument('--opts',
                                help='options to replace yaml config',
                                default=None,
                                nargs=argparse.REMAINDER)

        sub_parser.set_defaults(run=_main)
        return sub_parser


def main(args):
    cfg = load_yaml(args.config)
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
    runner_cfg['kwargs']['training'] = training
    runner = RUNNER_REGISTRY.get(runner_cfg['type'])(cfg, **runner_cfg['kwargs'])
    train_func = {"train": runner.train, "eval": runner.evaluate}
    train_func[train_phase]()
    if env.world_size > 1:
        finalize()


def _main(args):
    DIST_BACKEND.backend = args.backend
    if args.launch == 'pytorch':
        logger.init_log()
        launch(main, args.num_gpus_per_machine, args.num_machines, args=args, start_method=args.fork_method)
    else:
        mp.set_start_method(args.fork_method, force=True)
        fork_method = mp.get_start_method(allow_none=True)
        assert fork_method == args.fork_method
        sys.stdout.flush()
        setup_distributed(args.port, args.launch, args.backend)
        logger.init_log()
        main(args)
