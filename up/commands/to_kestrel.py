from __future__ import division

# Standard Library
import os
import argparse
import sys

import torch.multiprocessing as mp
from up.utils.env.dist_helper import setup_distributed, finalize
from up.utils.general.yaml_loader import load_yaml
from up.utils.env.launch import launch
from .subcommand import Subcommand
from up.utils.general.registry_factory import SUBCOMMAND_REGISTRY, RUNNER_REGISTRY
from up.utils.general.log_helper import default_logger as logger
from up.utils.general.user_analysis_helper import send_info
from up.utils.general.global_flag import DIST_BACKEND


__all__ = ['ToKestrel']


@SUBCOMMAND_REGISTRY.register('to_kestrel')
class ToKestrel(Subcommand):
    def add_subparser(self, name, parser):
        sub_parser = parser.add_parser(name,
                                       description='subcommand for to caffe',
                                       help='convert a model to caffe model')
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
                                default='linklink',
                                help='model backend')
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
        sub_parser.add_argument('--config',
                                dest='config',
                                required=True,
                                help='settings of detection in yaml format')
        sub_parser.add_argument('--save_to',
                                dest='save_to',
                                default="tokestrel",
                                type=str,
                                help='path to save kestrel model')
        sub_parser.add_argument('--serialize',
                                dest='serialize',
                                action='store_true',
                                help='wether to do serialization, if your model runs on tensor-rt')
        sub_parser.add_argument('--cfg_type',
                                dest='cfg_type',
                                type=str,
                                default='up',
                                help='config type (up or pod)')
        sub_parser.add_argument('--opts',
                                help='options to replace yaml config',
                                default=None,
                                nargs=argparse.REMAINDER)
        sub_parser.set_defaults(run=_main)
        return sub_parser


def main(args):
    assert (os.path.exists(args.config)), args.config
    cfg = load_yaml(args.config, args.cfg_type)
    cfg['args'] = {
        'opts': args.opts
    }
    cfg['runtime'] = cfg.setdefault('runtime', {})
    runner_cfg = cfg['runtime'].get('runner', {})
    runner_cfg['type'] = runner_cfg.get('type', 'base')
    runner_cfg['kwargs'] = runner_cfg.get('kwargs', {})
    runner_cfg['kwargs']["training"] = False
    cfg['dataset']['data_pool'] = []
    cfg['runtime']['runner'] = runner_cfg
    send_info(cfg, func="to_kestrel")
    runner = RUNNER_REGISTRY.get(runner_cfg['type'])(cfg, **runner_cfg['kwargs'])
    runner.to_kestrel(args.save_to, args.serialize)
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
        try:
            setup_distributed(args.port, args.launch, args.backend)
        except KeyError:
            logger.info("Setup distributed env failed.")
        main(args)
