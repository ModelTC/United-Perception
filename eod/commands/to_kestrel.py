from __future__ import division

# Standard Library
import argparse
import os
import sys

# Import from third library
import torch.multiprocessing as mp

from eod.utils.env.dist_helper import setup_distributed, finalize
from eod.utils.general.yaml_loader import load_yaml  # IncludeLoader

# Import from local
from .subcommand import Subcommand
from eod.utils.general.registry_factory import SUBCOMMAND_REGISTRY
from eod.utils.general.global_flag import DIST_BACKEND
from eod.utils.general.log_helper import default_logger as logger
from eod.utils.general.tokestrel_helper import to_kestrel


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
                                default='dist',
                                help='model backend')
        sub_parser.add_argument('--config',
                                dest='config',
                                required=True,
                                help='settings of detection in yaml format')
        sub_parser.add_argument('--save_to',
                                dest='save_to',
                                default=None,
                                type=str,
                                help='path to save kestrel model')
        sub_parser.add_argument('--serialize',
                                dest='serialize',
                                action='store_true',
                                help='wether to do serialization, if your model runs on tensor-rt')
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
        sub_parser.add_argument('--opts',
                                help='options to replace yaml config',
                                default=None,
                                nargs=argparse.REMAINDER)

        sub_parser.set_defaults(run=_main)
        return sub_parser


def main(args):
    assert (os.path.exists(args.config)), args.config
    cfg = load_yaml(args.config)
    cfg['args'] = {
        'ddp': args.backend == 'dist',
        'config_path': args.config,
        'opts': args.opts
    }
    to_kestrel(cfg, args.save_to, args.serialize)

    if len(cfg['dataset'].get('data_pool', ['train:train', 'test:test'])):
        finalize()


def _main(args):
    DIST_BACKEND.backend = args.backend
    mp.set_start_method(args.fork_method, force=True)
    fork_method = mp.get_start_method(allow_none=True)
    assert fork_method == args.fork_method
    sys.stdout.flush()
    setup_distributed(args.port, args.launch, args.backend)
    logger.init_log()
    main(args)
