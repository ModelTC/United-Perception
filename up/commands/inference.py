from __future__ import division

# Standard Library
import os
import argparse
import torch.multiprocessing as mp
import sys

# Import from local
from .subcommand import Subcommand
from up.utils.general.yaml_loader import load_yaml  # IncludeLoader
from up.utils.general.registry_factory import SUBCOMMAND_REGISTRY, INFERENCER_REGISTRY
from up.utils.general.user_analysis_helper import send_info
from up.utils.general.global_flag import DIST_BACKEND
from up.utils.env.launch import launch
from up.utils.env.dist_helper import setup_distributed, env, finalize
from up.utils.general.log_helper import default_logger as logger

__all__ = ['Inference']


@SUBCOMMAND_REGISTRY.register('inference')
class Inference(Subcommand):
    def add_subparser(self, name, parser):
        sub_parser = parser.add_parser(name,
                                       description='subcommand for inferencing',
                                       help='inference an image')
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
        sub_parser.add_argument('-i',
                                '--img_path',
                                dest='image_path',
                                required=True,
                                help='path of images needed to be inferenced')
        sub_parser.add_argument('-c',
                                '--ckpt',
                                dest='ckpt',
                                required=True,
                                help='path of model')
        sub_parser.add_argument('-v',
                                '--vis_dir',
                                dest='vis_dir',
                                default='vis_dir',
                                help='directory saving visualization results')
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
    cfg = load_yaml(args.config, args.cfg_type)
    cfg['args'] = {
        'image_path': args.image_path,
        'ckpt': args.ckpt,
        'vis_dir': args.vis_dir,
        'opts': args.opts
    }
    if not os.path.exists(args.vis_dir):
        os.makedirs(args.vis_dir, exist_ok=True)
    send_info(cfg, func="inference")
    cfg['runtime'] = cfg.setdefault('runtime', {})
    infer_cfg = cfg['runtime'].get('inferencer', {})
    infer_cfg['type'] = infer_cfg.get('type', 'base')
    infer_cfg['kwargs'] = infer_cfg.get('kwargs', {})
    # default visualizer
    vis_cfg = infer_cfg['kwargs'].setdefault('visualizer', {})
    vis_cfg['type'] = vis_cfg.get('type', 'plt')
    vis_cfg['kwargs'] = vis_cfg.get('kwargs', {})
    cfg['runtime']['inferencer'] = infer_cfg
    inferencer = INFERENCER_REGISTRY.get(infer_cfg['type'])(cfg, **infer_cfg['kwargs'])
    inferencer.predict()
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
        try:
            setup_distributed(args.port, args.launch, args.backend)
        except KeyError:
            logger.info("Setup distributed env failed.")
        main(args)
