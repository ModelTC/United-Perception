from __future__ import division

# Standard Library
import os
import argparse

# Import from local
from .subcommand import Subcommand
from eod.utils.general.yaml_loader import load_yaml  # IncludeLoader
from eod.utils.general.registry_factory import SUBCOMMAND_REGISTRY, INFERENCER_REGISTRY


__all__ = ['Inference']


@SUBCOMMAND_REGISTRY.register('inference')
class Inference(Subcommand):
    def add_subparser(self, name, parser):
        sub_parser = parser.add_parser(name,
                                       description='subcommand for inferencing',
                                       help='inference an image')
        sub_parser.add_argument('--config',
                                dest='config',
                                required=True,
                                help='settings of detection in yaml format')
        sub_parser.add_argument('--ckpt',
                                dest='ckpt',
                                required=True,
                                help='ckpt loaded for inferencing')
        sub_parser.add_argument('-i',
                                '--img_path',
                                dest='image_path',
                                required=True,
                                help='path of images needed to be inferenced')
        sub_parser.add_argument('-v',
                                '--vis_dir',
                                dest='vis_dir',
                                default='vis_dir',
                                help='directory saving visualization results')
        sub_parser.add_argument('--opts',
                                help='options to replace yaml config',
                                default=None,
                                nargs=argparse.REMAINDER)
        sub_parser.set_defaults(run=_main)
        return sub_parser


def main(args):
    cfg = load_yaml(args.config)
    cfg['args'] = {
        'ckpt': args.ckpt,
        'image_path': args.image_path,
        'vis_dir': args.vis_dir,
        'opts': args.opts
    }
    if not os.path.exists(args.vis_dir):
        os.makedirs(args.vis_dir, exist_ok=True)
    cfg['runtime'] = cfg.setdefault('runtime', {})
    infer_cfg = cfg['runtime'].get('inferencer', {})
    infer_cfg['type'] = infer_cfg.get('type', 'base')
    infer_cfg['kwargs'] = infer_cfg.get('kwargs', {})
    cfg['runtime']['inferencer'] = infer_cfg
    inferencer = INFERENCER_REGISTRY.get(infer_cfg['type'])(cfg, **infer_cfg['kwargs'])
    inferencer.predict()


def _main(args):
    main(args)
