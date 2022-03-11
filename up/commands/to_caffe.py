from __future__ import division

# Import from local
import argparse
from up.utils.general.yaml_loader import load_yaml
from .subcommand import Subcommand
from up.utils.general.registry_factory import SUBCOMMAND_REGISTRY, RUNNER_REGISTRY
from up.utils.general.user_analysis_helper import send_info


__all__ = ['ToCaffe']


@SUBCOMMAND_REGISTRY.register('to_caffe')
class ToCaffe(Subcommand):
    def add_subparser(self, name, parser):
        sub_parser = parser.add_parser(name,
                                       description='subcommand for to caffe',
                                       help='convert a model to caffe model')
        sub_parser.add_argument('--config',
                                dest='config',
                                required=True,
                                help='settings of detection in yaml format')
        sub_parser.add_argument('--save_prefix',
                                dest='save_prefix',
                                required=True,
                                help='prefix of saved files')
        sub_parser.add_argument('--input_size',
                                dest='input_size',
                                required=True,
                                type=lambda x: tuple(map(int, x.split('x'))),
                                help='input shape "CxHxW" to network, delimited by "x". e.g. 3x512x512')
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
        'config_path': args.config,
        'opts': args.opts
    }
    cfg['runtime'] = cfg.setdefault('runtime', {})
    runner_cfg = cfg['runtime'].get('runner', {})
    runner_cfg['type'] = runner_cfg.get('type', 'base')
    runner_cfg['kwargs'] = runner_cfg.get('kwargs', {})
    cfg['runtime']['runner'] = runner_cfg
    send_info(cfg, func="to_caffe")
    runner = RUNNER_REGISTRY.get(runner_cfg['type'])(cfg, **runner_cfg['kwargs'])
    runner.to_caffe(args.save_prefix, args.input_size)


def _main(args):
    main(args)
