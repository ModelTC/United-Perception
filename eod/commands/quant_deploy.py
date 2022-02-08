from __future__ import division

# Standard Library
import argparse

# Import from local
from .subcommand import Subcommand
from eod.utils.general.yaml_loader import load_yaml  # IncludeLoader
from eod.utils.general.registry_factory import SUBCOMMAND_REGISTRY, DEPLOY_REGISTRY

__all__ = ['QuantDeploy']
@SUBCOMMAND_REGISTRY.register('quant_deploy')
class QuantDeploy(Subcommand):
    def add_subparser(self, name, parser):
        sub_parser = parser.add_parser(name,
                                       description='subcommand for deploying',
                                       help='deploy a model')

        sub_parser.add_argument('--config',
                                dest='config',
                                required=True,
                                help='settings of detection in yaml format')
        sub_parser.add_argument('--ckpt',
                                dest='ckpt',
                                required=True,
                                help='ckpt loaded for inferencing')
        sub_parser.add_argument('--input_shape',
                                dest='input_shape',
                                type=str,
                                required=True,
                                help='input shape NCHW')
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
        'input_shape': args.input_shape,
        'opts': args.opts
    }
    cfg['runtime'] = cfg.setdefault('runtime', {})
    runner_cfg = cfg['runtime'].get('runner', {})
    runner_cfg['type'] = runner_cfg.get('type', 'base')
    runner_cfg['kwargs'] = runner_cfg.get('kwargs', {})
    cfg['runtime']['runner'] = runner_cfg

    if runner_cfg['type'] == "quant":
        quant_deploy = DEPLOY_REGISTRY.get("quant")(cfg)
        quant_deploy.deploy()
    else:
        print("Need quant in cfg yaml.")


def _main(args):
    main(args)
