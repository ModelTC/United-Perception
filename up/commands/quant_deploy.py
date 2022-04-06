from __future__ import division

# Standard Library
import argparse


# Import from local
from .subcommand import Subcommand
from up.utils.general.yaml_loader import load_yaml  # IncludeLoader
from up.utils.general.registry_factory import SUBCOMMAND_REGISTRY, DEPLOY_REGISTRY
from up.utils.general.user_analysis_helper import send_info
from up.utils.general.global_flag import QUANT_FLAG

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
        'opts': args.opts
    }
    cfg['runtime'] = cfg.setdefault('runtime', {})
    runner_cfg = cfg['runtime'].get('runner', {})
    runner_cfg['type'] = runner_cfg.get('type', 'base')
    runner_cfg['kwargs'] = runner_cfg.get('kwargs', {})
    cfg['runtime']['runner'] = runner_cfg
    QUANT_FLAG.flag = True

    send_info(cfg, func="quant_deploy")
    if runner_cfg['type'] == "quant":
        quant_deploy = DEPLOY_REGISTRY.get("quant")(cfg)
        quant_deploy.deploy()
    else:
        print("Need quant in cfg yaml.")


def _main(args):
    main(args)
