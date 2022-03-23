from __future__ import division

# Standard Library
import os

# Import from up
from up.utils.general.yaml_loader import load_yaml
from up.utils.general.user_analysis_helper import send_info

# Import from local
from .subcommand import Subcommand
from up.utils.general.registry_factory import SUBCOMMAND_REGISTRY, RUNNER_REGISTRY

__all__ = ['AdelaDeploy']


@SUBCOMMAND_REGISTRY.register('adela_deploy')
class AdelaDeploy(Subcommand):
    def add_subparser(self, name, parser):
        sub_parser = parser.add_parser(name,
                                       description='subcommand for package to deploy on adela',
                                       help='package pytorch model to running on adela')
        sub_parser.add_argument('--config',
                                dest='config',
                                required=True,
                                help='settings of detection in yaml format')
        sub_parser.add_argument("--ks_model",
                                dest='ks_model',
                                type=str,
                                required=True,
                                default="kestrel_model/kestrel_model_1.0.0.tar",
                                help="kestrel model path")
        sub_parser.add_argument('--cfg_type',
                                dest='cfg_type',
                                type=str,
                                default='up',
                                help='config type (up or pod)')
        sub_parser.set_defaults(run=_main)

        return sub_parser


def _main(args):
    assert (os.path.exists(args.config)), args.config
    cfg = load_yaml(args.config, args.cfg_type)
    cfg['args'] = {
        'ks_model': args.ks_model,
    }
    cfg['runtime'] = cfg.setdefault('runtime', {})
    runner_cfg = cfg['runtime'].get('runner', {})
    runner_cfg['type'] = runner_cfg.get('type', 'base')
    runner_cfg['kwargs'] = runner_cfg.get('kwargs', {})
    runner_cfg['ks_model'] = args.ks_model
    cfg['runtime']['runner'] = runner_cfg
    send_info(cfg, func="adela")
    runner = RUNNER_REGISTRY.get(runner_cfg['type'])(cfg, **runner_cfg['kwargs'])
    runner.to_adela(save_to=runner_cfg['ks_model'])
