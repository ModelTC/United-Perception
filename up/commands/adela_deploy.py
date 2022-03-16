from __future__ import division

# Standard Library
import os

# Import from pod
# from pod.utils.dist_helper import finalize, setup_distributed
from up.utils.general.yaml_loader import load_yaml  # IncludeLoader
from up.utils.general.user_analysis_helper import send_info

# Import from local
from .subcommand import Subcommand
from up.utils.general.registry_factory import SUBCOMMAND_REGISTRY, RUNNER_REGISTRY

__all__ = ['AdelaDeploy']


@SUBCOMMAND_REGISTRY.register('AdelaDeploy')
class AdelaDeploy(Subcommand):
    def add_subparser(self, name, parser):
        sub_parser = parser.add_parser(name,
                                       description='subcommand for package to deploy on adela',
                                       help='package pytorch model to running on adela')
        sub_parser.add_argument('--config',
                                dest='config',
                                required=True,
                                help='settings of detection in yaml format')
        sub_parser.add_argument("--release_json",
                                default="release.json",
                                help="release information")
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

    cfg = load_yaml(args.config, args.cfg_type)
    cfg['args'] = {
        'release_json': args.release_json,
    }
    cfg['runtime'] = cfg.setdefault('runtime', {})
    runner_cfg = cfg['runtime'].get('runner', {})
    runner_cfg['type'] = runner_cfg.get('type', 'base')
    runner_cfg['kwargs'] = runner_cfg.get('kwargs', {})
    runner_cfg['release_name'] = args.release_json
    cfg['runtime']['runner'] = runner_cfg
    send_info(cfg, func="adela")
    runner = RUNNER_REGISTRY.get(runner_cfg['type'])(cfg, **runner_cfg['kwargs'])
    runner.to_adela(release_name=runner_cfg['release_name'])
    # finalize()
