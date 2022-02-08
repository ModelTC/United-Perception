from __future__ import division

# Standard Library
import os
from prettytable import PrettyTable

# Import from local
from .subcommand import Subcommand
from up.utils.general.yaml_loader import load_yaml
from up.utils.general.cfg_helper import format_cfg
from up.utils.general.registry_factory import SUBCOMMAND_REGISTRY, MODEL_HELPER_REGISTRY
from up.utils.general.log_helper import default_logger as logger
from up.utils.general.computation_calculator import flops_cal, clever_format
from up.utils.general.user_analysis_helper import send_info


__all__ = ['Flops']


@SUBCOMMAND_REGISTRY.register('flops')
class Flops(Subcommand):
    def add_subparser(self, name, parser):
        sub_parser = parser.add_parser(name,
                                       description='subcommand for computation flops',
                                       help='Compute flops, macc and params number')
        sub_parser.add_argument('--config',
                                dest='config',
                                required=True,
                                help='settings of detection in yaml format')
        sub_parser.add_argument('-i',
                                '--input_size',
                                dest='input_size',
                                default='3,244,244',
                                help='format: C,H,W. e.g. 3,244,244')
        sub_parser.add_argument('--depth',
                                type=int,
                                default=0,
                                help='recursive depth')
        sub_parser.add_argument('--cfg_type',
                                dest='cfg_type',
                                type=str,
                                default='up',
                                help='config type (up or pod)')
        sub_parser.set_defaults(run=_main)
        return sub_parser


def main(args):
    # Load, merge and upgrade cfg
    assert (os.path.exists(args.config))
    cfg = load_yaml(args.config, args.cfg_type)

    send_info(cfg, func='flops')
    # Just for compatiable with older setting
    logger.info("Running with config:\n{}".format(format_cfg(cfg)))

    cfg['runtime'] = cfg.setdefault('runtime', {})
    model_helper_cfg = cfg['runtime'].get('model_helper', {})
    model_helper_cfg['type'] = model_helper_cfg.get('type', 'base')
    model_helper_cfg['kwargs'] = model_helper_cfg.get('kwargs', {})
    cfg['runtime']['model_helper'] = model_helper_cfg

    # build model
    model_helper_type = cfg['runtime']['model_helper']['type']
    model_helper_kwargs = cfg['runtime']['model_helper']['kwargs']
    model_helper_ins = MODEL_HELPER_REGISTRY[model_helper_type]
    model = model_helper_ins(cfg['net'], **model_helper_kwargs).cuda()
    logger.info('build model done')

    input_shape = [int(x) for x in args.input_size.split(',')]

    model.eval()
    computation_cost_dict = flops_cal(model, input_shape, args.depth)

    table = PrettyTable()
    table.title = 'Computation Cost & Parameters Table'
    table.field_names = ['module', 'macc', 'flops', 'parameters', 'memory', 'time']
    table.align = 'l'

    for name, cost in computation_cost_dict.items():
        macc, flops, params, memory = clever_format([cost['macc'], cost['ops'], cost['params'], cost['memory']], '%.3f')
        time = "%.6f s" % cost['time']
        table.add_row([name, macc, flops, params, memory, time])
    logger.info('\n{}'.format(table))


def _main(args):
    main(args)
