import json
from .subcommand import Subcommand
import functools
from eod.utils.general.registry_factory import EVALUATOR_REGISTRY, SUBCOMMAND_REGISTRY

__all__ = ['Eval']


@SUBCOMMAND_REGISTRY.register('eval')
class Eval(Subcommand):
    def add_subparser(self, name, parser):
        parser = parser.add_parser(name, help='sub-command for evaluation')
        subparsers = parser.add_subparsers(help='sub-command for evaluation')
        for name, evaluator in EVALUATOR_REGISTRY.items():
            subparser = evaluator.add_subparser(name, subparsers)
            subparser.set_defaults(run=functools.partial(_main, evaluator))

        return subparsers


def _main(evaluator_cls, args):
    print('building evaluator')
    evaluator = evaluator_cls.from_args(args)
    print('evaluator builded, start to evaluate')
    metrics = evaluator.eval(args.res_file)
    print(json.dumps(metrics, indent=2))
