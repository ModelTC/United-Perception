# Standard Library
import argparse

from eod import __version__
from eod.utils.general.registry_factory import SUBCOMMAND_REGISTRY


def main():

    parser = argparse.ArgumentParser(description="Run Easy Object Detector")
    parser.add_argument('--version', action='version', version=__version__)

    subparsers = parser.add_subparsers(title='subcommands')

    for subname, subcommand in SUBCOMMAND_REGISTRY.items():
        subcommand().add_subparser(subname, subparsers)

    args = parser.parse_args()

    if 'run' in dir(args):
        args.run(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
