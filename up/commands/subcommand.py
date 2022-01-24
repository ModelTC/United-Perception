"""
Base class for subcommands
"""
# Standard Library
import abc


class Subcommand(abc.ABC):
    """
    An abstract class representing subcommands .
    If you wanted to (for example) create your own custom `special-evaluate` command to use like

    you would create a ``Subcommand`` subclass and then pass it as an override to
    :func:`~commands.main` .
    """
    @abc.abstractmethod
    def add_subparser(self, name, parser):
        # pylint: disable=protected-access
        raise NotImplementedError
