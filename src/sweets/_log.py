"""Exports a get_log function which sets up easy logging.

Uses the standard python logging utilities + Rich formatting.

Usage:

    from ._log import get_log
    logger = get_log(__name__)

    logger.info("Something happened")
    logger.warning("Something concerning happened")
    logger.error("Something bad happened")
    logger.critical("Something just awful happened")
    logger.debug("Extra printing we often don't need to see.")
    # Custom output for this module:
    logger.success("Something great happened: highlight this success")
"""

import logging

from dolphin._log import log_runtime
from rich.console import Console
from rich.logging import RichHandler

__all__ = ["get_log", "log_runtime", "console"]

console = Console()


def get_log(
    name: str = "sweets._log",
    debug: bool = False,
) -> logging.Logger:
    """Create a nice log format for use across multiple files.

    Default logging level is INFO

    Parameters
    ----------
    debug : bool, optional
        If true, sets logging level to DEBUG (Default value = False)
    name : str, optional
        The name the logger will use when printing statements
        (Default value = "sweets._log")
    filename : Filename, optional
        If provided, will log to this file in addition to stderr.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    return format_log(logger, debug=debug)


def format_log(logger: logging.Logger, debug: bool = False) -> logging.Logger:
    """Make the logging output pretty and colored with times.

    Parameters
    ----------
    logger : logging.Logger
        The logger to format
    debug : bool (Default value = False)
        If true, sets logging level to DEBUG
    filename : Filename, optional
        If provided, will log to this file in addition to stderr.

    Returns
    -------
    logging.Logger
    """
    log_level = logging.DEBUG if debug else logging.INFO

    if not logger.handlers:
        logger.addHandler(RichHandler(rich_tracebacks=True, level=log_level))
        logger.setLevel(log_level)

    if debug:
        logger.setLevel(debug)

    return logger
