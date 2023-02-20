"""Exports a get_log function which sets up easy logging.

Uses the standard python logging utilities + Rich formatting.

Usage:

    from ._log import get_log
    logger = get_log()

    logger.info("Something happened")
    logger.warning("Something concerning happened")
    logger.error("Something bad happened")
    logger.critical("Something just awful happened")
    logger.debug("Extra printing we often don't need to see.")
    # Custom output for this module:
    logger.success("Something great happened: highlight this success")
"""
import logging
import time
from collections.abc import Callable
from functools import wraps
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler

from ._types import Filename

__all__ = ["get_log", "log_runtime"]


def get_log(
    debug: bool = False,
    filename: Optional[Filename] = None,
) -> logging.Logger:
    """Create a nice log format for use across multiple files.

    Default logging level is INFO

    Parameters
    ----------
    debug : bool, optional
        If true, sets logging level to DEBUG (Default value = False)
    filename : Filename, optional
        If provided, will log to this file in addition to stderr.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger("sweets")
    return format_log(logger, debug=debug, filename=filename)


def format_log(
    logger: logging.Logger, debug: bool = False, filename: Optional[Filename] = None
) -> logging.Logger:
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

    if filename is not None:
        console = Console(file=open(filename, "w"))
        logger.addHandler(RichHandler(console=console, level=log_level))

    if debug:
        logger.setLevel(debug)

    return logger


def log_runtime(f: Callable) -> Callable:
    """Decorate a function to time how long it takes to run.

    Usage
    -----
    @log_runtime
    def test_func():
        return 2 + 4
    """
    logger = get_log()

    @wraps(f)
    def wrapper(*args, **kwargs):
        t1 = time.time()

        result = f(*args, **kwargs)

        t2 = time.time()
        elapsed_seconds = t2 - t1
        elapsed_minutes = elapsed_seconds / 60.0
        time_string = (
            f"Total elapsed time for {f.__module__}.{f.__name__} : "
            f"{elapsed_minutes:.2f} minutes ({elapsed_seconds:.2f} seconds)"
        )

        logger.info(time_string)
        return result

    return wrapper
