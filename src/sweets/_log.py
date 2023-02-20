"""Exports a get_log function which sets up easy logging.

Uses the standard python logging utilities, just provides
nice formatting out of the box across multiple files.

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

from rich.logging import RichHandler

__all__ = ["get_log", "log_runtime"]


def get_log(debug: bool = False, name: str = "dolphin._log") -> logging.Logger:
    """Create a nice log format for use across multiple files.

    Default logging level is INFO

    Parameters
    ----------
    debug : bool, optional
        If true, sets logging level to DEBUG (Default value = False)
    name : str, optional
        The name the logger will use when printing statements
        (Default value = "dolphin._log")

    Returns
    -------
    logging.Logger
    """
    format_ = "%(message)s"
    # format_ = "[%(asctime)s] [%(levelname)s %(filename)s] %(message)s"
    # formatter = Formatter(format_, datefmt="%m/%d %H:%M:%S")
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=log_level, format=format_, datefmt="[%X]", handlers=[RichHandler()]
    )

    return logging.getLogger(name)


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
        elapsed_time = t2 - t1
        time_string = "Total elapsed time for {} : {} minutes ({} seconds)".format(
            f.__name__,
            "{0:.2f}".format(elapsed_time / 60.0),
            "{0:.2f}".format(elapsed_time),
        )

        logger.info(time_string)
        return result

    return wrapper
