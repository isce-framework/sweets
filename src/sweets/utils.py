import os
import sys
from pathlib import Path


def get_cache_dir(force_posix=False):
    """Return the config folder for the application.

    Source:
    https://github.com/pallets/click/blob/a63679e77f9be2eb99e2f0884d617f9635a485e2/src/click/utils.py#L408

    The following folders could be returned:
    Mac OS X:
      ``~/Library/Application Support/sweets``
    Mac OS X (POSIX):
      ``~/.sweets``
    Unix:
      ``~/.cache/sweets``
    Unix (POSIX):
      ``~/.sweets``

    Parameters
    ----------
    force_posix : bool
        If this is set to `True` then on any POSIX system the
        folder will be stored in the home folder with a leading
        dot instead of the XDG config home or darwin's
        application support folder.

    """
    app_name = "sweets"
    if force_posix:
        path = Path("~/.sweets") / app_name
    elif sys.platform == "darwin":
        path = Path("~/Library/Application Support") / app_name
    else:
        path = Path(os.environ.get("XDG_CONFIG_HOME", "~/.cache")) / app_name
    path = path.expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path
