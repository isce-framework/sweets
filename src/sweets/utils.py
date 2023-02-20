import os
import sys


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
        path = os.path.join(os.path.expanduser("~/." + app_name))
    if sys.platform == "darwin":
        path = os.path.join(
            os.path.expanduser("~/Library/Application Support"), app_name
        )
    path = os.path.join(
        os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.cache")),
        app_name,
    )
    if not os.path.exists(path):
        os.makedirs(path)
    return path
