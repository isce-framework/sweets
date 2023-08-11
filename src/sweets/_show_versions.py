"""Utility methods to print system info for debugging.

Adapted from `rasterio.show_versions`,
which was adapted from `sklearn.utils._show_versions`
which was adapted from `pandas.show_versions`
"""
from __future__ import annotations

import importlib
import platform
import sys
from typing import Optional

import sweets

__all__ = ["show_versions"]


def _get_sys_info() -> dict[str, str]:
    """System information.

    Returns
    -------
    dict
        system and Python version information
    """
    return {
        "python": sys.version.replace("\n", " "),
        "executable": sys.executable,
        "machine": platform.platform(),
    }


def _get_opera_info() -> dict[str, Optional[str]]:
    """Information on system on core modules.

    Returns
    -------
    dict
        opera module information
    """
    blob = {
        "sweets": sweets.__version__,
        "dolphin": _get_version("dolphin"),
        "isce3": _get_version("isce3"),
        "compass": _get_version("compass"),
        "s1reader": _get_version("s1reader"),
    }
    return blob


def _get_version(module_name: str) -> Optional[str]:
    if module_name in sys.modules:
        mod = sys.modules[module_name]
    else:
        try:
            mod = importlib.import_module(module_name)
        except ImportError:
            return None
    try:
        return mod.__version__
    except AttributeError:
        return mod.version


def _get_deps_info() -> dict[str, Optional[str]]:
    """Overview of the installed version of main dependencies.

    Returns
    -------
    dict:
        version information on relevant Python libraries
    """
    deps = [
        "numpy",
        "numba",
        "osgeo.gdal",
        "h5py",
        "pydantic",
        "rioxarray",
        "setuptools",
        "shapely",
        "xarray",
    ]
    return {name: _get_version(name) for name in deps}


def _print_info_dict(info_dict: dict) -> None:
    """Print the information dictionary."""
    for key, stat in info_dict.items():
        print(f"{key:>12}: {stat}")


def show_versions() -> None:
    """Print useful debugging information.

    Examples
    --------
    > python -c "import sweets; sweets.show_versions()"
    """
    print("sweets core packages:")
    _print_info_dict(_get_opera_info())
    print("\nSystem:")
    _print_info_dict(_get_sys_info())
    print("\nPython deps:")
    _print_info_dict(_get_deps_info())
