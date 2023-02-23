import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import rasterio as rio
from rasterio.vrt import WarpedVRT
from shapely import geometry

from ._types import Filename


def get_cache_dir(force_posix: bool = False) -> Path:
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


def get_transformed_bounds(filename: Filename, epsg_code: Optional[int] = None):
    """Get the bounds of a raster, possibly in a different CRS.

    Parameters
    ----------
    filename : str
        Path to the raster file.
    epsg_code : Optional[int]
        EPSG code of the CRS to transform to.
        If not provided, or the raster is already in the desired CRS,
        the bounds will not be transformed.

    Returns
    -------
    tuple
        The bounds of the raster as (left, bottom, right, top)
    """
    with rio.open(filename) as src:
        if epsg_code is None or src.crs.to_epsg() == epsg_code:
            return tuple(src.bounds)
        with WarpedVRT(src, crs=f"EPSG:{epsg_code}") as vrt:
            return tuple(vrt.bounds)


def get_intersection_bounds(
    fname1: Filename, fname2: Filename, epsg_code: int = 4326
) -> Tuple[float, float, float, float]:
    """Find the (left, bot, right, top) bounds of the raster intersection.

    Parameters
    ----------
    fname1 : str
        Path to the first raster file.
    fname2 : str
        Path to the second raster file.
    epsg_code : int
        EPSG code of the CRS of the desired output bounds.

    Returns
    -------
    tuple
        The bounds of the raster intersection in the new CRS.
        Bounds have format (left, bot, right, top)
    """
    return get_overlapping_bounds(
        get_transformed_bounds(fname1, epsg_code),
        get_transformed_bounds(fname2, epsg_code),
    )


def get_overlapping_bounds(
    bbox1: Tuple[float, float, float, float], bbox2: Tuple[float, float, float, float]
) -> Tuple[float, float, float, float]:
    """Find the (left, bot, right, top) bounds of the bbox intersection.

    Parameters
    ----------
    bbox1 : tuple
        The first bounding box in the format (left, bot, right, top)
    bbox2 : tuple
        The second bounding box in the format (left, bot, right, top)

    Returns
    -------
    tuple
        The bounds of the bbox intersection.
        Bounds have format (left, bot, right, top)
    """
    b1 = geometry.box(*bbox1)
    b2 = geometry.box(*bbox2)
    return b1.intersection(b2).bounds
