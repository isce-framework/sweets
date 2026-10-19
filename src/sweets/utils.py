from __future__ import annotations

import json
import os
from pathlib import Path

import rasterio as rio
from rasterio.vrt import WarpedVRT
from shapely import geometry, wkt

from ._types import Filename


def get_cache_dir() -> Path:
    """Return the per-user cache directory used for downloaded artifacts.

    Resolved to ``$XDG_CACHE_HOME/sweets`` (or ``~/.cache/sweets`` if unset)
    on every platform — explicitly *not* macOS's ``~/Library/Application
    Support/sweets`` because the space in ``Application Support`` trips
    upstream sardem's ``unzip_cmd.split(" ")`` water-mask download path.
    """
    app_name = "sweets"
    base = os.environ.get("XDG_CACHE_HOME", "~/.cache")
    path = Path(base).expanduser() / app_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def to_wkt(geojson: str) -> str:
    """Convert a geojson string to a WKT string.

    Parameters
    ----------
    geojson : str
        A geojson string.

    Returns
    -------
    str
        A WKT string.
    """
    return wkt.dumps(geometry.shape(json.loads(geojson)))


def to_bbox(*, geojson: str | None = None, wkt_str: str | None = None) -> tuple:
    """Convert a geojson or WKT string to a bounding box.

    Parameters
    ----------
    geojson : Optional[str]
        A geojson string.
    wkt_str : Optional[str]
        A WKT string.

    Returns
    -------
    Tuple
        A tuple of (left, bottom, right, top) bounds.

    Raises
    ------
    ValueError
        If neither geojson nor wkt are provided.
    """
    if geojson is not None:
        geom = geometry.shape(json.loads(geojson))
    elif wkt_str is not None:
        geom = wkt.loads(wkt_str)
    else:
        raise ValueError("Must provide either geojson or wkt_str")
    return tuple(geom.bounds)


def get_transformed_bounds(filename: Filename, epsg_code: int | None = None):
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
) -> tuple[float, float, float, float]:
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
    bbox1: tuple[float, float, float, float], bbox2: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
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
