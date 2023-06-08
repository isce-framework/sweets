from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from math import nan
from pathlib import Path
from typing import Optional, Tuple

import rasterio as rio
from osgeo import gdal
from rasterio.vrt import WarpedVRT
from shapely import geometry, wkt

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


def to_bbox(*, geojson: Optional[str] = None, wkt_str: Optional[str] = None) -> Tuple:
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


@dataclass
class Stats:
    """Class holding the raster stats returned by `gdalinfo`."""

    min: float
    max: float
    mean: float
    stddev: float
    pct_valid: float


def get_raster_stats(filename: Filename, band: int = 1) -> Stats:
    """Get the (Min, Max, Mean, StdDev, pct_valid) of the 1-band file."""
    try:
        ii = gdal.Info(os.fspath(filename), stats=True, format="json")
    except RuntimeError as e:
        if "No such file or directory" in e.args[0]:
            raise
        elif "no valid pixels found" in e.args[0]:
            return Stats(nan, nan, nan, nan, 0.0)

    s = ii["bands"][band - 1]["metadata"][""]
    return Stats(
        float(s["STATISTICS_MAXIMUM"]),
        float(s["STATISTICS_MINIMUM"]),
        float(s["STATISTICS_MEAN"]),
        float(s["STATISTICS_STDDEV"]),
        float(s["STATISTICS_VALID_PERCENT"]),
    )


def is_valid(filename: Filename) -> tuple[bool, str]:
    """Check if GDAL can open the file and if there are any valid pixels.

    Parameters
    ----------
    filename : Filename
        Path to file.

    Returns
    -------
    bool:
        False if bad file, or no valid pixels.
    str:
        Reason behind a `False` value given by GDAL.
    """
    try:
        get_raster_stats(filename)
    except RuntimeError as e:
        return False, str(e)
    return True, ""


def get_bad_files(
    path: Filename,
    ext: str = ".int",
    valid_threshold: float = 0.6,
    max_jobs: int = 20,
) -> tuple[list[Path], list[Stats]]:
    """Check all files in `path` for (partially) invalid rasters."""
    files = sorted(Path(path).glob(f"*.{ext}"))
    with ThreadPoolExecutor(max_jobs) as exc:
        stats = list(exc.map(get_raster_stats, files))

    bad_files = [
        (f, s) for (f, s) in zip(files, stats) if s.pct_valid < valid_threshold
    ]
    if not bad_files:
        return [], []
    out_files, out_stats = list(zip(*bad_files))
    return out_files, out_stats  # type: ignore


def remove_invalid_ifgs(bad_files: list[Filename]):
    """Remove invalid ifgs and their unwrapped counterparts."""
    for ff in bad_files:
        u = str(ff).replace("stitched", "unwrapped").replace(".int", ".unw")
        try:
            Path(u).unlink()
        except FileNotFoundError:
            continue
        Path(ff).unlink()
