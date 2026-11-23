"""High-resolution water mask creation from ASF water mask tiles.

ASF's water mask tiles mosaic OpenStreetMap and ESA WorldCover data at high
resolution. The public bucket is at
``https://asf-dem-west.s3.amazonaws.com/WATER_MASK/TILES/`` and uses a
5-degree grid with the naming convention
``{lat_dir}{lat:02d}{lon_dir}{lon:03d}.tif`` (e.g. ``n55w165.tif``).

Source convention: ``0 = land``, ``1 = water``. The
:class:`WaterValue` enum lets the caller pick the output convention; sweets
itself wants ``0 = water (invalid)`` / ``1 = land (valid)`` to match dolphin's
mask convention.

Optional shoreline buffering expands the water class into adjacent land
pixels via morphological dilation, which is helpful for InSAR users who
want to mask out coastal noise.
"""

from __future__ import annotations

from enum import Enum
from os import fspath
from pathlib import Path

import numpy as np
from loguru import logger
from osgeo import gdal
from scipy import ndimage

gdal.UseExceptions()

TILE_URL_BASE = "https://asf-dem-west.s3.amazonaws.com/WATER_MASK/TILES/"


class WaterValue(Enum):
    """Convention for water mask pixel values."""

    ZERO = 0  # water=0 (invalid), land=1 (valid) — dolphin convention
    ONE = 1  # water=1, land=0 — ASF tile native convention


def _coord_to_tile_name(lon: float, lat: float) -> str:
    """Convert a coordinate to ASF tile filename.

    Tiles are on a 5-degree grid. E.g., (lon=-163.5, lat=59.0) -> n55w165.tif
    """
    lat_floor = int(np.floor(lat / 5) * 5)
    lon_floor = int(np.floor(lon / 5) * 5)

    lat_dir = "n" if lat_floor >= 0 else "s"
    lon_dir = "e" if lon_floor >= 0 else "w"

    return f"{lat_dir}{abs(lat_floor):02d}{lon_dir}{abs(lon_floor):03d}.tif"


def _get_tile_urls(west: float, south: float, east: float, north: float) -> list[str]:
    """Return /vsicurl URLs for all tiles covering the given bounds."""
    tiles: list[str] = []
    corners = [
        (west, north),
        (west, south),
        (east, north),
        (east, south),
    ]
    # Wide extents (high latitudes) might span more than two 5-deg cells in
    # longitude — sample the midpoint too.
    width = east - west
    if width > 5:
        mid_lon = west + width / 2
        corners.extend([(mid_lon, north), (mid_lon, south)])

    for lon, lat in corners:
        url = f"/vsicurl/{TILE_URL_BASE}{_coord_to_tile_name(lon, lat)}"
        if url not in tiles:
            tiles.append(url)
    return tiles


def _buffer_mask(
    mask: np.ndarray,
    buffer_pixels: int,
    water_value: WaterValue,
) -> np.ndarray:
    """Expand water regions by `buffer_pixels` using morphological dilation."""
    if buffer_pixels <= 0:
        return mask

    diameter = 2 * buffer_pixels + 1
    y, x = np.ogrid[:diameter, :diameter]
    center = buffer_pixels
    struct = ((x - center) ** 2 + (y - center) ** 2) <= buffer_pixels**2

    if water_value == WaterValue.ONE:
        return ndimage.binary_dilation(mask, structure=struct).astype(np.uint8)
    # WaterValue.ZERO: water=0, land=1 — erode the land class instead.
    land = mask == 1
    eroded_land = ~ndimage.binary_dilation(~land, structure=struct)
    return eroded_land.astype(np.uint8)


def create_water_mask(
    bounds: tuple[float, float, float, float],
    output: Path,
    resolution: float | None = None,
    buffer_meters: float = 0.0,
    water_value: WaterValue = WaterValue.ZERO,
    overwrite: bool = False,
) -> Path:
    """Create a water mask GeoTIFF for the given bounds.

    Parameters
    ----------
    bounds : tuple[float, float, float, float]
        ``(west, south, east, north)`` in EPSG:4326 degrees.
    output : Path
        Output GeoTIFF path.
    resolution : float, optional
        Pixel size in degrees. If None, uses native tile resolution
        (~0.0001 deg).
    buffer_meters : float
        Buffer distance to expand water mask into land. Approximate
        conversion to pixels uses 111 km/degree at the AOI center latitude.
    water_value : WaterValue
        Output convention. Defaults to ``ZERO`` (water=0, land=1) which
        matches dolphin's mask convention.
    overwrite : bool
        If False and `output` exists, skip creation.

    Returns
    -------
    Path
        Path to the output raster.

    """
    output = Path(output)
    if output.exists() and not overwrite:
        logger.info(f"Water mask already exists: {output}")
        return output

    west, south, east, north = bounds
    tiles = _get_tile_urls(west, south, east, north)
    if not tiles:
        msg = f"No water mask tiles found for bounds {bounds}"
        raise ValueError(msg)

    tile_names = [t.rsplit("/", 1)[-1] for t in tiles]
    logger.info(f"Building water mask from {len(tiles)} tile(s): {tile_names}")

    vrt_options = gdal.BuildVRTOptions(resampleAlg="nearest")
    vrt_ds = gdal.BuildVRT("", tiles, options=vrt_options)
    if vrt_ds is None:
        msg = f"Failed to build VRT from tiles: {tiles}"
        raise RuntimeError(msg)

    if resolution is None:
        tile_ds = gdal.Open(tiles[0])
        resolution = float(tile_ds.GetGeoTransform()[1])
        tile_ds = None
    assert resolution is not None

    width = int(np.ceil((east - west) / resolution))
    height = int(np.ceil((north - south) / resolution))

    warp_options = gdal.WarpOptions(
        format="MEM",
        outputBounds=(west, south, east, north),
        width=width,
        height=height,
        resampleAlg="nearest",
    )
    mem_ds = gdal.Warp("", vrt_ds, options=warp_options)
    vrt_ds = None
    if mem_ds is None:
        msg = "Failed to warp water mask to target extent"
        raise RuntimeError(msg)

    data = mem_ds.GetRasterBand(1).ReadAsArray()
    geotransform = mem_ds.GetGeoTransform()
    projection = mem_ds.GetProjection()
    mem_ds = None

    # ASF tiles ship as 0=land, 1=water. Flip if the caller wants the
    # dolphin convention (0=water, 1=land).
    if water_value == WaterValue.ZERO:
        data = 1 - data

    if buffer_meters > 0:
        center_lat = (south + north) / 2
        meters_per_deg = 111_000 * np.cos(np.radians(center_lat))
        buffer_pixels = int(np.ceil(buffer_meters / (resolution * meters_per_deg)))
        logger.info(
            f"Buffering water by {buffer_pixels} pixels (~{buffer_meters:.0f} m)"
        )
        data = _buffer_mask(data, buffer_pixels, water_value)

    output.parent.mkdir(parents=True, exist_ok=True)
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(
        fspath(output),
        width,
        height,
        1,
        gdal.GDT_Byte,
        options=["COMPRESS=LZW", "TILED=YES"],
    )
    out_ds.SetGeoTransform(geotransform)
    out_ds.SetProjection(projection)
    band = out_ds.GetRasterBand(1)
    band.WriteArray(data)
    band.SetNoDataValue(255)
    if water_value == WaterValue.ZERO:
        band.SetDescription("water_mask: 0=water (invalid), 1=land (valid)")
    else:
        band.SetDescription("water_mask: 1=water (invalid), 0=land (valid)")
    band.FlushCache()
    out_ds = None

    logger.info(f"Created water mask: {output}")
    return output
