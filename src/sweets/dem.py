from __future__ import annotations

import contextlib
from os import fspath
from pathlib import Path
from typing import Tuple

import sardem.dem
from loguru import logger

from sweets._log import log_runtime
from sweets._types import Filename
from sweets._water_mask import WaterValue
from sweets._water_mask import create_water_mask as _create_water_mask_tiles
from sweets.utils import get_cache_dir


def _egm2008_shift(lon: float, lat: float) -> float:
    """Return the height change PROJ applies at (lon, lat) from EGM2008 to WGS84 ellipsoidal heights."""
    from osgeo import osr

    src, dst = osr.SpatialReference(), osr.SpatialReference()
    src.SetFromUserInput("EPSG:4326+3855")
    dst.ImportFromEPSG(4979)
    for srs in (src, dst):
        srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return osr.CoordinateTransformation(src, dst).TransformPoint(lon, lat, 0.0)[2]


@contextlib.contextmanager
def _geoid_grid_access(bbox: Tuple[float, float, float, float]):
    """Let PROJ fetch the EGM2008 grid while the DEM is made, or raise if it cannot.

    sardem converts Copernicus heights (EGM2008) to WGS84 ellipsoidal heights with
    a GDAL warp. PROJ needs the geoid grid for that; conda-forge `proj` ships no
    geoid grids and PROJ network access is off by default. Without the grid, PROJ
    applies a no-op vertical shift and the DEM silently keeps geoid heights (tens of
    metres off), which then biases every radar-geometry step that uses it.

    Network access must be enabled before the first lookup because PROJ remembers a
    grid it could not find. A locally installed grid is used as is. The previous
    setting is restored afterwards.
    """
    from osgeo import osr

    previous = osr.GetPROJEnableNetwork()
    osr.SetPROJEnableNetwork(True)
    try:
        lon, lat = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
        if abs(_egm2008_shift(lon, lat)) < 1e-6:
            raise RuntimeError(
                "PROJ cannot apply the EGM2008 geoid, so the DEM would keep geoid heights "
                "instead of WGS84 ellipsoidal heights. Allow network access, or install the "
                "grid (`projsync --file us_nga_egm08_25.tif --user-writable-directory`, or "
                "the conda-forge `proj-data` package), then retry."
            )
        yield
    finally:
        osr.SetPROJEnableNetwork(previous)


@log_runtime
def create_dem(output_name: Filename, bbox: Tuple[float, float, float, float]) -> Path:
    """Download a Copernicus Global DEM clipped to `bbox`, in WGS84 ellipsoidal heights."""
    output_name = Path(output_name).resolve()
    if output_name.exists():
        logger.info(f"DEM already exists: {output_name}")
        return output_name

    with _geoid_grid_access(bbox):
        sardem.dem.main(
            output_name=fspath(output_name),
            bbox=bbox,
            data_source="COP",
            cache_dir=get_cache_dir(),
            output_format="GTiff",
            output_type="Float32",
        )
    return output_name


@log_runtime
def create_water_mask(
    output_name: Path,
    bbox: Tuple[float, float, float, float],
    buffer_meters: float = 0.0,
) -> Path:
    """Create a high-resolution binary land(1) / water(0) mask.

    Mosaics ASF's `WATER_MASK/TILES` product (OpenStreetMap + ESA WorldCover,
    served as 5-degree GeoTIFFs at ~0.0001-deg native posting) for `bbox`,
    inverts to dolphin's convention (0=water/invalid, 1=land/valid), and
    optionally dilates the water class by `buffer_meters` to mask shoreline
    noise.

    Replaces the older ``NASA_WATER`` / SRTMSWBD path (broken on macOS,
    restricted to ENVI) and the brief Copernicus-DEM threshold hack.
    """
    return _create_water_mask_tiles(
        bounds=bbox,
        output=Path(output_name).resolve(),
        buffer_meters=buffer_meters,
        water_value=WaterValue.ZERO,
    )
