from __future__ import annotations

from os import fspath
from pathlib import Path

import sardem.dem
from loguru import logger

from sweets._log import log_runtime
from sweets._types import Filename
from sweets._water_mask import WaterValue
from sweets._water_mask import create_water_mask as _create_water_mask_tiles
from sweets.utils import get_cache_dir


@log_runtime
def create_dem(output_name: Filename, bbox: tuple[float, float, float, float]) -> Path:
    """Download a Copernicus Global DEM clipped to `bbox`."""
    output_name = Path(output_name).resolve()
    if output_name.exists():
        logger.info(f"DEM already exists: {output_name}")
        return output_name

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
    bbox: tuple[float, float, float, float],
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
