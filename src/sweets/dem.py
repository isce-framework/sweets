from os import fspath
from pathlib import Path
from typing import Tuple

import sardem.dem
from osgeo import gdal

from sweets._log import get_log, log_runtime
from sweets._types import Filename
from sweets.utils import get_cache_dir

logger = get_log(__name__)


@log_runtime
def create_dem(output_name: Filename, bbox: Tuple[float, float, float, float]) -> Path:
    """Create the output file."""
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
    output_name: Path, bbox: Tuple[float, float, float, float]
) -> Path:
    """Create the output file."""
    output_name = Path(output_name).resolve()
    if output_name.exists():
        logger.info(f"Water mask already exists: {output_name}")
        return output_name

    sardem.dem.main(
        output_name=fspath(output_name),
        bbox=bbox,
        cache_dir=get_cache_dir(),
        output_format="ROI_PAC",
        data_source="NASA_WATER",
        output_type="uint8",
    )
    # Flip the mask so that 1 is land and 0 is water
    ds = gdal.Open(fspath(output_name), gdal.GA_Update)
    band = ds.GetRasterBand(1)
    band.WriteArray(1 - band.ReadAsArray())
    return output_name
