from os import fspath
from pathlib import Path
from typing import List

import sardem.dem
from pydantic import BaseModel, Field

from sweets._log import log_runtime
from sweets.utils import get_cache_dir


class DEM(BaseModel):
    """Class for downloading DEM data from ASF."""

    output_name: Path = Field(
        Path("dem.tif"),
        description="Output file name.",
    )
    bbox: List[float] = Field(
        ...,
        description=(
            "Bounding box in WGS84 lon/lat coordinates: [min_lon, min_lat, max_lon,"
            " max_lat]"
        ),
    )

    @log_runtime
    def create(self):
        """Create the DEM."""
        sardem.dem.main(
            output_name=fspath(self.output_name),
            bbox=self.bbox,
            data_source="COP",
            cache_dir=get_cache_dir(),
            output_format="GTiff",
            output_type="Float32",
        )
