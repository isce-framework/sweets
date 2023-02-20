from datetime import date, datetime

# from os import fspath
from pathlib import Path
from typing import Any, Optional, Tuple

from dask.distributed import Client

# from dask import delayed
from dateutil.parser import parse
from pydantic import BaseModel, Extra, Field, PrivateAttr, validator

from ._burst_db import get_burst_db
from ._geocode_slcs import _create_config_files
from ._log import get_log, log_runtime
from ._orbits import download_orbits
from .dem import DEM
from .download import ASFQuery

logger = get_log()


class Workflow(BaseModel):
    """Class for end-to-end processing of Sentinel-1 data."""

    # Steps
    bbox: Tuple[float, ...] = Field(
        ...,
        description=(
            "lower left lon, lat, upper right format e.g."
            " bbox=(-150.2,65.0,-150.1,65.5)"
        ),
    )
    start: datetime = Field(
        None,
        description=(
            "Starting time for search. Many acceptable inputs e.g. '3 months and a day"
            " ago' 'May 30, 2018' '2010-10-30T00:00:00Z'"
        ),
    )
    end: datetime = Field(
        None,
        description=(
            "Ending time for search. Many acceptable inputs e.g. '3 months and a day"
            " ago' 'May 30, 2018' '2010-10-30T00:00:00Z'"
        ),
    )
    track: Optional[int] = Field(
        None,
        description="Path number",
    )
    dem: DEM = Field(
        None,
        description="DEM parameters.",
    )
    asf_query: ASFQuery = Field(
        None,
        description="ASF query parameters.",
    )
    orbit_dir: Path = Field(
        Path("orbits"),
        description="Directory for orbit files.",
    )
    n_workers: int = Field(
        4,
        description="Number of workers to use for processing.",
    )
    _client: Client = PrivateAttr()

    class Config:
        extra = Extra.forbid  # raise error if extra fields passed in

    @validator("start", "end", pre=True)
    def _parse_date(cls, v):
        if isinstance(v, datetime):
            return v
        elif isinstance(v, date):
            # Convert to datetime
            return datetime.combine(v, datetime.min.time())
        return parse(v)

    @validator("dem", pre=True, always=True)
    def _use_same_bbox(cls, v, values):
        if v is not None:
            return v
        bbox = values.get("bbox")
        return DEM(bbox=bbox)

    @validator("asf_query", pre=True, always=True)
    def _create_query(cls, v, values):
        if v is not None:
            return v
        bbox = values.get("bbox")
        return ASFQuery(
            bbox=bbox,
            start=values.get("start"),
            end=values.get("end"),
            relativeOrbit=values.get("track"),
        )

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        # Scale later
        logger.info("Starting Dask cluster")
        self._client = Client(n_workers=1, threads_per_worker=1)

    @log_runtime
    def run(self):
        """Run the workflow."""
        # start our Dask cluster

        logger.info(f"Scaling dask cluster to {self.n_workers} workers")
        self._client.cluster.scale(self.n_workers)
        # TODO: background processing maybe with prefect
        # https://examples.dask.org/applications/prefect-etl.html

        dem_file = self._client.submit(self.dem.create)
        burst_db_file = self._client.submit(get_burst_db)
        # dem_create = delayed(self.dem.create)
        # dem_file = dem_create()
        # burst_db_file = delayed(get_burst_db)()

        downloaded_files = self._client.submit(self.asf_query.download)
        # Use .parent so that next step knows it depends on the result
        slc_data_path = downloaded_files.result()[0].parent

        # asf_download = delayed(self.asf_query.download)
        # downloaded_files = asf_download()
        # slc_data_path = self.asf_query.out_dir

        orbit_files = self._client.submit(
            download_orbits, slc_data_path, self.orbit_dir
        )

        # Wait until all the files are downloaded
        downloaded_files, dem_file, burst_db_file, orbit_files = self._client.gather(
            # return self._client.gather(
            [downloaded_files, dem_file, burst_db_file, orbit_files]
        )

        # TODO: unzip, probably in download
        # any reason to do this async?
        cfg_files = _create_config_files(
            slc_dir=slc_data_path,
            burst_db_file=burst_db_file,
            dem_file=dem_file,
            orbit_dir=self.orbit_dir,
            bbox=self.bbox,
            out_dir=Path("gslcs"),
        )
        return cfg_files
