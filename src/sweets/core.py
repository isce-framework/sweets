from datetime import date, datetime

# from os import fspath
from pathlib import Path
from typing import Any, Optional, Tuple

from dask.distributed import Client
from dateutil.parser import parse
from dolphin import io, stitching, unwrap
from dolphin.interferogram import Network
from dolphin.workflows import group_by_burst
from dolphin.workflows.config import OPERA_DATASET_NAME
from pydantic import BaseModel, Extra, Field, PrivateAttr, validator

from ._burst_db import get_burst_db
from ._geocode_slcs import create_config_files, run_geocode
from ._interferograms import create_cor, create_ifg
from ._log import get_log, log_runtime
from ._orbits import download_orbits
from .dem import DEM
from .download import ASFQuery

logger = get_log(__name__)

# TODO: save AOI, pad the DEM a little beyond that
# figure out what to do with cropping the bursts... when to do that? probably before ifgs


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

    # Interferogram network
    # TODO: make into separate class
    looks: Tuple[int, int] = Field(
        (6, 12),
        description="Row looks, column looks. Default is 6, 12 (for 60x60 m).",
    )
    max_bandwidth: Optional[float] = Field(
        4,
        description="Form interferograms using the nearest n- dates",
    )
    max_temporal_baseline: Optional[float] = Field(
        None,
        description="Alt. to max_bandwidth: maximum temporal baseline in days.",
    )

    n_workers: int = Field(
        4,
        description="Number of workers to use for processing.",
    )
    threads_per_worker: int = Field(
        2,
        description="Number of threads per worker.",
    )
    overwrite: bool = Field(
        False,
        description="Overwrite existing files.",
    )
    _client: Client = PrivateAttr()
    # log_file: Path = Field(
    #     Path("sweets.log"),
    #     description="Path to log file.",
    # )

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
        # Expand the bbox a little bit so DEM fully covers the data
        bbox = values.get("bbox")
        dem_bbox = (
            bbox[0] - 0.15,
            bbox[1] - 0.15,
            bbox[2] + 0.15,
            bbox[3] + 0.15,
        )
        return DEM(bbox=dem_bbox)

    @validator("asf_query", pre=True, always=True)
    def _create_query(cls, v, values):
        if v is not None:
            return v
        # Shrink the data query bbox by a little bit
        bbox = values.get("bbox")
        query_bbox = (
            bbox[0] + 0.05,
            bbox[1] + 0.05,
            bbox[2] - 0.05,
            bbox[3] - 0.05,
        )
        meta = dict(
            bbox=query_bbox,
            start=values.get("start"),
            end=values.get("end"),
            relativeOrbit=values.get("track"),
        )

        if "orbit_dir" in values:
            # only set if they've passed one
            meta["orbit_dir"] = values["orbit_dir"]
        return ASFQuery(**meta)

    @validator("max_temporal_baseline")
    def _check_max_temporal_baseline(cls, v, values):
        """Make sure they didn't specify max_bandwidth and max_temporal_baseline."""
        max_bandwidth = values.get("max_bandwidth")
        if max_bandwidth is not None and v is not None:
            raise ValueError(
                "Cannot specify both max_bandwidth and max_temporal_baseline"
            )
        return v

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        # Start with 1 worker, scale later upon kicking off `run`
        self._client = Client(n_workers=1, threads_per_worker=self.threads_per_worker)

    @log_runtime
    def run(self):
        """Run the workflow."""
        # logger = get_log(filename=self.log_file)
        logger.info(f"Scaling dask cluster to {self.n_workers} workers")
        self._client.cluster.scale(self.n_workers)
        # TODO: background processing maybe with prefect
        # https://examples.dask.org/applications/prefect-etl.html

        dem_file = self._client.submit(self.dem.create)
        burst_db_file = self._client.submit(get_burst_db)
        # dem_create = delayed(self.dem.create)
        # dem_file = dem_create()
        # burst_db_file = delayed(get_burst_db)()

        # TODO: probably can download a few at a time
        downloaded_files = self._client.submit(self.asf_query.download)
        # Use .parent of the .result() so that next step depends on the result
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
        compass_cfg_files = create_config_files(
            slc_dir=slc_data_path,
            burst_db_file=burst_db_file,
            dem_file=dem_file,
            orbit_dir=self.orbit_dir,
            bbox=self.bbox,
            out_dir=Path("gslcs"),
            overwrite=self.overwrite,
        )

        # Run the geocodings
        gslc_futures = []
        for cfg_file in compass_cfg_files:
            gslc_futures.append(self._client.submit(run_geocode, cfg_file))
        gslc_files = self._client.gather(gslc_futures)

        # These GSLCs will all have their own bbox.
        # Make VRTs for each burst that all have the same specified bbox.

        # Group the SLCs by burst:
        # {'t078_165573_iw2': [PosixPath('gslcs/t078_165573_iw2/20221029/...
        burst_to_gslc = group_by_burst(gslc_files)
        # burst_to_ifgs = defaultdict(list)
        ifg_path_list = []
        for burst, gslc_files in burst_to_gslc.items():
            outdir = Path("interferograms") / burst
            outdir.mkdir(parents=True, exist_ok=True)
            network = Network(
                gslc_files,
                outdir=outdir,
                max_temporal_baseline=self.max_temporal_baseline,
                max_bandwidth=self.max_bandwidth,
                subdataset=OPERA_DATASET_NAME,
            )
            logger.info(
                f"Creating {len(network)} interferograms for burst {burst} in {outdir}"
            )

            ifg_futures = []
            for vrt_ifg in network.ifg_list:
                ifg_fut = self._client.submit(
                    create_ifg, vrt_ifg, self.looks, bbox=self.bbox
                )
                # ifg_fut = self._client.submit(
                #     create_ifg,
                #     vrt_ifg,
                #     self.looks,
                #     overlapping_with=dem_file,
                # )

                ifg_futures.append(ifg_fut)

            cur_ifg_paths = self._client.gather(ifg_futures)
            # burst_to_ifgs[burst].extend(cur_ifg_paths)
            ifg_path_list.extend(cur_ifg_paths)
            # Make sure we free up the memory
            self._client.cancel(ifg_futures)

        # #####################
        # Stitch interferograms
        # #####################
        grouped_images = stitching._group_by_date(ifg_path_list)

        stitched_dir = Path("interferograms") / "stitched"
        stitched_dir.mkdir(parents=True, exist_ok=True)
        stitched_ifg_files = []
        cor_files = []
        ifg_futures = []
        for dates, cur_images in grouped_images.items():
            logger.info(f"{dates}: Stitching {len(cur_images)} images.")
            stitched_dir.mkdir(parents=True, exist_ok=True)
            outfile = stitched_dir / (io._format_date_pair(*dates) + ".int")
            stitched_ifg_files.append(outfile)

            ifg_futures.append(
                self._client.submit(
                    stitching.merge_images,
                    cur_images,
                    outfile=outfile,
                    driver="ENVI",
                    overwrite=self.overwrite,
                )
            )
        self._client.gather(ifg_futures)  # create ifg returns nothing

        # Also need to write out a temp correlation file for unwrapping
        cor_futures = []
        for ifg_file in stitched_ifg_files:
            cor_futures.append(self._client.submit(create_cor, ifg_file))
        cor_files = self._client.gather(cor_futures)

        # #########################
        # Unwrap the interferograms
        # #########################
        unwrap_futures = []
        unwrapped_files = []
        for ifg_file, cor_file in zip(stitched_ifg_files, cor_files):
            outfile = ifg_file.with_suffix(".unw")
            if outfile.exists():
                logger.info(f"{outfile} exists. Skipping.")
                unwrapped_files.append(outfile)
            else:
                unwrap_futures.append(
                    self._client.submit(
                        unwrap.unwrap,
                        ifg_file,
                        cor_file,
                        outfile,
                        do_tile=False,  # Probably make this an option too
                        init_method="mst",  # TODO: make this an option?
                        looks=self.looks,
                    )
                )

        # Add in the rest of the ones we ran
        unwrapped_files.extend(self._client.gather(unwrap_futures))

        return unwrapped_files
