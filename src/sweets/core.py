import json
import sys
from datetime import date, datetime
from io import StringIO
from pathlib import Path
from typing import Any, Optional, TextIO, Tuple, Union

from dask.distributed import Client, Future
from dateutil.parser import parse
from dolphin import io, stitching, unwrap
from dolphin.interferogram import Network
from dolphin.workflows import group_by_burst
from dolphin.workflows.config import OPERA_DATASET_NAME
from pydantic import BaseModel, Extra, Field, PrivateAttr, validator
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap

from ._burst_db import get_burst_db
from ._config import _add_comments
from ._geocode_slcs import create_config_files, run_geocode
from ._interferograms import create_cor, create_ifg
from ._log import get_log, log_runtime
from ._orbit import download_orbits
from ._types import Filename
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
            "Starting time for search. Can be datetime or string (goes to"
            " `dateutil.parse`)"
        ),
    )
    end: datetime = Field(
        default_factory=datetime.now,
        description=(
            "Ending time for search. Can be datetime or string (goes to"
            " `dateutil.parse`)"
        ),
    )
    track: int = Field(
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
    max_bandwidth: Optional[int] = Field(
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
        extra = Extra.allow  # Let us set arbitrary attributes later

    @validator("start", "end", pre=True)
    def _parse_date(cls, v):
        if isinstance(v, datetime):
            return v
        elif isinstance(v, date):
            # Convert to datetime
            return datetime.combine(v, datetime.min.time())
        return parse(v)

    @validator("dem", pre=True, always=True)
    def _use_same_bbox(cls, v, values, config, field):
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
        bbox = values.get("bbox")
        track = values.get("track")
        if track is None:
            raise ValueError("Must specify either track for asf_query")

        asf_params = dict(
            bbox=bbox,
            start=values.get("start"),
            end=values.get("end"),
            relativeOrbit=track,
        )
        if "orbit_dir" in values:
            # only set if they've passed one
            asf_params["orbit_dir"] = values["orbit_dir"]
        return ASFQuery(**asf_params)

    @validator("max_temporal_baseline", pre=True)
    def _check_max_temporal_baseline(cls, v, values):
        """Make sure they didn't specify max_bandwidth and max_temporal_baseline."""
        if v is None:
            return v
        max_bandwidth = values.get("max_bandwidth")
        if max_bandwidth == cls.schema()["properties"]["max_bandwidth"]["default"]:
            values["max_bandwidth"] = None
        else:
            raise ValueError(
                "Cannot specify both max_bandwidth and max_temporal_baseline"
            )
        return v

    def _to_yaml_obj(self) -> CommentedMap:
        # Make the YAML object to add comments to
        # We can't just do `dumps` for some reason, need a stream
        y = YAML()
        ss = StringIO()
        y.dump(json.loads(self.json(by_alias=True)), ss)
        yaml_obj = y.load(ss.getvalue())
        return yaml_obj

    def to_yaml(
        self, output_path: Union[Filename, TextIO] = sys.stdout, with_comments=True
    ):
        """Save workflow configuration as a yaml file.

        Used to record the default-filled version of a supplied yaml.

        Parameters
        ----------
        output_path : Pathlike
            Path to the yaml file to save.
        with_comments : bool, default = False.
            Whether to add comments containing the type/descriptions to all fields.
        """
        yaml_obj = self._to_yaml_obj()

        if with_comments:
            _add_comments(yaml_obj, self.schema())

        y = YAML()
        if hasattr(output_path, "write"):
            y.dump(yaml_obj, output_path)
        else:
            with open(output_path, "w") as f:
                y.dump(yaml_obj, f)

    @classmethod
    def from_yaml(cls, yaml_path: Filename):
        """Load a workflow configuration from a yaml file.

        Parameters
        ----------
        yaml_path : Pathlike
            Path to the yaml file to load.

        Returns
        -------
        Config
            Workflow configuration
        """
        y = YAML(typ="safe")
        with open(yaml_path, "r") as f:
            data = y.load(f)

        return cls(**data)

    def __init__(self, start_dask=True, **data: Any) -> None:
        super().__init__(**data)
        # Start with 1 worker, scale later upon kicking off `run`
        if start_dask:
            self._client = Client(
                n_workers=1, threads_per_worker=self.threads_per_worker
            )
        else:
            self._client = None

    @log_runtime
    def _download_dem(self) -> Future:
        """Kick off download/creation the DEM."""
        return self._client.submit(self.dem.create)

    @log_runtime
    def _download_burst_db(self) -> Future:
        """Kick off download of burst database to get the GSLC bbox/EPSG."""
        return self._client.submit(get_burst_db)

    @log_runtime
    def _download_rslcs(self) -> Tuple[Future, Future]:
        """Download Sentinel zip files from ASF."""
        # TODO: probably can download a few at a time
        rslc_futures = self._client.submit(self.asf_query.download)
        return rslc_futures

    @log_runtime
    def _geocode_slcs(self, slc_files, dem_file, burst_db_file):
        # TODO: unzip, probably in download
        # any reason to do this async?
        compass_cfg_files = create_config_files(
            slc_dir=slc_files[0].parent,
            burst_db_file=burst_db_file,
            dem_file=dem_file,
            orbit_dir=self.orbit_dir,
            bbox=self.bbox,
            out_dir=Path("gslcs"),
            overwrite=self.overwrite,
        )

        def cfg_to_filename(cfg_path: Path) -> str:
            # Convert the YAML filename to a .h5 filename with date switched
            # geo_runconfig_20221029_t078_165578_iw3.yaml -> t078_165578_iw2_20221029.h5
            date = cfg_path.name.split("_")[2]
            burst = "_".join(cfg_path.stem.split("_")[3:])
            return f"{burst}_{date}.h5"

        # Check which ones we have without submitting a future
        gslc_futures = []
        gslc_files = []
        existing_paths = Path("gslcs").glob("**/*.h5")
        name_to_paths = {p.name: p for p in existing_paths}
        logger.info(f"Found {len(name_to_paths)} existing GSLC files")
        for cfg_file in compass_cfg_files:
            name = cfg_to_filename(cfg_file)
            if name in name_to_paths:
                gslc_files.append(name_to_paths[name])
            else:
                # Run the geocoding if we dont have it already
                gslc_futures.append(self._client.submit(run_geocode, cfg_file))
        gslc_files.extend(self._client.gather(gslc_futures))
        return gslc_files

    @log_runtime
    def _create_burst_interferograms(self, gslc_files):
        # Group the SLCs by burst:
        # {'t078_165573_iw2': [PosixPath('gslcs/t078_165573_iw2/20221029/...], 't078_...
        burst_to_gslc = group_by_burst(gslc_files)
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
                f"{len(network)} interferograms to create for burst {burst} in {outdir}"
            )
            logger.info(f"{len(list(outdir.glob('*.tif')))} existing interferograms")
            ifg_futures = []
            for vrt_ifg in network.ifg_list:
                ifg_fut = self._client.submit(
                    create_ifg, vrt_ifg, self.looks, bbox=self.bbox
                )
                #     overlapping_with=dem_file,

                ifg_futures.append(ifg_fut)

            cur_ifg_paths = self._client.gather(ifg_futures)
            ifg_path_list.extend(cur_ifg_paths)
            # Make sure we free up the memory
            self._client.cancel(ifg_futures)

        return ifg_path_list

    def _stitch_interferograms(self, ifg_path_list):
        grouped_images = stitching._group_by_date(ifg_path_list)

        # TODO: Should prob make sure these dirs are in some good place
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
        return stitched_ifg_files, cor_files

    def _unwrap_ifgs(self, ifg_files, cor_files):
        unwrap_futures = []
        unwrapped_files = []
        outdir = Path("interferograms") / "unwrapped"
        outdir.mkdir(parents=True, exist_ok=True)
        for ifg_file, cor_file in zip(ifg_files, cor_files):
            # outfile = ifg_file.with_suffix(".unw")
            outfile = outdir / ifg_file.name.replace(".int", ".unw")
            unwrapped_files.append(outfile)
            if outfile.exists():
                logger.info(f"{outfile} exists. Skipping.")
            else:
                logger.info(f"Unwrapping {ifg_file} to {outfile}")
                unwrap_futures.append(
                    self._client.submit(
                        unwrap.unwrap,
                        ifg_file,
                        outfile,
                        cor_file,
                        looks=self.looks,
                        do_tile=False,  # Probably make this an option too
                        init_method="mst",  # TODO: make this an option?
                        alt_line_data=False,
                    )
                )

        # Add in the rest of the ones we ran
        completed_procs = self._client.gather(unwrap_futures)
        logger.info(f"Unwrapped {len(completed_procs)} interferograms.")
        # TODO: Maybe check the return codes here? or log the snaphu output?
        return unwrapped_files

    @log_runtime
    def run(self, config_file: Filename = "sweets_config.yaml"):
        """Run the workflow."""
        logger.info("Saving config to {config_file}")
        self.to_yaml(config_file)
        # TODO: background processing maybe with prefect
        # https://examples.dask.org/applications/prefect-etl.html
        # TODO: maybe log the run times like this:
        # https://distributed.dask.org/en/stable/logging.html
        logger.info(f"Scaling dask cluster to {self.n_workers} workers")
        self._client.cluster.scale(self.n_workers)

        dem_fut = self._download_dem()
        burst_db_fut = self._download_burst_db()
        rslc_futures = self._download_rslcs()
        # Use .parent of the .result() so that next step depends on the result
        rslc_data_path = rslc_futures.result()[0].parent
        orbit_futures = self._client.submit(
            download_orbits, rslc_data_path, self.orbit_dir
        )

        # Gather the futures once everything is downloaded
        dem_file, burst_db_file = self._client.gather([dem_fut, burst_db_fut])
        rslc_files = rslc_futures.result()
        orbit_futures.result()

        gslc_files = self._geocode_slcs(rslc_files, dem_file, burst_db_file)
        ifg_path_list = self._create_burst_interferograms(gslc_files)
        stitched_ifg_files, cor_files = self._stitch_interferograms(ifg_path_list)
        unwrapped_files = self._unwrap_ifgs(stitched_ifg_files, cor_files)

        return unwrapped_files
