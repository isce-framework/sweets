from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor, wait
from functools import partial
from pathlib import Path
from typing import Any, Literal, Optional

import h5py
import numpy as np
from dolphin import io, stitching, unwrap
from dolphin._types import Bbox
from dolphin.interferogram import Network
from dolphin.utils import group_by_date, set_num_threads
from dolphin.workflows import group_by_burst
from dolphin.workflows.config import OPERA_DATASET_ROOT, YamlModel
from pydantic import ConfigDict, Field, field_validator, model_validator
from shapely import geometry, wkt

from ._burst_db import get_burst_db
from ._geocode_slcs import create_config_files, run_geocode, run_static_layers
from ._log import get_log, log_runtime
from ._netrc import setup_nasa_netrc
from ._orbit import download_orbits
from ._types import Filename
from .dem import create_dem, create_water_mask
from .download import ASFQuery
from .interferogram import InterferogramOptions, create_cor, create_ifg

logger = get_log(__name__)

UNW_SUFFIX = ".unw.tif"


class Workflow(YamlModel):
    """Class for end-to-end processing of Sentinel-1 data."""

    work_dir: Path = Field(
        default_factory=Path.cwd,
        description="Root of working directory for processing.",
        validate_default=True,
    )

    bbox: Bbox = Field(
        None,
        description=(
            "Area of interest: (left, bottom, right, top) longitude/latitude "
            "e.g. `bbox=(-150.2,65.0,-150.1,65.5)`"
        ),
    )
    wkt: Optional[str] = Field(
        None,
        description="Well Known Text (WKT) string (overrides bbox)",
    )

    orbit_dir: Path = Field(
        Path("orbits"),
        description="Directory for orbit files.",
        validate_default=True,
    )

    asf_query: ASFQuery
    skip_download_if_exists: bool = Field(
        True,
        description=(
            "Don't re-query ASF if there's any existing data in the download directory."
            " Otherwise, will re-query and only skip files that match checksums (done"
            " by aria2)."
        ),
    )
    interferogram_options: InterferogramOptions = Field(
        default_factory=InterferogramOptions
    )
    do_unwrap: bool = Field(
        True,
        description="Run the unwrapping step for all interferograms.",
    )
    slc_posting: tuple[float, float] = Field(
        (10, 5),
        description="Spacing of geocoded SLCs (in meters) along the (y, x)-directions.",
    )
    pol_type: Literal["co-pol", "cross-pol"] = Field(
        "co-pol",
        description="Type of polarization to process for GSLCs",
    )

    n_workers: int = Field(
        1,
        description="Number of workers to use for processing.",
    )
    threads_per_worker: int = Field(
        8,
        description=(
            "Number of threads per worker, set using OMP_NUM_THREADS. This affects the"
            " number of threads used by isce3 geocodeSlc, as well as the number of"
            " threads numpy uses."
        ),
    )
    overwrite: bool = Field(
        False,
        description="Overwrite existing files.",
    )
    model_config = ConfigDict(extra="allow")

    @field_validator("wkt", mode="before")
    @classmethod
    def _check_file_and_parse_wkt(cls, v):
        if v is not None:
            if Path(v).exists():
                v = Path(v).read_text().strip()

            try:
                wkt.loads(v)
            except Exception as e:
                raise ValueError(f"Invalid WKT string: {e}")
        return v

    # Note: this model_validator's `info.data` only contains the fields that have
    # been passed in by a user.
    @model_validator(mode="before")
    @classmethod
    def _check_unset_dirs(cls, values: Any) -> "Workflow":
        # TODO: Use the newer checks for fields set
        if isinstance(values, dict):
            if "asf_query" not in values:
                values["asf_query"] = {}
            # Orbits dir and data dir can be outside the working dir if someone
            # wants to point to existing data.
            # So we only want to move them inside the working dir if they weren't
            # explicitly set.
            values["_orbit_dir_is_set"] = "orbit_dir" in values
            values["_data_dir_is_set"] = "out_dir" in values["asf_query"]

            # also if they passed a wkt to the outer constructor, we need to
            # pass through to the ASF query
            values["asf_query"]["wkt"] = values.get("asf_query", {}).get(
                "wkt"
            ) or values.get("wkt")
            values["asf_query"]["bbox"] = values.get("asf_query", {}).get(
                "bbox"
            ) or values.get("bbox")
            # sync the other way too:
            values["wkt"] = values["asf_query"]["wkt"]
            values["bbox"] = values["asf_query"]["bbox"]
            if not values.get("bbox") and not values.get("wkt"):
                raise ValueError("Must specify either `bbox` or `wkt`")

        return values

    # expanduser and resolve each of the dirs:
    @field_validator("work_dir", "orbit_dir")
    @classmethod
    def _expand_dirs(cls, v):
        return Path(v).expanduser().resolve()

    # # while this one has all the fields
    # @model_validator()
    # def _move_inside_workdir(self):
    #     # TODO: pydantic has made this easier with new attributes to check attrs set
    #     if not values["_orbit_dir_is_set"]:
    #         values["orbit_dir"] = (values["work_dir"] / values["orbit_dir"]).resolve()
    #     if not values["_data_dir_is_set"] and "asf_query" in values:
    #         values["asf_query"].out_dir = (
    #             values["work_dir"] / values["asf_query"].out_dir
    #         ).resolve()

    #     return values

    @model_validator(mode="after")
    def _set_bbox_and_wkt(self, values):
        # If they've specified a bbox, set the wkt
        if not self.bbox:
            self.bbox = wkt.loads(self.wkt).bounds
        else:
            # otherwise, make WKT just a 5 point polygon
            self.wkt = wkt.dumps(geometry.box(*self.bbox))
        return values

    def save(self, config_file: Filename = "sweets_config.yaml"):
        """Save the workflow configuration."""
        logger.info(f"Saving config to {config_file}")
        self.to_yaml(config_file)

    @classmethod
    def load(cls, config_file: Filename = "sweets_config.yaml"):
        """Load the workflow configuration."""
        logger.info(f"Loading config from {config_file}")
        return cls.from_yaml(config_file)

    # Override the constructor to allow recursively construct without validation
    @classmethod
    def construct(cls, **kwargs):
        if "asf_query" not in kwargs:
            kwargs["asf_query"] = ASFQuery._construct_empty()
        return super().construct(
            **kwargs,
        )

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        # Track the directories that need to be created at start of workflow
        self.log_dir = self.work_dir / "logs"
        self.gslc_dir = self.work_dir / "gslcs"
        self.geom_dir = self.work_dir / "geometry"
        self.ifg_dir = self.work_dir / "interferograms"
        self.stitched_ifg_dir = self.ifg_dir / "stitched"
        self.unw_dir = self.ifg_dir / "unwrapped"
        self._dem_filename = self.work_dir / "dem.tif"
        self._water_mask_filename = self.work_dir / "watermask.flg"

        # Expanded version used for internal processing
        self._dem_bbox = (
            self.bbox[0] - 0.25,
            self.bbox[1] - 0.25,
            self.bbox[2] + 0.25,
            self.bbox[3] + 0.25,
        )

    # Intermediate outputs:
    # From step 1:
    def _get_existing_rslcs(self) -> list[Path]:
        ext = ".SAFE" if self.asf_query.unzip else ".zip"
        return sorted(self.asf_query.out_dir.glob("S*" + ext))

    # From step 2:
    def _get_existing_gslcs(self) -> list[Path]:
        return sorted(self.gslc_dir.glob("t*/*/t*.h5"))

    def _get_burst_static_layers(self) -> list[Path]:
        return sorted(self.gslc_dir.glob("t*/*/static_*.h5"))

    # From step 3:
    def _get_existing_burst_ifgs(self) -> list[Path]:
        return sorted(self.ifg_dir.glob("t*/2*_2*.tif"))

    # From step 4:
    def _get_existing_stitched_ifgs(self) -> tuple[list[Path], list[Path]]:
        ifg_file_list = sorted(Path(self.ifg_dir / "stitched").glob("2*.int"))
        cor_file_list = [f.with_suffix(".cor") for f in ifg_file_list]
        return ifg_file_list, cor_file_list

    # Download helpers to kick off for step 1:
    def _download_dem(self) -> Future:
        """Kick off download/creation the DEM."""
        return self._client.submit(create_dem, self._dem_filename, self._dem_bbox)

    def _download_burst_db(self) -> Future:
        """Kick off download of burst database to get the GSLC bbox/EPSG."""
        return self._client.submit(get_burst_db)

    def _download_water_mask(self) -> Future:
        """Kick off download of water mask."""
        return self._client.submit(
            create_water_mask, self._water_mask_filename, self._dem_bbox
        )

    def _download_rslcs(self) -> list[Path]:
        """Download Sentinel zip files from ASF."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        # The final name will depend on if we're unzipping or not
        existing_files = self._get_existing_rslcs()

        if existing_files and self.skip_download_if_exists:
            logger.info(
                f"Found {len(existing_files)} existing files in"
                f" {self.asf_query.out_dir}. Skipping download."
            )
            return existing_files

        # If we didn't have any, we need to download them
        # TODO: how should we handle partial/failed downloads... do we really
        # want to re-search for them each time?
        # Maybe there can be a "force" flag to re-download everything?
        # or perhaps an API search, then if the number matches, we can skip
        # rather than let aria2c start and do the checksums
        return self.asf_query.download(log_dir=self.log_dir)

    @log_runtime
    def _geocode_slcs(self, slc_files, dem_file, burst_db_file):
        self.log_dir.mkdir(parents=True, exist_ok=True)
        compass_cfg_files = create_config_files(
            slc_dir=slc_files[0].parent,
            burst_db_file=burst_db_file,
            dem_file=dem_file,
            orbit_dir=self.orbit_dir,
            bbox=self.bbox,
            y_posting=self.slc_posting[0],
            x_posting=self.slc_posting[1],
            pol_type=self.pol_type,
            out_dir=self.gslc_dir,
            overwrite=self.overwrite,
            using_zipped=not self.asf_query.unzip,
        )

        def cfg_to_filename(cfg_path: Path) -> str:
            # Convert the YAML filename to a .h5 filename with date switched
            # geo_runconfig_20221029_t078_165578_iw3.yaml -> t078_165578_iw2_20221029.h5
            date = cfg_path.name.split("_")[2]
            burst = "_".join(cfg_path.stem.split("_")[3:])
            return f"{burst}_{date}.h5"

        # Check which ones we have without submitting a future
        all_gslc_files = []
        todo_gslc = []
        existing_paths = self._get_existing_gslcs()
        name_to_paths = {p.name: p for p in existing_paths}
        logger.info(f"Found {len(name_to_paths)} existing GSLC files")
        for cfg_file in compass_cfg_files:
            name = cfg_to_filename(cfg_file)
            if name in name_to_paths:
                all_gslc_files.append(name_to_paths[name])
            else:
                # Run the geocoding if we dont have it already
                todo_gslc.append(cfg_file)

        run_func = partial(run_geocode, log_dir=self.log_dir)
        with ProcessPoolExecutor(max_workers=self.n_workers) as _client:
            new_files = _client.map(run_func, todo_gslc)

        new_files = list(new_files)
        all_gslc_files.extend(new_files)

        # Get the first config file (by date) for each of the bursts.
        # We only need to create the static layers once per burst
        existing_static_paths = self._get_burst_static_layers()
        name_to_paths = {p.name: p for p in existing_static_paths}
        logger.info(f"Found {len(name_to_paths)} existing geometry files")
        day1_cfg_paths = [
            paths[0] for paths in group_by_burst(compass_cfg_files).values()
        ]
        static_files = []
        todo_static = []
        for cfg_file in day1_cfg_paths:
            name = cfg_to_filename(cfg_file)
            if name in name_to_paths:
                static_files.append(name_to_paths[name])
            else:
                # Run the geocoding if we dont have it already
                todo_static.append(cfg_file)

        run_func = partial(run_static_layers, log_dir=self.log_dir)
        with ProcessPoolExecutor(max_workers=self.n_workers) as _client:
            new_files = _client.map(run_func, todo_static)

    @log_runtime
    def _stitch_geometry(self, geom_path_list):
        self.geom_dir.mkdir(parents=True, exist_ok=True)

        file_list = []
        # TODO: do I want to handle this missing data problem differently?
        # if there's a burst with the 1st day missing so the static_layers_
        # file is a different date... is that a problem?
        for burst, files in group_by_burst(geom_path_list, minimum_images=1).items():
            if len(files) > 1:
                logger.warning(f"Found {len(files)} static_layers files for {burst}")
            file_list.append(files[0])
        logger.info(f"Stitching {len(file_list)} images.")

        # Convert row/col looks to strides for the right shape
        looks = self.interferogram_options.looks
        strides = {"x": looks[1], "y": looks[0]}
        stitched_geom_files = []
        # local_incidence_angle needed by anyone?
        datasets = ["los_east", "los_north", "layover_shadow_mask"]
        for ds_name in datasets:
            outfile = self.geom_dir / f"{ds_name}.tif"
            logger.info(f"Creating {outfile}")
            stitched_geom_files.append(outfile)
            # Used to be:
            # /science/SENTINEL1/CSLC/grids/static_layers
            # we might also move this to dolphin if we do use the layers
            ds_path = f"{OPERA_DATASET_ROOT}/data/{ds_name}"
            cur_files = [io.format_nc_filename(f, ds_name=ds_path) for f in file_list]

            stitching.merge_images(
                cur_files,
                outfile=outfile,
                driver="GTiff",
                out_bounds=self.bbox,
                out_bounds_epsg=4326,
                target_aligned_pixels=True,
                in_nodata=0,
                strides=strides,
                resample_alg="nearest",
                overwrite=self.overwrite,
            )

        # Create the height (from dem) at the same resolution as the interferograms
        height_file = self.geom_dir / "height.tif"
        logger.info(f"Creating {height_file}")
        stitched_geom_files.append(height_file)
        stitching.warp_to_match(
            input_file=self._dem_filename,
            match_file=stitched_geom_files[0],
            output_file=height_file,
            resample_alg="cubic",
        )

        return stitched_geom_files

    @staticmethod
    def _get_subdataset(f):
        if not (str(f).endswith(".h5") or str(f).endswith(".nc")):
            return ""
        with h5py.File(f) as hf:
            for pol_str in ["VV", "HV", "VH", "HH"]:
                dset = f"/data/{pol_str}"
                if dset in hf:
                    return dset

    @log_runtime
    def _create_burst_interferograms(self, gslc_files):
        # Group the SLCs by burst:
        # {'t078_165573_iw2': [PosixPath('gslcs/t078_165573_iw2/20221029/...], 't078_...
        burst_to_gslc = group_by_burst(gslc_files)
        burst_to_ifg = group_by_burst(self._get_existing_burst_ifgs())
        ifg_path_list = []
        for burst, gslc_files in burst_to_gslc.items():
            subdatasets = [self._get_subdataset(f) for f in gslc_files]
            outdir = self.ifg_dir / burst
            outdir.mkdir(parents=True, exist_ok=True)
            network = Network(
                gslc_files,
                outdir=outdir,
                max_temporal_baseline=self.interferogram_options.max_temporal_baseline,
                max_bandwidth=self.interferogram_options.max_bandwidth,
                subdataset=subdatasets,
                write=False,
            )
            logger.info(
                f"{len(network)} interferograms to create for burst {burst} in {outdir}"
            )
            cur_existing = burst_to_ifg.get(burst, [])
            logger.info(f"{len(cur_existing)} existing interferograms")
            if len(network) == len(cur_existing):
                ifg_path_list.extend(cur_existing)
            else:
                ifg_futures = []
                with ThreadPoolExecutor(max_workers=self.n_workers) as _client:
                    for vrt_ifg in network.ifg_list:
                        outfile = vrt_ifg.path.with_suffix(".tif")
                        ifg_fut = _client.submit(
                            create_ifg,
                            vrt_ifg.ref_slc,
                            vrt_ifg.sec_slc,
                            outfile,
                            looks=self.interferogram_options.looks,
                        )

                        ifg_futures.append(ifg_fut)

                    for fut in ifg_futures:
                        ifg_path_list.append(fut.result())

        return ifg_path_list

    @log_runtime
    def _stitch_interferograms(self, ifg_path_list):
        self.stitched_ifg_dir.mkdir(parents=True, exist_ok=True)
        grouped_images = group_by_date(ifg_path_list)
        stitched_ifg_files = []
        for dates, cur_images in grouped_images.items():
            logger.info(f"{dates}: Stitching {len(cur_images)} images.")
            outfile = self.stitched_ifg_dir / (io._format_date_pair(*dates) + ".int")
            stitched_ifg_files.append(outfile)

            stitching.merge_images(
                cur_images,
                outfile=outfile,
                driver="ENVI",
                out_bounds=self.bbox,
                out_bounds_epsg=4326,
                target_aligned_pixels=True,
                overwrite=self.overwrite,
            )

        # Also need to write out a temp correlation file for unwrapping
        with ProcessPoolExecutor(max_workers=self.n_workers) as _client:
            cor_files = list(_client.map(create_cor, stitched_ifg_files))

        return stitched_ifg_files, cor_files

    def _unwrap_ifgs(self, ifg_files, cor_files):
        unwrapped_files = []
        if not self.do_unwrap:
            logger.info("Skipping unwrapping")
            return unwrapped_files

        self.unw_dir.mkdir(parents=True, exist_ok=True)
        # Warp the water mask to match the interferogram
        self._warped_water_mask = self._water_mask_filename.parent / "warped_mask.tif"
        if self._warped_water_mask.exists():
            logger.info(f"Mask already exists at {self._warped_water_mask}")
        else:
            stitching.warp_to_match(
                input_file=self._water_mask_filename,
                match_file=ifg_files[0],
                output_file=self._warped_water_mask,
            )

        ifg_to_future = {}
        with ProcessPoolExecutor(max_workers=self.n_workers) as _client:
            for ifg_file, cor_file in zip(ifg_files, cor_files):
                outfile = self.unw_dir / ifg_file.name.replace(".int", UNW_SUFFIX)
                unwrapped_files.append(outfile)
                if outfile.exists():
                    logger.info(f"Skipping {outfile}, already exists.")
                else:
                    ifg_to_future[ifg_file] = _client.submit(
                        unwrap.unwrap,
                        ifg_filename=ifg_file,
                        corr_filename=cor_file,
                        unw_filename=outfile,
                        nlooks=int(np.prod(self.interferogram_options.looks)),
                        mask_file=self._warped_water_mask,
                        # do_tile=False,  # Probably make this an option too
                        init_method="mst",  # TODO: make this an option?
                    )

            # Add in the rest of the ones we ran
            for ifg_file, f in ifg_to_future.items():
                logger.info(f"Unwrapping {ifg_file}")
                f.result()
        # TODO: Maybe check the return codes here? or log the snaphu output?
        return unwrapped_files

    @log_runtime
    def run(self, starting_step: int = 1):
        """Run the workflow."""
        setup_nasa_netrc()
        set_num_threads(self.threads_per_worker)

        # First step: data download
        logger.info(f"Setting up {self.n_workers} workers for ThreadPoolExecutor")
        if starting_step <= 1:
            with ThreadPoolExecutor(max_workers=self.n_workers) as _client:
                # TODO: fix this odd workaround once the isce3 hanging issues
                # are being resolved
                self._client = _client

                dem_fut = self._download_dem()
                burst_db_fut = self._download_burst_db()
                water_mask_future = self._download_water_mask()
                # Gather the futures once everything is downloaded
                burst_db_file = burst_db_fut.result()
                dem_fut.result()
                wait([water_mask_future])

            rslc_files = self._download_rslcs()

        # Second step:
        if starting_step <= 2:
            burst_db_file = get_burst_db()
            orbits_future = self._client.submit(
                download_orbits, self.asf_query.out_dir, self.orbit_dir
            )
            rslc_files = self._get_existing_rslcs()
            wait([orbits_future])
            self._geocode_slcs(rslc_files, self._dem_filename, burst_db_file)

            geom_path_list = self._get_burst_static_layers()
            logger.info(f"Found {len(geom_path_list)} burst static layers")
            self._stitch_geometry(geom_path_list)

        if starting_step <= 3:
            gslc_files = self._get_existing_gslcs()
            logger.info(
                f"Found {len(gslc_files)} existing GSLC files in {self.gslc_dir}"
            )
            self._create_burst_interferograms(gslc_files)

        if starting_step <= 4:
            logger.info(f"Searching for existing burst ifgs in {self.ifg_dir}")
            ifg_path_list = self._get_existing_burst_ifgs()
            logger.info(f"Found {len(ifg_path_list)} burst ifgs")

            self._stitch_interferograms(ifg_path_list)

        stitched_ifg_files, cor_files = self._get_existing_stitched_ifgs()
        logger.info(f"Found {len(stitched_ifg_files)} stitched ifgs")

        # make sure we have the water mask
        create_water_mask(self._water_mask_filename, self._dem_bbox)
        unwrapped_files = self._unwrap_ifgs(stitched_ifg_files, cor_files)

        return unwrapped_files
