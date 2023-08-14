from __future__ import annotations

import shutil
from os import fspath
from pathlib import Path
from typing import List, Optional, Tuple

import journal
from compass import s1_geocode_slc, s1_geocode_stack
from compass.utils.geo_runconfig import GeoRunConfig

from ._log import get_log
from ._types import Filename

logger = get_log(__name__)


def run_geocode(run_config_path: Filename, log_dir: Filename = Path(".")) -> Path:
    """Run a single geocoding workflow on an SLC.

    Parameters
    ----------
    run_config_path : Filename
        Path to the run config file.
    x_posting : float, default = 5
        Spacing of geocoded SLCs (in meters) along the x-direction.
    y_posting : float, default = 10
        Spacing of geocoded SLCs (in meters) along the y-direction.
    log_dir : Filename, default = "."
        Directory to store the log files.
        Log file is named `s1_geocode_slc_{burst_id}_{date}.log` within log_dir.

    Returns
    -------
    Path
        Path of geocoded HDF5 file.
    """
    cfg = GeoRunConfig.load_from_yaml(str(run_config_path), "s1_cslc_geo")

    burst_id_tup, params = list(cfg.output_paths.items())[0]
    # Need to load the config to get the output paths
    # Check if it's already been run
    outfile = Path(params.hdf5_path)
    if not outfile.exists():
        logger.info(f"Running geocoding for {run_config_path}")

        # Redirect all isce and compass logs to the same file
        burst_id_date = "_".join(burst_id_tup)
        logfile = Path(log_dir) / f"s1_geocode_slc_{burst_id_date}.log"
        journal.info("s1_geocode_slc").device = journal.logfile(fspath(logfile), "w")
        journal.info("isce").device = journal.logfile(fspath(logfile), "w")

        s1_geocode_slc.run(cfg)
    else:
        logger.info(
            f"Skipping geocoding for {run_config_path}, {outfile} already exists."
        )

    return outfile


def create_config_files(
    slc_dir: Filename,
    burst_db_file: Filename,
    dem_file: Filename,
    orbit_dir: Filename,
    bbox: Optional[Tuple[float, ...]] = None,
    x_posting: float = 5,
    y_posting: float = 10,
    pol_type: str = "co-pol",
    out_dir: Filename = Path("gslcs"),
    overwrite: bool = False,
    using_zipped: bool = False,
) -> List[Path]:
    """Create the geocoding config files for a stack of SLCs.

    Parameters
    ----------
    slc_dir : Filename
        Directory containing the SLCs as .zip or .SAFE
    burst_db_file : Filename
        File containing the burst database
    dem_file : Filename
        File containing the DEM
    orbit_dir : Filename
        Directory containing the orbit files
    bbox : Optional[Tuple[float, ...]], optional
        [lon_min, lat_min, lon_max, lat_max] to limit the processing.
        Note that this does not change each burst's bounding box, but
        limits which bursts are used (skipped if they don't overlap).
        By default None (process all bursts in zip file for all files)
    x_posting : float, optional
        X posting (in meters) of geocoded output, by default 5 m.
    y_posting : float, optional
        Y posting (in meters) of geocoded output, by default 10 m.
    pol_type : str, choices = {"co-pol", "cross-pol"}
        Type of polarization to process.
    out_dir : Filename, optional
        Directory to store geocoded results, by default Path("gslcs")
    overwrite : bool, optional
        If true, will overwrite existing config files, by default False
    using_zipped : bool, optional
        If true, will search for. zip files instead of unzipped .SAFE directories.
        By default False.

    Returns
    -------
    List[Path]
        Paths of runconfig files to pass to s1_cslc.py
    """
    # Check if they already exist:
    runconfig_path = Path(out_dir) / "runconfigs"
    if overwrite:
        shutil.rmtree(runconfig_path, ignore_errors=True)

    runconfig_path.mkdir(parents=True, exist_ok=True)
    config_files = sorted(runconfig_path.glob("*"))

    if len(config_files) > 0:
        logger.info(f"Found {len(config_files)} geocoding config files.")
        return config_files
    s1_geocode_stack.run(
        slc_dir=fspath(slc_dir),
        dem_file=fspath(dem_file),
        orbit_dir=fspath(orbit_dir),
        work_dir=fspath(out_dir),
        burst_db_file=fspath(burst_db_file),
        bbox=bbox,
        pol=pol_type,
        x_spac=x_posting,
        y_spac=y_posting,
        using_zipped=using_zipped,
    )
    return sorted((Path(out_dir) / "runconfigs").glob("*"))
