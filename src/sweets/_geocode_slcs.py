from os import fspath
from pathlib import Path
from typing import List, Optional, Tuple

from compass import s1_geocode_slc, s1_geocode_stack
from compass.utils.geo_runconfig import GeoRunConfig

from ._log import get_log
from ._types import Filename

logger = get_log(__name__)

# TODO: do i ever care to change these?
X_SPAC = 5
Y_SPAC = 10
POL = "co-pol"


def run_geocode(run_config_path: Filename) -> Path:
    """Run a single geocoding workflow on an SLC.

    Parameters
    ----------
    run_config_path : Filename
        Path to the run config file.

    Returns
    -------
    List[Path]
        Paths of geocoded HDF5 files.
    """
    # Need to load the config to get the output paths
    cfg = GeoRunConfig.load_from_yaml(run_config_path, "s1_cslc_geo")
    # Check if it's already been run
    outfile = Path(list(cfg.output_paths.values())[0].hdf5_path)
    if not outfile.exists():
        logger.info(f"Running geocoding for {run_config_path}")
        s1_geocode_slc.run(cfg)
    else:
        logger.info(f"Skipping geocoding for {run_config_path}, {outfile} exists.")

    return outfile


def create_config_files(
    slc_dir: Filename,
    burst_db_file: Filename,
    dem_file: Filename,
    orbit_dir: Filename,
    bbox: Optional[Tuple[float, ...]] = None,
    out_dir: Filename = Path("gslcs"),
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
    out_dir : Filename, optional
        Directory to store geocoded results, by default Path("gslcs")

    Returns
    -------
    List[Path]
        Paths of runconfig files to pass to s1_cslc.py
    """
    s1_geocode_stack.run(
        slc_dir=fspath(slc_dir),
        dem_file=fspath(dem_file),
        orbit_dir=fspath(orbit_dir),
        work_dir=fspath(out_dir),
        burst_db_file=fspath(burst_db_file),
        bbox=bbox,
        pol=POL,
        x_spac=X_SPAC,
        y_spac=Y_SPAC,
    )
    return sorted((Path(out_dir) / "runconfigs").glob("*"))
