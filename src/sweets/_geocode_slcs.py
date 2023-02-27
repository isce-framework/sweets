import shutil
import subprocess
from os import fspath
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import h5py
import numpy as np
from compass import s1_geocode_slc, s1_geocode_stack
from compass.utils.geo_runconfig import GeoRunConfig

from ._log import get_log
from ._types import Filename

logger = get_log(__name__)

# TODO: do i ever care to change these?
X_SPAC = 5
Y_SPAC = 10
POL = "co-pol"


def run_geocode(run_config_path: Filename, compress: bool = True) -> Path:
    """Run a single geocoding workflow on an SLC.

    Parameters
    ----------
    run_config_path : Filename
        Path to the run config file.
    compress : bool
        If true, will compress the output HDF5 file.

    Returns
    -------
    Path
        Path of geocoded HDF5 file.
    """
    # Need to load the config to get the output paths
    cfg = GeoRunConfig.load_from_yaml(run_config_path, "s1_cslc_geo")
    # Check if it's already been run
    outfile = Path(list(cfg.output_paths.values())[0].hdf5_path)
    if not outfile.exists():
        logger.info(f"Running geocoding for {run_config_path}")
        s1_geocode_slc.run(cfg)
        if compress:
            logger.info(f"Compressing {outfile}...")
            repack_and_compress(outfile)
    else:
        logger.info(
            f"Skipping geocoding for {run_config_path}, {outfile} already exists."
        )

    return outfile


def repack_and_compress(
    slc_file: Filename,
    gzip: int = 4,
    chunks: Sequence[int] = (128, 128),
    outfile: Optional[Filename] = None,
    overwrite: bool = True,
):
    """Chunk output product and compress it."""
    temp_out = str(slc_file).replace(".h5", "_zeroed.h5")
    outfile = outfile or str(slc_file).replace(".h5", "_repacked.h5")

    def zero_mantissa(data, bits_to_keep=10):
        float32_mantissa_bits = 23
        nzero = float32_mantissa_bits - bits_to_keep
        # Make all ones
        allbits = (1 << 32) - 1

        bitmask = (allbits << nzero) & allbits
        dr = data.real.view(np.uint32)
        dr &= bitmask
        di = data.imag.view(np.uint32)
        di &= bitmask
        return data

    logger.debug(f"Copying {slc_file} to {temp_out}, zeroing mantissa")
    shutil.copy(slc_file, temp_out)
    with h5py.File(temp_out, "r+") as hf:
        dset_name = "science/SENTINEL1/CSLC/grids/VV"
        dset = hf[dset_name]
        data = dset[:]

        data = zero_mantissa(data)
        dset[:] = data

    cmd = (
        f"h5repack -f {dset_name}:SHUF -l {dset_name}:CHUNK={chunks[0]}x{chunks[1]} -f"
        f" {dset_name}:GZIP={gzip} {temp_out} {outfile}"
    )
    logger.debug(cmd)

    # h5repack is very chatty with it's logging
    subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL)
    # move back to overwrite
    Path(temp_out).unlink()

    if overwrite:
        shutil.move(fspath(outfile), fspath(slc_file))


def create_config_files(
    slc_dir: Filename,
    burst_db_file: Filename,
    dem_file: Filename,
    orbit_dir: Filename,
    bbox: Optional[Tuple[float, ...]] = None,
    out_dir: Filename = Path("gslcs"),
    overwrite: bool = False,
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
    overwrite : bool, optional
        If true, will overwrite existing config files, by default False

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
        pol=POL,
        x_spac=X_SPAC,
        y_spac=Y_SPAC,
    )
    return sorted((Path(out_dir) / "runconfigs").glob("*"))
