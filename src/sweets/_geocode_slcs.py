from __future__ import annotations

import shutil
from os import fspath
from pathlib import Path
from typing import List, Literal, Optional, Tuple

import compass.s1_geocode_slc
import compass.s1_static_layers
import journal
import yaml  # type: ignore[import-untyped]
from compass import s1_geocode_stack
from compass.utils.geo_runconfig import GeoRunConfig

from loguru import logger

from ._types import Filename

ModuleNames = Literal["s1_geocode_slc", "s1_static_layers"]


def run_geocode(run_config_path: Filename, log_dir: Filename = Path(".")) -> Path:
    """Run a single geocoding workflow on an SLC.

    Parameters
    ----------
    run_config_path : Filename
        Path to the run config file.
    log_dir : Filename, default = "."
        Directory to store the log files.
        Log file is named `s1_geocode_slc_{burst_id}_{date}.log` within log_dir.

    Returns
    -------
    Path
        Path of geocoded HDF5 file.
    """
    return _run_config(run_config_path, log_dir, "s1_geocode_slc")


def run_static_layers(run_config_path: Filename, log_dir: Filename = Path(".")) -> Path:
    """Run the geometry (static layer) creation for an SLC.

    Parameters
    ----------
    run_config_path : Filename
        Path to the run config file.
    log_dir : Filename, default = "."
        Directory to store the log files.
        Log file is named `s1_static_layers_{burst_id}_{date}.log` within log_dir.

    Returns
    -------
    Path
        Path of geocoded HDF5 file.
    """
    return _run_config(run_config_path, log_dir, "s1_static_layers")


def _get_cfg_setup(
    run_config_path: Filename,
    module_name: ModuleNames,
) -> tuple[GeoRunConfig, Path, str]:
    # Need to load the config to get the output paths
    cfg = GeoRunConfig.load_from_yaml(str(run_config_path), "s1_cslc_geo")
    burst_id_tup, params = list(cfg.output_paths.items())[0]
    burst_id_date = "_".join(burst_id_tup)
    outfile = Path(params.hdf5_path)
    if module_name == "s1_static_layers":
        # Static layers are per-burst, not per-date — COMPASS writes them as
        # `static_layers_<burst>.h5`. Strip the trailing date from the name
        # before adding the prefix.
        burst_no_date = outfile.stem.rsplit("_", 1)[0]
        outfile = outfile.with_name(f"static_layers_{burst_no_date}.h5")
    return cfg, outfile, burst_id_date


def _run_config(
    run_config_path: Filename, log_dir: Filename, module_name: ModuleNames
) -> Path:
    cfg, outfile, burst_id_date = _get_cfg_setup(run_config_path, module_name)
    # Check if it's already been run
    if not outfile.exists():
        logger.info(f"Running {module_name} for {run_config_path}")

        # Redirect all isce and compass logs to the same file
        logfile = Path(log_dir) / f"s1_geocode_slc_{burst_id_date}.log"
        journal.info("s1_geocode_slc").device = journal.logfile(fspath(logfile), "w")
        journal.info("isce").device = journal.logfile(fspath(logfile), "w")

        # both s1_geocode_slc and s1_static_layers have the same interface
        module = getattr(compass, module_name)
        module.run(cfg)
    else:
        logger.info(
            f"Skipping workflow for {run_config_path}, {outfile} already exists."
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
    gpu_enabled: bool = True,
    gpu_id: int = 0,
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
    gpu_enabled : bool, optional
        Set ``runconfig.groups.worker.gpu_enabled`` in each emitted
        COMPASS runconfig. ``s1_geocode_stack.run`` does not expose this,
        so sweets patches the dumped YAMLs in-place after they are
        written. Sweets treats this as "use GPU if available": when the
        running isce3 has no CUDA support, the flag is silently
        downgraded to ``False`` before patching, since
        ``isce3.core.gpu_check.use_gpu`` would otherwise raise and abort
        the workflow. Defaults to True.
    gpu_id : int, optional
        Index of the CUDA device for COMPASS to use when
        ``gpu_enabled=True``. Ignored otherwise. Defaults to 0.

    Returns
    -------
    List[Path]
        Paths of runconfig files to pass to s1_cslc.py
    """
    gpu_enabled = _resolve_gpu_enabled(gpu_enabled)

    runconfig_path = Path(out_dir) / "runconfigs"
    if overwrite:
        shutil.rmtree(runconfig_path, ignore_errors=True)

    runconfig_path.mkdir(parents=True, exist_ok=True)
    config_files = sorted(runconfig_path.glob("*"))

    if len(config_files) > 0:
        logger.info(f"Found {len(config_files)} geocoding config files.")
        # Re-patch on resume so a previously-written `gpu_enabled: true`
        # doesn't crash a CPU-only environment, and vice versa.
        _patch_worker_settings(config_files, gpu_enabled=gpu_enabled, gpu_id=gpu_id)
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
    written = sorted((Path(out_dir) / "runconfigs").glob("*"))
    _patch_worker_settings(written, gpu_enabled=gpu_enabled, gpu_id=gpu_id)
    return written


def _resolve_gpu_enabled(requested: bool) -> bool:
    """Downgrade ``gpu_enabled=True`` to ``False`` on CPU-only isce3 builds.

    COMPASS calls ``isce3.core.gpu_check.use_gpu(True, ...)`` which raises
    a hard error when ``isce3.cuda`` is not importable, instead of falling
    back to CPU. Mirror the same ``hasattr(isce3, "cuda")`` probe used
    inside ``use_gpu`` so sweets users get "use GPU if available" semantics.
    """
    if not requested:
        return False
    import isce3

    if hasattr(isce3, "cuda"):
        return True
    logger.warning(
        "gpu_enabled=True but this isce3 build has no CUDA support; "
        "falling back to CPU for COMPASS geocoding."
    )
    return False


def _patch_worker_settings(
    runconfig_files: List[Path], *, gpu_enabled: bool, gpu_id: int
) -> None:
    """Overwrite ``runconfig.groups.worker`` in each runconfig YAML.

    ``s1_geocode_stack.run`` writes runconfigs from COMPASS's bundled
    ``defaults/s1_cslc_geo.yaml``, which hard-codes ``gpu_enabled: False``
    and offers no API to override it. We re-open each YAML and rewrite
    the worker section so resumed runs (which only re-read the YAML)
    pick up the same setting.
    """
    for cfg_path in runconfig_files:
        with open(cfg_path) as f:
            doc = yaml.safe_load(f)
        worker = doc["runconfig"]["groups"]["worker"]
        worker["gpu_enabled"] = gpu_enabled
        worker["gpu_id"] = gpu_id
        with open(cfg_path, "w") as f:
            yaml.safe_dump(doc, f, default_flow_style=False)
