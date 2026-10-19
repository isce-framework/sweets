"""Tropospheric correction step using OPERA L4 TROPO-ZENITH products.

Wraps :func:`opera_utils.tropo.create_tropo_corrections_for_stack` (the
high-level workflow on `scottstanie/opera-utils@develop-scott`) and applies
the resulting per-date LOS-projected delays to dolphin's unwrapped phase
outputs.

The flow is:

1. Register an :class:`OperaCslcReader` so opera_utils' SLCReader registry
   knows how to parse OPERA CSLC HDF5s for ``datetime`` / ``bounds``.
2. Call ``create_tropo_corrections_for_stack`` with the workflow's CSLC
   stack, the (already-stitched) DEM and the (already-stitched) local
   incidence angle raster, producing one ``tropo_correction_<dt>.tif`` per
   acquisition referenced to the first date.
3. For each unwrapped interferogram pair from dolphin, look up the two
   tropo files, compute the differential, convert metres of LOS delay to
   radians of phase via ``4 * pi / wavelength``, and subtract from the
   unwrapped phase. Write the corrected raster alongside the originals.

The OPERA CSLC reader is registered lazily the first time
:func:`create_tropo_corrections` runs, so importing this module stays cheap
and does not require the ``opera_utils.tropo`` subpackage to be present.
"""

from __future__ import annotations

import re
import warnings
from datetime import datetime
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import rasterio
import rioxarray as rxr
from loguru import logger
from pydantic import BaseModel, Field
from shapely import wkt as shp_wkt

from ._log import log_runtime

# Sentinel-1 C-band carrier wavelength used by OPERA CSLCs and burst2safe
# SAFEs alike.
S1_WAVELENGTH_M = 0.05546576

# OPERA tropo correction filenames look like
# `tropo_correction_20210606T005125.tif`. The reference (first) date is
# saved as `reference_<dt>.tif`.
_TROPO_FILENAME_RE = re.compile(r"^(?:reference|tropo_correction)_(\d{8}T\d{6})\.tif$")


class TropoOptions(BaseModel):
    """Sweets-side configuration for the optional tropo correction step."""

    enabled: bool = Field(
        default=False,
        description=(
            "Whether to run the OPERA L4 TROPO-ZENITH correction after"
            " dolphin produces unwrapped interferograms."
        ),
    )
    height_max: float = Field(
        default=10000.0,
        description="Max DEM height (m) included when cropping the tropo cube.",
    )
    margin_deg: float = Field(
        default=0.3,
        description="Padding (degrees) added around the AOI when cropping.",
    )
    interp_method: str = Field(
        default="linear",
        description="Interpolation method passed through to apply_tropo.",
    )
    num_workers: int = Field(
        default=2,
        ge=1,
        description="Parallel workers for crop_tropo / apply_tropo.",
    )


# ----------------------------------------------------------------------------
# OPERA CSLC SLCReader implementation
# ----------------------------------------------------------------------------


class OperaCslcReader:
    """SLCReader implementation for OPERA CSLC HDF5 files.

    The CSLC HDF5 layout (as written by COMPASS / OPERA) puts identification
    metadata under `/identification/` — `zero_doppler_start_time`,
    `bounding_polygon` and the per-burst incidence angles live in the static
    layers file rather than the per-date CSLC, but for the tropo workflow we
    only need the datetime and bounds, both of which are in `/identification`.
    """

    @staticmethod
    def _identification(slc_file: Path) -> dict[str, object]:
        with h5py.File(slc_file, "r") as hf:
            ident = hf["/identification"]
            return {k: ident[k][()] for k in ident.keys()}

    def read_datetime(self, slc_file: Path) -> datetime:
        ident = self._identification(slc_file)
        raw = ident["zero_doppler_start_time"]
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8")
        # OPERA's HDF5 attribute is `YYYY-MM-DD HH:MM:SS.ffffff`
        return datetime.fromisoformat(str(raw).strip())

    def read_bounds(self, slc_file: Path) -> tuple[float, float, float, float]:
        ident = self._identification(slc_file)
        raw = ident["bounding_polygon"]
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8")
        poly = shp_wkt.loads(str(raw))
        west, south, east, north = poly.bounds
        return (west, south, east, north)

    def read_incidence_angle(self, slc_file: Path) -> float:
        # The per-burst incidence is in the static_layers file, not the
        # per-date CSLC. The tropo workflow always passes a separate
        # `incidence_angle` raster (the stitched local_incidence_angle.tif),
        # so this method is intentionally a no-op stub — extract_stack_info
        # never calls it.
        msg = (
            "OperaCslcReader does not provide per-file incidence angle;"
            " pass `incidence_angle_path` to apply_tropo / the workflow."
        )
        raise NotImplementedError(msg)


def _register_opera_cslc_reader() -> None:
    """Idempotently register the OPERA CSLC reader with opera_utils.tropo."""
    from opera_utils.tropo._slc_stack import _sensor_registry, register_sensor

    for name in ("opera-cslc", "sentinel1"):
        if name not in _sensor_registry:
            register_sensor(name, OperaCslcReader())


def _force_threaded_dns_resolver() -> None:
    """Sidestep aiohttp's c-ares resolver, which DNS-times out on some networks.

    Both burst2safe and the OPERA tropo `crop_tropo` open HTTPS URLs via
    aiohttp under the hood; aiohttp's default ``AsyncResolver`` (aiodns)
    fails with ``Timeout while contacting DNS servers`` on networks where
    c-ares can't reach a usable resolver. Falling back to the stdlib
    threaded resolver fixes it without changing behavior elsewhere.
    """
    import aiohttp.resolver

    aiohttp.resolver.DefaultResolver = aiohttp.resolver.ThreadedResolver


_force_threaded_dns_resolver()


# ----------------------------------------------------------------------------
# Workflow integration
# ----------------------------------------------------------------------------


@log_runtime
def create_tropo_corrections(
    slc_files: list[Path],
    dem_path: Path,
    incidence_angle_path: Path,
    output_dir: Path,
    options: TropoOptions | None = None,
) -> list[Path]:
    """Run the OPERA tropo workflow over a stack of SLC files.

    Parameters
    ----------
    slc_files
        Paths to the per-date CSLC HDF5s (or COMPASS-produced GSLC HDF5s).
    dem_path
        DEM raster used by apply_tropo to interpolate to surface heights.
    incidence_angle_path
        Per-pixel incidence-angle raster (produced by stitch_geometry).
    output_dir
        Directory where ``tropo_correction_<dt>.tif`` files will be written.
    options
        Sweets-side knobs (defaults if None).

    Returns
    -------
    list[Path]
        Sorted paths to the produced tropo correction GeoTIFFs.

    """
    from opera_utils.tropo import create_tropo_corrections_for_stack

    _register_opera_cslc_reader()

    options = options or TropoOptions()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    return create_tropo_corrections_for_stack(
        slc_files=list(slc_files),
        dem_path=Path(dem_path),
        output_dir=output_dir,
        sensor="opera-cslc",
        incidence_angle=Path(incidence_angle_path),
        margin_deg=options.margin_deg,
        height_max=options.height_max,
        subtract_first_date=True,
        num_workers=options.num_workers,
    )


def _parse_tropo_filename(p: Path) -> datetime | None:
    """Return the datetime encoded in a tropo correction filename."""
    m = _TROPO_FILENAME_RE.match(p.name)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y%m%dT%H%M%S")


def _group_tropo_files_by_date(tropo_files: list[Path]) -> dict[str, list[Path]]:
    """Bucket per-burst tropo correction rasters under their YYYYMMDD.

    `opera_utils.tropo.create_tropo_corrections_for_stack` writes one
    correction GeoTIFF per CSLC sensing time — interpolated from the
    6-hourly OPERA L4 TROPO-ZENITH products down to the exact time.
    For an OPERA burst stack that gives ~9 files per date (one per
    burst), each ~1-3 seconds apart. Dolphin collapses these back to
    a single stitched raster per date pair, so on the sweets side we
    average the per-burst tropo rasters into a single "per-date"
    correction before applying.
    """
    out: dict[str, list[Path]] = {}
    for p in tropo_files:
        dt = _parse_tropo_filename(p)
        if dt is None:
            continue
        out.setdefault(dt.strftime("%Y%m%d"), []).append(p)
    return out


def _mean_tropo_on_grid(tropo_files: list[Path], target) -> np.ndarray:
    """Reproject a list of tropo rasters onto `target` and return the mean.

    `target` is an xarray DataArray that carries the desired grid + CRS
    (usually the dolphin interferogram we're about to correct). Pixels
    where any input is nodata are carried as NaN in the mean.
    """
    stack: list[np.ndarray] = []
    for p in tropo_files:
        da = rxr.open_rasterio(p, masked=True).squeeze(drop=True)
        matched = da.rio.reproject_match(target)
        stack.append(np.asarray(matched.values, dtype=np.float32))
    return np.nanmean(np.stack(stack, axis=0), axis=0)


def _ifg_dates(path: Path) -> tuple[str, str] | None:
    """Pull the (date1, date2) pair out of a `<date1>_<date2>*.tif` name."""
    m = re.match(r"(\d{8})_(\d{8})", path.name)
    if not m:
        return None
    return m.group(1), m.group(2)


TargetUnits = Literal["radians", "meters"]


def _apply_one_pair(
    target_path: Path,
    tropo_by_date: dict[str, list[Path]],
    output_path: Path,
    scale: float,
) -> Path | None:
    """Subtract the per-date differential tropo from one raster.

    ``scale`` is applied to the metres-of-LOS-delay difference before
    subtraction. Use ``4*pi/wavelength`` for radians-of-phase targets
    (unwrapped ifgs) and ``1.0`` for metres-of-displacement targets
    (dolphin timeseries).
    """
    pair = _ifg_dates(target_path)
    if pair is None:
        logger.warning(f"Skipping {target_path.name}: no date pair in filename")
        return None
    d1, d2 = pair
    if d1 not in tropo_by_date or d2 not in tropo_by_date:
        missing = d1 if d1 not in tropo_by_date else d2
        logger.warning(f"Skipping {target_path.name}: no tropo for {missing}")
        return None

    with rasterio.open(target_path) as src:
        arr = src.read(1)
        profile = src.profile.copy()

    target = rxr.open_rasterio(target_path, masked=True).squeeze(drop=True)
    # `nanmean` warns on all-NaN slices for pixels the tropo grid doesn't
    # cover; the resulting NaN is what we want, so the warning is noise.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Mean of empty slice")
        t1_mean = _mean_tropo_on_grid(tropo_by_date[d1], target)
        t2_mean = _mean_tropo_on_grid(tropo_by_date[d2], target)
    diff_m = t2_mean - t1_mean

    corrected = arr.astype(np.float32) - (scale * diff_m).astype(np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile.update(dtype="float32", count=1, compress="deflate")
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(corrected, 1)
    logger.info(f"Wrote {output_path.name}")
    return output_path


@log_runtime
def apply_tropo_to_pairs(
    target_files: list[Path],
    tropo_files: list[Path],
    output_dir: Path,
    units: TargetUnits,
    suffix: str = ".tropo_corrected.tif",
    wavelength: float = S1_WAVELENGTH_M,
) -> list[Path]:
    """Subtract differential tropo from each date-pair raster.

    For each pair (date1, date2):

        corrected = original - scale * (mean(tropo[d2]) - mean(tropo[d1]))

    where ``scale`` depends on ``units``:

    - ``radians``: ``4*pi/wavelength`` — converts metres of LOS delay
      to radians of phase for unwrapped interferograms.
    - ``meters``: ``1.0`` — the dolphin timeseries rasters are already in
      metres of LOS displacement after the radians-to-metres conversion
      driven by the config wavelength.

    ``mean(tropo[d])`` is the per-burst average of tropo correction rasters
    tagged with date ``d`` (see :func:`_group_tropo_files_by_date`): an
    OPERA burst stack produces ~9 correction rasters per date (one per
    burst sensing time), which we average down before applying so the
    result matches dolphin's per-date-pair stitched output grid.

    Parameters
    ----------
    target_files
        Rasters to correct. Expected to be named ``<date1>_<date2>*.tif``.
    tropo_files
        Paths to ``tropo_correction_<dt>.tif`` rasters produced by
        :func:`create_tropo_corrections`.
    output_dir
        Where corrected rasters will be written.
    units
        Units of ``target_files`` — either ``"radians"`` (phase) or
        ``"meters"`` (displacement).
    suffix
        Suffix appended to ``<date1>_<date2>`` to form the output name.
    wavelength
        Radar carrier wavelength in metres. Defaults to S1 C-band.

    Returns
    -------
    list[Path]
        Paths to the written corrected rasters.

    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tropo_by_date = _group_tropo_files_by_date(list(tropo_files))
    if not tropo_by_date:
        msg = f"No tropo correction files found in {tropo_files!r}"
        raise ValueError(msg)

    scale = (4.0 * np.pi / wavelength) if units == "radians" else 1.0
    written: list[Path] = []
    for src_path in sorted(target_files):
        pair = _ifg_dates(src_path)
        if pair is None:
            continue
        out_path = output_dir / f"{pair[0]}_{pair[1]}{suffix}"
        result = _apply_one_pair(src_path, tropo_by_date, out_path, scale)
        if result is not None:
            written.append(result)
    return written


@log_runtime
def run_tropo_correction(
    slc_files: list[Path],
    dem_path: Path,
    incidence_angle_path: Path,
    dolphin_work_dir: Path,
    options: TropoOptions | None = None,
    wavelength: float = S1_WAVELENGTH_M,
) -> list[Path]:
    """Build tropo corrections + apply to dolphin's unwrapped and timeseries.

    Runs :func:`create_tropo_corrections` to produce the per-CSLC tropo
    rasters, then applies the differential correction to every
    ``<date1>_<date2>*.tif`` raster it finds under ``dolphin_work_dir``:

    - ``unwrapped/<pair>.unw.tif`` (the unwrapped interferogram)
    - ``timeseries/<pair>.tif`` (the post-inversion per-pair timeseries
      raster, when dolphin has produced one — for single-pair stacks
      this is just the unwrapped pair referenced to a point)

    Corrected outputs are written to ``<dolphin_work_dir>/tropo_corrected/``
    with filenames ``<pair>.unw.tropo_corrected.tif`` and
    ``<pair>.timeseries.tropo_corrected.tif`` respectively.
    """
    options = options or TropoOptions()
    dolphin_work_dir = Path(dolphin_work_dir).resolve()
    tropo_corr_dir = dolphin_work_dir / "tropo"
    corrected_dir = dolphin_work_dir / "tropo_corrected"

    tropo_files = create_tropo_corrections(
        slc_files=slc_files,
        dem_path=dem_path,
        incidence_angle_path=incidence_angle_path,
        output_dir=tropo_corr_dir,
        options=options,
    )

    written: list[Path] = []

    # Apply to unwrapped interferograms (radians of phase).
    unwrapped_dir = dolphin_work_dir / "unwrapped"
    unw_files = sorted(p for p in unwrapped_dir.glob("*.unw.tif"))
    if unw_files:
        written.extend(
            apply_tropo_to_pairs(
                target_files=unw_files,
                tropo_files=tropo_files,
                output_dir=corrected_dir,
                units="radians",
                suffix=".unw.tropo_corrected.tif",
                wavelength=wavelength,
            )
        )

    # Apply to the per-pair timeseries rasters (metres of LOS displacement,
    # post inversion + radians-to-metres conversion via wavelength). Single-
    # pair stacks also get one `<d1>_<d2>.tif` here.
    timeseries_dir = dolphin_work_dir / "timeseries"
    ts_files = sorted(
        p for p in timeseries_dir.glob("[0-9]*_[0-9]*.tif") if _ifg_dates(p) is not None
    )
    if ts_files:
        written.extend(
            apply_tropo_to_pairs(
                target_files=ts_files,
                tropo_files=tropo_files,
                output_dir=corrected_dir,
                units="meters",
                suffix=".timeseries.tropo_corrected.tif",
                wavelength=wavelength,
            )
        )

    if not written:
        logger.warning(
            "Tropo correction: no unwrapped or timeseries pair rasters found"
            f" under {dolphin_work_dir}; nothing applied."
        )
    return written
