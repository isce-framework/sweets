"""Blockwise multilooked interferogram and coherence computation.

Reads geocoded SLCs (COMPASS HDF5 with ``/data/VV`` subdataset) in
azimuth blocks, cross-multiplies, multilooks with boxcar or Gaussian
filtering, and writes wrapped-phase and coherence GeoTIFFs.

The implementation is pure NumPy / SciPy — no JAX or ISCE3 signal
dependency — so it runs on any machine where dolphin is installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from loguru import logger
from pydantic import BaseModel, Field
from scipy.ndimage import gaussian_filter

from dolphin.io import (
    format_nc_filename,
    get_raster_crs,
    get_raster_gt,
    get_raster_xysize,
    load_gdal,
    write_arr,
    write_block,
)

from dolphin._types import Filename

__all__ = ["CrossmulOptions", "run_crossmul"]

FilterType = Literal["boxcar", "gaussian"]

_DEFAULT_LINES_PER_BLOCK = 512
_GTIFF_OPTIONS = ["COMPRESS=DEFLATE", "TILED=YES", "ZLEVEL=6", "BIGTIFF=IF_SAFER"]


class CrossmulOptions(BaseModel, frozen=True):
    """Options for interferogram formation via multilook crossmul.

    Parameters
    ----------
    looks : tuple[int, int]
        Multilook factors ``(azimuth_looks, range_looks)``.  Controls the
        filter window size and output pixel spacing relative to the input SLCs.
    filter_type : {"boxcar", "gaussian"}
        ``"boxcar"`` for uniform averaging; ``"gaussian"`` for Gaussian-
        weighted averaging (sigma = looks/2, GMTSAR convention).
    lines_per_block : int
        Number of *output* azimuth rows written per processing block.
        Controls peak memory: each block requires
        ``lines_per_block * az_looks * ncols * 8`` bytes of input.
    """

    looks: tuple[int, int] = Field(
        default=(10, 40),
        description=(
            "Multilook factors (azimuth, range).  With COMPASS geocoding at"
            " 10 m x 2.5 m, the default (10, 40) gives ~100 m x 100 m output."
        ),
    )
    filter_type: FilterType = Field(
        default="boxcar",
        description=(
            "'boxcar' for uniform spatial averaging (fast reshape+mean),"
            " 'gaussian' for Gaussian-weighted averaging"
            " (sigma = looks / 2, GMTSAR style)."
        ),
    )
    lines_per_block: int = Field(
        default=_DEFAULT_LINES_PER_BLOCK,
        description="Output azimuth rows per processing block.",
        ge=1,
    )


def _boxcar_multilook(arr: np.ndarray, looks: tuple[int, int]) -> np.ndarray:
    """Boxcar multilook via reshape+mean.

    Parameters
    ----------
    arr
        2-D array (real or complex).
    looks
        ``(az_looks, rg_looks)`` window size.

    Returns
    -------
    np.ndarray
        Multilooked array, shape ``(nrows // az_looks, ncols // rg_looks)``.

    """
    az_looks, rg_looks = looks
    nrows, ncols = arr.shape
    nrows_out = nrows // az_looks
    ncols_out = ncols // rg_looks
    arr = arr[: nrows_out * az_looks, : ncols_out * rg_looks]
    return arr.reshape(nrows_out, az_looks, ncols_out, rg_looks).mean(axis=(1, 3))


def _gaussian_multilook(arr: np.ndarray, looks: tuple[int, int]) -> np.ndarray:
    """Gaussian multilook: convolve then decimate.

    Parameters
    ----------
    arr
        2-D array (real or complex).
    looks
        ``(az_looks, rg_looks)`` — controls the Gaussian sigma
        (sigma = looks / 2) and decimation stride.

    Returns
    -------
    np.ndarray
        Decimated array, shape ``(nrows // az_looks, ncols // rg_looks)``.

    """
    az_looks, rg_looks = looks
    sigma = (az_looks / 2.0, rg_looks / 2.0)
    if np.iscomplexobj(arr):
        filtered_r = gaussian_filter(arr.real.astype(np.float64), sigma)
        filtered_i = gaussian_filter(arr.imag.astype(np.float64), sigma)
        filtered = (filtered_r + 1j * filtered_i).astype(arr.dtype)
    else:
        filtered = gaussian_filter(arr.astype(np.float64), sigma).astype(arr.dtype)
    return filtered[::az_looks, ::rg_looks]


def _multilook(
    arr: np.ndarray, looks: tuple[int, int], filter_type: FilterType
) -> np.ndarray:
    if filter_type == "gaussian":
        return _gaussian_multilook(arr, looks)
    return _boxcar_multilook(arr, looks)


def _compute_coherence(
    ifg_ml: np.ndarray,
    ref_power_ml: np.ndarray,
    sec_power_ml: np.ndarray,
) -> np.ndarray:
    """Compute sample coherence magnitude."""
    denom = np.sqrt(ref_power_ml * sec_power_ml)
    # safe_divide: avoid 0/0 in NaN-free regions; NaN propagates correctly
    with np.errstate(invalid="ignore", divide="ignore"):
        coh = np.where(denom > 0, np.abs(ifg_ml) / denom, 0.0).astype(np.float32)
    return np.clip(coh, 0.0, 1.0)


def _output_geotransform(
    input_gt: list[float], az_looks: int, rg_looks: int
) -> tuple[float, ...]:
    """Compute output geotransform after multilooking.

    Shifts the origin by half the extra pixel so the output pixel center
    coincides with the center of the first multilook window.
    """
    x0, dx, _, y0, _, dy = input_gt
    new_dx = dx * rg_looks
    new_dy = dy * az_looks
    # shift origin from corner of first input pixel to center of first output
    new_x0 = x0 + (rg_looks - 1) / 2.0 * dx
    new_y0 = y0 + (az_looks - 1) / 2.0 * dy
    return (new_x0, new_dx, 0.0, new_y0, 0.0, new_dy)


def run_crossmul(
    ref_file: Filename,
    sec_file: Filename,
    pair_dir: Path,
    *,
    date1: str,
    date2: str,
    options: CrossmulOptions | None = None,
    subdataset: str = "/data/VV",
) -> tuple[Path, Path]:
    """Form a multilooked wrapped interferogram and coherence from two GSLCs.

    Parameters
    ----------
    ref_file
        Path to the reference (earlier) geocoded SLC (COMPASS HDF5).
    sec_file
        Path to the secondary (later) geocoded SLC.
    pair_dir
        Output directory for this pair (created if absent).
    date1
        Reference date string ``YYYYMMDD``.
    date2
        Secondary date string ``YYYYMMDD``.
    options
        Multilook and filter settings.  Defaults to ``CrossmulOptions()``.
    subdataset
        HDF5 dataset path within the SLC file.

    Returns
    -------
    phase_path, coherence_path : tuple[Path, Path]
        Wrapped phase (radians, float32) and coherence (float32) GeoTIFFs.

    """
    if options is None:
        options = CrossmulOptions()
    pair_dir.mkdir(parents=True, exist_ok=True)

    phase_path = pair_dir / f"{date1}_{date2}_wrapped_phase.tif"
    coh_path = pair_dir / f"{date1}_{date2}_coherence.tif"

    if phase_path.exists() and coh_path.exists():
        logger.info(f"Pair {date1}_{date2} already exists; skipping crossmul.")
        return phase_path, coh_path

    ref_fmt = format_nc_filename(str(ref_file), subdataset)
    sec_fmt = format_nc_filename(str(sec_file), subdataset)

    ncols, nrows = get_raster_xysize(ref_fmt)
    gt = get_raster_gt(ref_fmt)
    crs = get_raster_crs(ref_fmt)

    az_looks, rg_looks = options.looks
    out_nrows = nrows // az_looks
    out_ncols = ncols // rg_looks
    out_gt = _output_geotransform(gt, az_looks, rg_looks)

    # Create output files (empty)
    write_arr(
        arr=None,
        output_name=phase_path,
        shape=(out_nrows, out_ncols),
        dtype=np.float32,
        driver="GTiff",
        options=_GTIFF_OPTIONS,
        geotransform=list(out_gt),
        projection=crs,
        nodata=float("nan"),
    )
    write_arr(
        arr=None,
        output_name=coh_path,
        shape=(out_nrows, out_ncols),
        dtype=np.float32,
        driver="GTiff",
        options=_GTIFF_OPTIONS,
        geotransform=list(out_gt),
        projection=crs,
        nodata=float("nan"),
    )

    block_in_rows = options.lines_per_block * az_looks
    out_row = 0

    for r0 in range(0, nrows - az_looks + 1, block_in_rows):
        r1 = min(r0 + block_in_rows, (nrows // az_looks) * az_looks)
        actual_in_rows = r1 - r0

        ref_block = load_gdal(ref_fmt, rows=slice(r0, r1)).astype(np.complex64)
        sec_block = load_gdal(sec_fmt, rows=slice(r0, r1)).astype(np.complex64)

        # Zero-fill NaN for crossmul; track valid mask
        ref_nan = ~np.isfinite(ref_block.real)
        sec_nan = ~np.isfinite(sec_block.real)
        ref_block[ref_nan] = 0.0
        sec_block[sec_nan] = 0.0

        ifg = ref_block * np.conj(sec_block)
        ref_power = (ref_block.real**2 + ref_block.imag**2).astype(np.float32)
        sec_power = (sec_block.real**2 + sec_block.imag**2).astype(np.float32)

        ifg_ml = _multilook(ifg, options.looks, options.filter_type)
        ref_ml = _multilook(ref_power, options.looks, options.filter_type)
        sec_ml = _multilook(sec_power, options.looks, options.filter_type)

        coh_block = _compute_coherence(ifg_ml, ref_ml, sec_ml)
        phase_block = np.angle(ifg_ml).astype(np.float32)

        # NaN pixels where both SLCs had no data
        both_nan = _multilook(
            (ref_nan | sec_nan).astype(np.float32), options.looks, options.filter_type
        )
        # If the multilooked "nan fraction" is > 0.5, mark output as NaN
        nodata_mask = both_nan > 0.5
        phase_block[nodata_mask] = np.nan
        coh_block[nodata_mask] = np.nan

        n_out = actual_in_rows // az_looks
        write_block(phase_block[:n_out], phase_path, row_start=out_row, col_start=0)
        write_block(coh_block[:n_out], coh_path, row_start=out_row, col_start=0)
        out_row += n_out

    logger.info(
        f"Crossmul {date1}/{date2}: {out_nrows}x{out_ncols} pixels"
        f" ({az_looks}x{rg_looks} looks, {options.filter_type})"
    )
    return phase_path, coh_path
