"""Blockwise multilooked interferogram and coherence computation.

Reads geocoded SLCs (COMPASS HDF5 with ``/data/VV`` subdataset) in
azimuth blocks, cross-multiplies, multilooks with boxcar or Gaussian
filtering, and writes complex interferogram and coherence GeoTIFFs.

Saving the complex interferogram (not wrapped phase) is deliberate: when
burst outputs are later stitched, GDAL interpolates real and imaginary
parts independently, which is correct.  Wrapped phase is derived *after*
stitching so that interpolation never crosses a +/-pi discontinuity.

The implementation uses NumPy / SciPy with a JAX-accelerated path for
Gaussian multilooking when JAX is available.  No ISCE3 signal dependency.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Literal

import numpy as np
from loguru import logger
from pydantic import BaseModel, Field
from scipy.ndimage import gaussian_filter

try:
    import jax  # noqa: F401

    _JAX_AVAILABLE = True
except ImportError:
    _JAX_AVAILABLE = False

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


def _gaussian_kernel_1d(sigma: float, truncate: float = 4.0) -> np.ndarray:
    """Build a normalized 1-D Gaussian kernel matching scipy's gaussian_filter."""
    radius = int(truncate * sigma + 0.5)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return (k / k.sum()).astype(np.float32)


@functools.lru_cache(maxsize=16)
def _build_gaussian_conv_fn(
    sigma_az: float, sigma_rg: float, az_looks: int, rg_looks: int
):
    """Return a JIT-compiled separable strided Gaussian conv function.

    Applies the Gaussian kernel only at output sample positions (strided
    conv) rather than filtering the full-resolution array and then
    decimating.  For (10, 40) looks this is ~50x fewer FLOPs than the
    scipy filter-then-decimate approach.

    The output shape is exactly ``(H // az_looks, W // rg_looks)``; input
    is clipped to a multiple of the stride before convolving.
    """
    import jax
    import jax.numpy as jnp
    from jax import lax as jax_lax

    k_az = jnp.array(_gaussian_kernel_1d(sigma_az))
    k_rg = jnp.array(_gaussian_kernel_1d(sigma_rg))
    pad_az = len(k_az) // 2
    pad_rg = len(k_rg) // 2

    @jax.jit
    def _conv(arr: jax.Array) -> jax.Array:
        # arr: (H, W) real float32
        # Clip to multiples of stride so output is exactly H//az x W//rg.
        H, W = arr.shape
        x = arr[: (H // az_looks) * az_looks, : (W // rg_looks) * rg_looks]
        x = x[None, None, :, :]  # (1, 1, H_c, W_c)

        # Range pass: horizontal strided conv -> (1, 1, H_c, W_c//rg_looks)
        x = jax_lax.conv_general_dilated(
            x,
            k_rg[None, None, None, :],
            window_strides=(1, rg_looks),
            padding=((0, 0), (pad_rg, pad_rg)),
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
        )
        # Azimuth pass: vertical strided conv -> (1, 1, H_c//az_looks, W_c//rg_looks)
        x = jax_lax.conv_general_dilated(
            x,
            k_az[None, None, :, None],
            window_strides=(az_looks, 1),
            padding=((pad_az, pad_az), (0, 0)),
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
        )
        return x[0, 0, :, :]

    return _conv


def _gaussian_multilook(arr: np.ndarray, looks: tuple[int, int]) -> np.ndarray:
    """Gaussian multilook: separable strided conv (JAX) or filter+decimate (scipy).

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
    sigma_az = az_looks / 2.0
    sigma_rg = rg_looks / 2.0

    if _JAX_AVAILABLE:
        import jax.numpy as jnp

        conv_fn = _build_gaussian_conv_fn(sigma_az, sigma_rg, az_looks, rg_looks)
        if np.iscomplexobj(arr):
            r = np.array(conv_fn(jnp.array(arr.real.astype(np.float32))))
            i = np.array(conv_fn(jnp.array(arr.imag.astype(np.float32))))
            return (r + 1j * i).astype(arr.dtype)
        return np.array(conv_fn(jnp.array(arr.astype(np.float32)))).astype(arr.dtype)

    # scipy fallback when JAX is unavailable.
    # Clip to multiples of stride first so output shape == (H//az, W//rg),
    # matching the JAX path and the write_block row allocation.
    H, W = arr.shape
    arr = arr[: (H // az_looks) * az_looks, : (W // rg_looks) * rg_looks]
    sigma = (sigma_az, sigma_rg)
    if np.iscomplexobj(arr):
        filtered_r = gaussian_filter(arr.real.astype(np.float64), sigma)
        filtered_i = gaussian_filter(arr.imag.astype(np.float64), sigma)
        return ((filtered_r + 1j * filtered_i).astype(arr.dtype))[
            ::az_looks, ::rg_looks
        ]
    return gaussian_filter(arr.astype(np.float64), sigma).astype(arr.dtype)[
        ::az_looks, ::rg_looks
    ]


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

    GDAL geotransform origins are upper-left pixel *corners*, so the output
    corner stays at ``(x0, y0)``; only the pixel spacing scales.
    """
    x0, dx, _, y0, _, dy = input_gt
    return (x0, dx * rg_looks, 0.0, y0, 0.0, dy * az_looks)


def run_crossmul(
    ref_file: Filename,
    sec_file: Filename,
    pair_dir: Path,
    *,
    date1: str,
    date2: str,
    options: CrossmulOptions | None = None,
    subdataset: str = "/data/VV",
    overwrite: bool = False,
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
    ifg_path, coherence_path : tuple[Path, Path]
        Complex interferogram (complex64) and coherence (float32) GeoTIFFs.
        Wrapped phase is *not* saved here; derive it with ``np.angle()``
        after any stitching step so interpolation never crosses a +/-pi
        discontinuity.

    """
    if options is None:
        options = CrossmulOptions()
    pair_dir.mkdir(parents=True, exist_ok=True)

    ifg_path = pair_dir / f"{date1}_{date2}_ifg.tif"
    coh_path = pair_dir / f"{date1}_{date2}_coherence.tif"

    if ifg_path.exists() and coh_path.exists() and not overwrite:
        logger.info(f"Pair {date1}_{date2} already exists; skipping crossmul.")
        return ifg_path, coh_path

    ref_fmt = format_nc_filename(str(ref_file), subdataset)
    sec_fmt = format_nc_filename(str(sec_file), subdataset)

    ncols, nrows = get_raster_xysize(ref_fmt)
    gt = get_raster_gt(ref_fmt)
    crs = get_raster_crs(ref_fmt)

    az_looks, rg_looks = options.looks
    out_nrows = nrows // az_looks
    out_ncols = ncols // rg_looks
    out_gt = _output_geotransform(gt, az_looks, rg_looks)

    # Create output files (empty).  Complex nodata = 0+0j; invalid pixels are
    # identified by coherence == 0 rather than a NaN sentinel.
    write_arr(
        arr=None,
        output_name=ifg_path,
        shape=(out_nrows, out_ncols),
        dtype=np.complex64,
        driver="GTiff",
        options=_GTIFF_OPTIONS,
        geotransform=list(out_gt),
        projection=crs,
        nodata=0,
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
        ifg_block = ifg_ml.astype(np.complex64)

        # Zero out pixels where both SLCs had no data
        both_nan = _multilook(
            (ref_nan | sec_nan).astype(np.float32), options.looks, options.filter_type
        )
        nodata_mask = both_nan > 0.5
        ifg_block[nodata_mask] = 0 + 0j
        coh_block[nodata_mask] = np.nan

        n_out = actual_in_rows // az_looks
        write_block(ifg_block[:n_out], ifg_path, row_start=out_row, col_start=0)
        write_block(coh_block[:n_out], coh_path, row_start=out_row, col_start=0)
        out_row += n_out

    logger.info(
        f"Crossmul {date1}/{date2}: {out_nrows}x{out_ncols} pixels"
        f" ({az_looks}x{rg_looks} looks, {options.filter_type})"
    )
    return ifg_path, coh_path
