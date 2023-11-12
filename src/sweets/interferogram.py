#!/usr/bin/env python
from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Optional

import dask
import dask.array as da
import h5py
import numpy as np
import rioxarray
from compass.utils.helpers import bbox_to_utm
from dask.distributed import Client
from dolphin import utils
from dolphin.io import DEFAULT_HDF5_OPTIONS, get_raster_xysize, load_gdal, write_arr
from opera_utils import OPERA_DATASET_NAME
from pydantic import BaseModel, Field, model_validator
from rich.progress import track

from ._log import get_log, log_runtime
from ._types import Filename
from .utils import get_intersection_bounds, get_overlapping_bounds

logger = get_log(__name__)


def create_ifg(
    ref_slc: Filename,
    sec_slc: Filename,
    outfile: Filename,
    *,
    looks: tuple[int, int],
    overwrite: bool = False,
    bbox: Optional[tuple[float, float, float, float]] = None,
    overlapping_with: Optional[Path] = None,
) -> Path:
    """Create a multi-looked, normalized interferogram from GDAL-readable SLCs.

    Parameters
    ----------
    ref_slc : Filename
        Path to reference SLC.
    sec_slc : Filename
        Path to secondary SLC.
    outfile : Filename
        Path to output file.
    looks : tuple[int, int]
        row looks, column looks.
    overwrite : bool, optional
        Overwrite existing interferogram in `outfile`, by default False.
    bbox : Optional[tuple[float, float, float, float]], optional
        Bounding box to crop the interferogram to, by default None.
        Assumes (lon_min, lat_min, lon_max, lat_max) format.
    overlapping_with : Optional[Path], optional
        Alternative to bbox: Path to another file to crop to the overlap.
        If the bounding boxes overlap, the output interferogram will be cropped
        to the intersection of the two bounding boxes
        If None, will not check for overlapping bounding boxes, by default None.


    Returns
    -------
    Path
        Path to Geotiff file containing the multi-looked, normalized interferogram.
    """
    outfile = Path(outfile)
    if outfile.exists():
        if not overwrite:
            logger.debug(f"Skipping {outfile} because it already exists.")
            return outfile
        else:
            logger.info(f"Overwriting {outfile} because overwrite=True.")
            outfile.unlink()

    dask_chunks = (1, 128 * 10, 128 * 10)
    da_ref = rioxarray.open_rasterio(ref_slc, chunks=dask_chunks).sel(band=1)
    da_sec = rioxarray.open_rasterio(sec_slc, chunks=dask_chunks).sel(band=1)
    bb_utm = None
    if bbox is not None:
        bb_target = bbox_to_utm(bbox, epsg_src=4326, epsg_dst=da_ref.rio.crs.to_epsg())
        bb_utm = get_overlapping_bounds(bb_target, da_ref.rio.bounds())
    elif overlapping_with:
        bb_utm = get_intersection_bounds(
            overlapping_with, ref_slc, epsg_code=da_ref.rio.crs.to_epsg()
        )
    if bb_utm is not None:
        # (left, bottom, right, top) -> (left, right), (top, bottom)
        da_ref = da_ref.sel(
            x=slice(bb_utm[0], bb_utm[2]), y=slice(bb_utm[3], bb_utm[1])
        )
        da_sec = da_sec.sel(
            x=slice(bb_utm[0], bb_utm[2]), y=slice(bb_utm[3], bb_utm[1])
        )

    logger.info(f"Creating {looks[0]}x{looks[1]} multi-looked interferogram: {outfile}")
    # PerformanceWarning: Reshaping is producing a large chunk...
    with dask.config.set(**{"array.slicing.split_large_chunks": False}):
        # _form_ifg(da_ref, da_sec, looks, outfile, ref_filename=vrt_ifg.ref_slc)
        _form_ifg(da_ref, da_sec, looks, outfile)
    del da_ref
    del da_sec

    return outfile


class InterferogramOptions(BaseModel):
    """Options for creating interferograms in workflow."""

    looks: tuple[int, int] = Field(
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

    @model_validator(mode="after")
    def _check_max_temporal_baseline(self):
        """Make sure they didn't specify max_bandwidth and max_temporal_baseline."""
        max_temporal_baseline = self.max_temporal_baseline
        if max_temporal_baseline is not None:
            self.max_bandwidth = None
            # TODO :use the new field set functions for this
            # raise ValueError(
            #     "Cannot specify both max_bandwidth and max_temporal_baseline"
            # )
        return self


def _take_looks_da(da: da.Array, row_looks: int, col_looks: int):
    return da.coarsen(x=col_looks, y=row_looks, boundary="trim").mean()


def _form_ifg(
    da1: da.Array,
    da2: da.Array,
    looks: tuple[int, int],
    outfile: Filename,
    # ref_filename=None,
):
    """Create a normalized ifg from two SLC files using dask.

    Parameters
    ----------
    da1 : dask.array.Array
        First SLC loaded as a dask array
    da2 : da.array.Array
        Secondary SLC loaded as a dask array
    looks : tuple[int, int]
        (row looks, column looks)
    outfile : Filename
        Output file to write to.
        Supported file types are .tif, .h5 and .int (binary)
    ref_filename : str, optional
        Reference filename where the geo metadata comes from, by default None

    """
    # da1 = da.from_array(da1)
    # da2 = da.from_array(da2)

    # Phase cross multiply for numerator
    # numer = utils.take_looks(da1 * da2.conj(), *looks, func_type="mean")
    numer = _take_looks_da(da1 * da2.conj(), *looks)

    # Normalize so absolute value is correlation
    # pow1 = utils.take_looks((da1 * da1.conj()).real, *looks, func_type="mean")
    # pow2 = utils.take_looks((da2 * da2.conj()).real, *looks, func_type="mean")
    pow1 = _take_looks_da(da1 * da1.conj(), *looks)
    pow2 = _take_looks_da(da2 * da2.conj(), *looks)
    denom = np.sqrt(pow1 * pow2)
    # RuntimeWarning: invalid value encountered in divide
    # filter out warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        ifg = (numer / denom).astype("complex64")

    # TODO: do I care to save it as ENVI? I don't think so.
    suffix = Path(outfile).suffix
    if suffix in (".tif", ".int"):
        # Make sure we didn't lose the geo information
        ifg.rio.write_crs(da1.rio.crs, inplace=True)
        ifg.rio.write_nodata(float("NaN"), inplace=True)

        if suffix == ".tif":
            ifg.rio.to_raster(outfile, tiled=True)
        else:
            ifg.rio.to_raster(outfile, driver="ENVI")

        # Since each multi-looked burst will be small, just load into memory.
        # ifg_np = ifg.compute()
        # # with open(outfile, "wb") as f:
        # # TODO: alternative is .store with a memmap
        # write_arr(
        #     arr=ifg_np,
        #     output_name=outfile,
        #     like_filename=ref_filename,
        #     strides={"x": looks[1], "y": looks[0]},
        #     nodata=np.nan,
        # )
    else:
        # TODO: saving as HDF5 will be more work to get the projection copied over
        DEFAULT_HDF5_OPTIONS["chunks"] = tuple(DEFAULT_HDF5_OPTIONS["chunks"])
        ifg.to_hdf5(outfile, "ifg", **DEFAULT_HDF5_OPTIONS)
    del ifg  # Deleting to tell dask it is done


def _form_ifg_name(slc1: Filename, slc2: Filename, out_dir: Filename) -> Path:
    """Form the name of the interferogram file.

    Parameters
    ----------
    slc1 : Filename
        First SLC
    slc2 : Filename
        Second SLC
    out_dir : Filename
        Output directory

    Returns
    -------
    Path
        Path to the interferogram file.

    """
    date1 = utils.get_dates(slc1)[0]
    date2 = utils.get_dates(slc2)[0]
    fmt = "%Y%m%d"
    ifg_name = f"{date1.strftime(fmt)}_{date2.strftime(fmt)}.h5"
    return Path(out_dir) / ifg_name


def create_cor(ifg_filename: Filename, outfile: Optional[Filename] = None):
    """Write out a binary correlation file for an interferogram.

    Assumes the interferogram has been normalized so that the absolute value
    is the correlation.

    Parameters
    ----------
    ifg_filename : Filename
        Complex interferogram filename
    outfile : Optional[Filename], optional
        Output filename, by default None
        If None, will use the same name as the interferogram but with the
        extension changed to .cor

    Returns
    -------
    Filename
        Output filename
    """
    if outfile is None:
        outfile = Path(ifg_filename).with_suffix(".cor")
    da_ifg = rioxarray.open_rasterio(ifg_filename, chunks=True)
    np.abs(da_ifg).rio.to_raster(outfile, driver="ENVI", suffix="add")
    return outfile


def _get_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--slcs", nargs=2, metavar=("ref_slc_file", "sec_slc_file"), required=True
    )
    parser.add_argument("--dset", default=OPERA_DATASET_NAME)
    parser.add_argument("-l", "--looks", type=int, nargs=2, default=(1, 1))
    parser.add_argument(
        "-o",
        "--outfile",
    )
    parser.add_argument("--n-workers", type=int, default=4)
    args = parser.parse_args()
    if not args.outfile:
        args.outfile = _form_ifg_name(args.slcs[0], args.slcs[1], ".")
        logger.debug(f"Setting outfile to {args.outfile}")
    return args


@log_runtime
def main():
    """Create one interferogram from two SLCs."""
    args = _get_cli_args()
    Client(
        threads_per_worker=4,
        n_workers=args.n_workers,
        memory_limit=f"{args.max_ram_gb}GB",
    )
    with h5py.File(args.slcs[0]) as hf1, h5py.File(args.slcs[1]) as hf2:
        da1 = da.from_array(hf1[args.dset])
        da2 = da.from_array(hf2[args.dset])
        create_ifg(da1, da2, args.looks, outfile=args.outfile)


def get_average_correlation(
    file_path: Filename,
    cor_ext: str = ".cor",
    output_name: Optional[Filename] = None,
    mask_where_incomplete: bool = True,
) -> np.ndarray:
    """Get the average correlation from a directory of correlation files.

    Parameters
    ----------
    file_path : Filename
        Path to the directory containing the correlation files.
    cor_ext : str, optional
        Extension of the correlation files, by default ".cor"
    output_name : str, optional
        Name of the output file.
        If not provided, outputs to "average_correlation.cor.tif" in the
        directory of `file_path`.
    mask_where_incomplete : bool, optional
        If True, will mask out pixels where the correlation is not present in all files.

    Returns
    -------
    np.ndarray
        Average correlation array.
    """
    cor_files = sorted(Path(file_path).glob(f"*{cor_ext}"))
    if not cor_files:
        raise ValueError(f"No files found with {cor_ext} in {file_path}")
    if output_name is None:
        output_name = cor_files[0].parent / "average_correlation.cor.tif"
    count_output_name = Path(output_name).parent / "average_correlation_count.tif"
    if Path(output_name).exists():
        return load_gdal(output_name)

    cols, rows = get_raster_xysize(cor_files[0])
    avg_c = np.zeros((rows, cols), dtype=np.float32)
    count = np.zeros((rows, cols), dtype=np.int32)
    for f in track(cor_files):
        cor = load_gdal(f)
        cor_masked = np.ma.masked_invalid(cor)
        avg_c += cor_masked
        bad_mask = np.logical_or(cor_masked.mask, cor_masked == 0)
        count[~bad_mask] += 1
    avg_c /= len(cor_files)
    if mask_where_incomplete:
        avg_c[count != len(cor_files)] = np.nan
    write_arr(
        arr=avg_c, like_filename=cor_files[0], output_name=output_name, nodata=np.nan
    )
    write_arr(
        arr=count, like_filename=cor_files[0], output_name=count_output_name, nodata=0
    )

    return avg_c


if __name__ == "__main__":
    main()
