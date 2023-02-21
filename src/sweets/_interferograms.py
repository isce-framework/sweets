import argparse
from pathlib import Path
from typing import Tuple

import dask
import dask.array as da
import h5py
import numpy as np
from dask.distributed import Client
from dolphin import utils
from dolphin.interferogram import VRTInterferogram
from dolphin.io import DEFAULT_HDF5_OPTIONS, write_arr
from dolphin.workflows.config import OPERA_DATASET_NAME
from numpy.typing import ArrayLike

from ._log import get_log, log_runtime
from ._types import Filename

logger = get_log(__name__)


def create_ifg(vrt_ifg: VRTInterferogram, looks: Tuple[int, int]) -> Path:
    """Create a multi-looked, normalized interferogram from a VRTInterferogram.

    Parameters
    ----------
    vrt_ifg : VRTInterferogram
        Virtual interferogram from `dolphin` containing single-looked ifg data.
    looks : Tuple[int, int]
        row looks, column looks.

    Returns
    -------
    Path
        Path to Geotiff file containing the multi-looked, normalized interferogram.
    """
    ref_slc = utils._get_path_from_gdal_str(vrt_ifg.ref_slc)
    sec_slc = utils._get_path_from_gdal_str(vrt_ifg.sec_slc)
    # suffix = ".int" if to_binary else ".h5"
    suffix = ".tif"
    outfile = vrt_ifg.path.with_suffix(suffix)
    with h5py.File(ref_slc, "r") as f, h5py.File(sec_slc, "r") as g:
        ref_dset = f[OPERA_DATASET_NAME]
        sec_dset = g[OPERA_DATASET_NAME]
        # PerformanceWarning: Reshaping is producing a large chunk...
        with dask.config.set(**{"array.slicing.split_large_chunks": False}):
            _form_ifg(ref_dset, sec_dset, looks, outfile, ref_filename=vrt_ifg.ref_slc)

    return outfile


def _form_ifg(
    dset1: ArrayLike,
    dset2: ArrayLike,
    looks: Tuple[int, int],
    outfile: Filename,
    ref_filename=None,
):
    """Create a normalized ifg from two SLC files using dask.

    Parameters
    ----------
    dset1 : ArrayLike
        First dataset
    dset2 : ArrayLike
        Second dataset
    looks : Tuple[int, int]
        (row looks, column looks)
    outfile : Filename
        Output file to write to.
        Supported file types are .h5 and .int (binary)
    ref_filename : str, optional
        Reference filename where the geo metadata comes from, by default None

    """
    da1 = da.from_array(dset1)
    da2 = da.from_array(dset2)

    # Phase cross multiply for numerator
    numer = utils.take_looks(da1 * da2.conj(), *looks, func_type="mean")

    # Normalize so absolute value is correlation
    pow1 = utils.take_looks((da1 * da1.conj()).real, *looks, func_type="mean")
    pow2 = utils.take_looks((da2 * da2.conj()).real, *looks, func_type="mean")
    denom = np.sqrt(pow1 * pow2)
    ifg = numer / denom

    # TODO: do I care to save it as ENVI? I don't think so.
    if Path(outfile).suffix == ".tif":
        # Since each multi-looked burst will be small, just load into memory.
        ifg_np = np.array(ifg.astype(np.complex64))
        # with open(outfile, "wb") as f:
        # TODO: alternative is .store with a memmap
        # https://docs.dask.org/en/stable/generated/dask.array.store.html#dask.array.store
        write_arr(
            arr=ifg_np,
            output_name=outfile,
            like_filename=ref_filename,
            strides={"x": looks[1], "y": looks[0]},
            nodata=np.nan,
        )
    else:
        # TODO: saving as HDF5 will be more work to get the projection copied over
        DEFAULT_HDF5_OPTIONS["chunks"] = tuple(DEFAULT_HDF5_OPTIONS["chunks"])
        ifg.to_hdf5(outfile, "ifg", **DEFAULT_HDF5_OPTIONS)


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


def _get_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--slcs", nargs=2, metavar=("ref_slc_file", "sec_slc_file"), required=True
    ),
    parser.add_argument("--dset", default="science/SENTINEL1/CSLC/grids/VV")
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
        dset1 = hf1[args.dset]
        dset2 = hf2[args.dset]
        create_ifg(dset1, dset2, args.looks, outfile=args.outfile)


if __name__ == "__main__":
    main()
