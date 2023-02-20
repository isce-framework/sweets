import argparse
from pathlib import Path
from typing import Tuple

import dask.array as da
import h5py
import numpy as np
from dask.distributed import Client
from dolphin import utils
from numpy.typing import ArrayLike

from ._log import get_log, log_runtime
from ._types import Filename

logger = get_log(__name__)


def create_ifg(
    dset1: ArrayLike, dset2: ArrayLike, looks: Tuple[int, int], outfile: Filename
):
    """Create a normalized ifg from two flat binary files using dask.

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

    """
    if Path(outfile).suffix != ".h5":
        raise ValueError("Output file must be a .h5 file")

    da1 = da.from_array(dset1)
    da2 = da.from_array(dset2)

    numer = utils.take_looks(da1 * da2.conj(), *looks, func_type="mean")

    pow1 = utils.take_looks((da1 * da1.conj()).real, *looks, func_type="mean")
    pow2 = utils.take_looks((da2 * da2.conj()).real, *looks, func_type="mean")
    denom = np.sqrt(pow1 * pow2)
    ifg = numer / denom

    ifg.to_hdf5(outfile, "ifg")


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
        processes=False,
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
