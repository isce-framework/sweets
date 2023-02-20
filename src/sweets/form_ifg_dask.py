from dask.distributed import Client, progress
from rich import print
import h5py
import argparse
import numpy as np
from dolphin import io, utils
import dask.array as da


def make_ifg_dask(dset1, dset2, looks, outfile="ifg.h5"):
    """Create a normalized ifg from two flat binary files using dask."""

    # cols, rows = io.get_raster_xysize(f1)
    # mmap1 = np.memmap(f1, mode="r", dtype="complex64", shape=(rows, cols))
    # mmap2 = np.memmap(f2, mode="r", dtype="complex64", shape=(rows, cols))

    da1 = da.from_array(dset1)
    da2 = da.from_array(dset2)

    numer = utils.take_looks(da1 * da2.conj(), *looks, func_type="mean")

    pow1 = utils.take_looks((da1 * da1.conj()).real, *looks, func_type="mean")
    pow2 = utils.take_looks((da2 * da2.conj()).real, *looks, func_type="mean")
    denom = np.sqrt(pow1 * pow2)
    ifg = numer / denom

    if outfile.endswith(".h5"):
        ifg.to_hdf5(outfile, "ifg")
    elif outfile.endswith(".zarr"):
        if looks != (1, 1):
            # ValueError: Attempt to save array to zarr with irregular chunking,
            # please call `arr.rechunk(...)` first.
            ifg = ifg.rechunk()
        ifg.to_zarr(outfile)


def get_cli_args():
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
    parser.add_argument("--max-ram-gb", type=int, default=8)
    parser.add_argument("--n-workers", type=int, default=4)
    args = parser.parse_args()
    if not args.outfile:
        date1 = utils.get_dates(args.slcs[0])[0]
        date2 = utils.get_dates(args.slcs[1])[0]
        fmt = "%Y%m%d"
        args.outfile = f"{date1.strftime(fmt)}_{date2.strftime(fmt)}.h5"
        print(f"Setting outfile to {args.outfile}")
    return args


def main():
    args = get_cli_args()
    client = Client(
        processes=False,
        threads_per_worker=4,
        n_workers=args.n_workers,
        memory_limit=f"{args.max_ram_gb}GB",
    )
    with h5py.File(args.slcs[0]) as hf1, h5py.File(args.slcs[1]) as hf2:
        dset1 = hf1[args.dset]
        dset2 = hf2[args.dset]
        make_ifg_dask(dset1, dset2, args.looks, outfile=args.outfile)


if __name__ == "__main__":
    main()
