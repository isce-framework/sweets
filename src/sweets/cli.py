import argparse
import os


def _get_cli_args():
    parser = argparse.ArgumentParser(
        prog=__package__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser._action_groups.pop()
    required = parser.add_argument_group("required arguments")
    optional = parser.add_argument_group("optional arguments")
    required.add_argument(
        "-b",
        "--bbox",
        nargs=4,
        metavar=("left", "bottom", "right", "top"),
        type=float,
        help=(
            "Bounding box of area of interest "
            " (e.g. --bbox -106.1 30.1 -103.1 33.1 ). \n"
            "--bbox points to the *edges* of the pixels, \n"
            " following the 'pixel is area' convention as used in gdal. "
        ),
    )
    required.add_argument(
        "--start",
        help="Starting date for query (recommended: YYYY-MM-DD)",
    )
    required.add_argument(
        "--track",
        type=int,
        required=True,
        help="Limit to one path / relativeOrbit",
    )

    optional.add_argument(
        "--end",
        help="Ending date for query (recommended: YYYY-MM-DD). Defaults to today.",
    )
    optional.add_argument(
        "--looks",
        type=int,
        nargs=2,
        metavar=("az_looks", "range_looks"),
        default=[6, 12],
        help=(
            "Number of looks in azimuth (rows) and range (cols) to use for"
            " interferograms (e.g. --looks 6 12). GSLCs are geocoded at 10m x 5m"
            " posting, so default looks of 6x12 are 60m x 60m."
        ),
    )
    optional.add_argument(
        "-t",
        "--max-temporal-baseline",
        type=int,
        default=180,
        help="Maximum temporal baseline (in days) to consider for interferograms.",
    )
    optional.add_argument(
        "-nw",
        "--n-workers",
        type=int,
        default=4,
        help="Number of dask workers (processes) to use for parallel processing.",
    )
    optional.add_argument(
        "-tpw",
        "--threads-per-worker",
        type=int,
        default=16,
        help=(
            "For each workers, number of threads to use (e.g. in numpy multithreading)."
        ),
    )
    return parser.parse_args()


def main(args=None):
    """Top-level command line interface to the workflows."""
    args = _get_cli_args()
    # Note: importing below here so that we can set the number of threads
    # https://docs.dask.org/en/stable/array-best-practices.html#avoid-oversubscribing-threads
    os.environ["OMP_NUM_THREADS"] = str(args.threads_per_worker)
    # Note that setting OMP_NUM_THREADS here to 1, but passing threads_per_worker
    # to the dask Client does not seem to work for COMPASS.
    # It will just use 1 threads.

    from sweets.core import Workflow

    arg_dict = {k: v for k, v in vars(args).items() if v is not None}

    workflow = Workflow(**arg_dict)
    workflow.run()
