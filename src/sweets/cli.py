import argparse
from datetime import datetime
from pathlib import Path


def _get_cli_args():
    parser = argparse.ArgumentParser(
        prog=__package__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser._action_groups.pop()
    aoi = parser.add_argument_group("Required args: specify area of interest")
    aoi.add_argument(
        "-b",
        "--bbox",
        nargs=4,
        metavar=("left", "bottom", "right", "top"),
        type=float,
        help=(
            "Bounding box of area of interest in decimal degrees longitude/latitude: \n"
            "  (e.g. --bbox -106.1 30.1 -103.1 33.1 ). \n"
        ),
    )
    aoi.add_argument(
        "--wkt",
        help=(
            "Alternate to bounding box specification: \nWKT string (or file containing"
            " polygon) for AOI bounds (e.g. from the ASF Vertex tool). \nIf passing "
            " a string polygon, you must enclose in quotes."
        ),
    )
    aoi.add_argument(
        "--geojson",
        type=argparse.FileType(),
        help=(
            "Alternate to bounding box specification: \n"
            "File containing the geojson object for DEM bounds"
        ),
    )
    aoi.add_argument(
        "--start",
        help="Starting date for query (recommended: YYYY-MM-DD)",
        required=True,
    )
    aoi.add_argument(
        "--track",
        type=int,
        required=True,
        help="Limit to one path / relativeOrbit",
    )

    optional = parser.add_argument_group("optional arguments")
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
        default=None,
        help="Maximum temporal baseline (in days) to consider for interferograms.",
    )
    optional.add_argument(
        "--max-bandwidth",
        type=int,
        default=4,
        help="Alternative to temporal baseline: form the nearest n- ifgs.",
    )
    # Allow them to specify the data directory, or the orbit directory
    optional.add_argument(
        "--data-dir",
        help=(
            "Directory to store data in (or directory containing existing downloads)."
            " If None, will store in `data/` "
        ),
    )
    optional.add_argument(
        "--orbit-dir",
        help=(
            "Directory to store orbit files in (or directory containing existing"
            " orbits). If None, will store in `orbits/` "
        ),
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

    # importing below for faster CLI startup
    from sweets.core import Workflow
    from sweets.utils import to_bbox

    if args.bbox is None:
        if args.geojson is not None:
            with open(args.geojson.name, "r") as f:
                args.bbox = to_bbox(geojson=f.read())
            args.geojson = None
        elif args.wkt is not None:
            if Path(args.wkt).exists():
                with open(args.wkt, "r") as f:
                    args.bbox = to_bbox(wkt=f.read())
            else:
                args.bbox = to_bbox(wkt=args.wkt)
        else:
            raise ValueError(
                "Must specify one of --bbox, --wkt, or --geojson for AOI bounds."
            )

    arg_dict = {k: v for k, v in vars(args).items() if v is not None}

    workflow = Workflow(**arg_dict)
    workflow.save(f"sweets_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml")
    workflow.run()
