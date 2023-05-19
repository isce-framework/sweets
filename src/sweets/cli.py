from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _get_cli_args() -> dict:
    parser = argparse.ArgumentParser(
        prog=__package__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers()
    config_parser = subparsers.add_parser(
        "config",
        help="Create a sweets_config.yaml file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    config_parser._action_groups.pop()

    base = config_parser.add_argument_group()
    base.add_argument(
        "--save-empty",
        action="store_true",
        help="Print an empty config file to `outfile",
    )
    base.add_argument(
        "-o",
        "--outfile",
        help=(
            "Path to save the config file to. \nIf not specified, will save to"
            " `sweets_config.yaml` in the current directory."
        ),
        default="sweets_config.yaml",
    )
    base.add_argument(
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
    base.add_argument(
        "--wkt",
        help=(
            "Alternate to bounding box specification: \nWKT string (or file containing"
            " polygon) for AOI bounds (e.g. from the ASF Vertex tool). \nIf passing "
            " a string polygon, you must enclose in quotes."
        ),
    )
    aoi = config_parser.add_argument_group(
        title="asf_query",
        description="Arguments specifying the area of interest for the S1 data query",
    )
    aoi.add_argument(
        "--track",
        "--relativeOrbit",
        dest="relativeOrbit",
        type=int,
        help="Required: the relative orbit/track",
    )
    aoi.add_argument(
        "--start",
        help="Starting date for query (recommended: YYYY-MM-DD)",
    )
    aoi.add_argument(
        "--end",
        help="Ending date for query (recommended: YYYY-MM-DD). Defaults to today.",
    )
    aoi.add_argument(
        "--frames",
        type=int,
        nargs=2,
        metavar=("start_frame", "end_frame"),
        help=(
            "Limit to a range of frames (e.g. --frames 1 10). Frame numbers come from"
            " ASF website"
        ),
    )
    aoi.add_argument(
        "--data-dir",
        help=(
            "Directory to store data in (or directory containing existing downloads)."
            " If None, will store in `data/` "
        ),
    )

    interferogram_options = config_parser.add_argument_group("interferogram_options")
    interferogram_options.add_argument(
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
    interferogram_options.add_argument(
        "-t",
        "--max-temporal-baseline",
        type=int,
        help="Maximum temporal baseline (in days) to consider for interferograms.",
    )
    interferogram_options.add_argument(
        "--max-bandwidth",
        type=int,
        default=4,
        help="Alternative to temporal baseline: form the nearest n- ifgs.",
    )
    base.add_argument(
        "--orbit-dir",
        help=(
            "Directory to store orbit files in (or directory containing existing"
            " orbits). If None, will store in `orbits/` "
        ),
    )
    base.add_argument(
        "-nw",
        "--n-workers",
        type=int,
        default=4,
        help="Number of dask workers (processes) to use for parallel processing.",
    )
    base.add_argument(
        "-tpw",
        "--threads-per-worker",
        type=int,
        default=16,
        help=(
            "For each workers, number of threads to use (e.g. in numpy multithreading)."
        ),
    )
    config_parser.set_defaults(func=create_config)

    # ##########################
    run_parser = subparsers.add_parser(
        "run",
        help="Run the workflow using a sweets_config.yaml file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    run_parser.add_argument(
        "config_file",
        type=Path,
        help=(
            "Path to a pre-existing sweets_config.yaml file. \nIf not specified, a new"
            " config will be created from the command line arguments."
        ),
    )
    run_parser.add_argument(
        "--starting-step",
        type=int,
        default=1,
        help=(
            "If > 1, will skip earlier steps of the workflow. Step: "
            "1. Download RSLC data from ASF. 2. Create GSLCs. "
            "3. Create burst interfereograms. "
            "4. Stitch burst interferograms into one per date. "
            "5. Unwrap. "
        ),
    )

    run_parser.set_defaults(func=run_workflow)

    arg_groups = {}

    args = parser.parse_args()
    arg_dict = vars(args)
    if arg_dict["func"].__name__ == "create_config":
        for group in config_parser._action_groups:
            # skip positional arguments
            if group.title == "positional arguments":
                continue
            group_dict = {
                a.dest: getattr(args, a.dest, None) for a in group._group_actions
            }
            # remove None values
            group_dict = {k: v for k, v in group_dict.items() if v is not None}
            if group.title:
                arg_groups[group.title] = group_dict
            else:
                arg_groups.update(group_dict)  # type: ignore
        arg_groups["func"] = arg_dict["func"]
        return arg_groups
    else:
        return arg_dict


def run_workflow(kwargs: dict):
    """Run the workflow using a sweets_config.yaml file."""
    # importing below for faster CLI startup
    from sweets.core import Workflow

    cfg_file = kwargs["config_file"]
    if not cfg_file.exists():
        raise ValueError(f"Config file {cfg_file} does not exist")

    if "yaml" not in cfg_file.suffix and "yml" not in cfg_file.suffix:
        raise ValueError(f"Config file {cfg_file} is not a yaml file.")

    workflow = Workflow.from_yaml(cfg_file)
    workflow.run(starting_step=kwargs.get("starting_step", 1))


def create_config(kwargs: dict):
    """Create a sweets_config.yaml file from command line arguments."""
    from sweets.core import Workflow

    outfile = kwargs.pop("outfile", None)
    print(f"Creating config file at {outfile}.", file=sys.stderr)
    if kwargs.pop("save_empty", False):
        Workflow.print_yaml_schema(outfile)
    else:
        workflow = Workflow(**kwargs, start_dask=False)
        workflow.to_yaml(outfile)


def main(args=None):
    """Top-level command line interface to the workflows."""
    args = _get_cli_args()
    func = args.pop("func")
    func(args)
