#!/usr/bin/env python
"""Script for downloading through https://asf.alaska.edu/api/.

Base taken from
https://github.com/scottyhq/isce_notes/blob/master/BatchProcessing.md
https://github.com/scottstanie/apertools/blob/master/apertools/asfdownload.py


You need a .netrc to download:

# cat ~/.netrc
machine urs.earthdata.nasa.gov
    login CHANGE
    password CHANGE

"""
import argparse
import json
import os
import subprocess
import sys
import zipfile
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, List, Optional, Tuple
from urllib.parse import urlencode

import requests
from dateutil.parser import parse
from dolphin.workflows.config import YamlModel
from pydantic import Extra, Field, PrivateAttr, root_validator, validator
from shapely import wkt
from shapely.geometry import box

from ._log import get_log, log_runtime
from ._types import Filename
from ._unzip import unzip_all

logger = get_log(__name__)

DIRNAME = os.path.dirname(os.path.abspath(__file__))


class ASFQuery(YamlModel):
    """Class holding the Sentinel-1 ASF query parameters."""

    out_dir: Path = Field(
        Path(".") / "data",
        description="Output directory for downloaded files",
    )
    bbox: tuple = Field(
        None,
        description=(
            "lower left lon, lat, upper right format e.g."
            " bbox=(-150.2,65.0,-150.1,65.5)"
        ),
    )
    wkt: Optional[str] = Field(
        None,
        description="Well Known Text (WKT) string",
    )
    wkt_file: Optional[Path] = Field(
        None,
        description="Well Known Text (WKT) file",
    )
    start: datetime = Field(
        None,
        description=(
            "Starting time for search. Can be datetime or string (goes to"
            " `dateutil.parse`)"
        ),
    )
    end: datetime = Field(
        default_factory=datetime.now,
        description=(
            "Ending time for search. Can be datetime or string (goes to"
            " `dateutil.parse`)"
        ),
    )
    track: Optional[int] = Field(
        None,
        alias="relativeOrbit",
        description="Path number",
    )
    flight_direction: Optional[str] = Field(
        None,
        alias="flightDirection",
        choices=["ASCENDING", "DESCENDING"],
        description="Ascending or descending",
    )
    frames: Optional[Tuple[int, int]] = Field(
        None,
        description="(start, end) range of ASF frames.",
    )
    unzip: bool = Field(
        False,
        description="Unzip downloaded files into .SAFE directories",
    )
    _url: str = PrivateAttr()

    class Config:
        extra = Extra.forbid  # raise error if extra fields passed in

    @validator("start", "end", pre=True)
    def _parse_date(cls, v):
        if isinstance(v, datetime):
            return v
        elif isinstance(v, date):
            # Convert to datetime
            return datetime.combine(v, datetime.min.time())
        return parse(v)

    @validator("out_dir", always=True)
    def _is_absolute(cls, v):
        return Path(v).resolve()

    @validator("flight_direction")
    def _accept_prefixes(cls, v):
        if v is None:
            return v
        if v.lower().startswith("a"):
            return "ASCENDING"
        elif v.lower().startswith("d"):
            return "DESCENDING"

    @root_validator
    def _check_search_area(cls, values):
        if not values.get("wkt"):
            if values.get("bbox") is not None:
                values["wkt"] = box(*values["bbox"]).wkt
            elif values.get("wkt_file") is not None:
                with open(values["wkt_file"]) as f:
                    values["wkt"] = wkt.load(f)
            else:
                raise ValueError("Must provide a bbox or wkt")

        # Check that end is after start
        if values.get("start") is not None and values.get("end") is not None:
            if values["end"] < values["start"]:
                raise ValueError("End must be after start")
        return values

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        # Form the url for the ASF query.
        self._url = self._form_url()

    def _form_url(self) -> str:
        """Form the url for the ASF query."""
        frame_str = f"{self.frames[0]}-{self.frames[1]}" if self.frames else None
        params = dict(
            # bbox is getting deprecated in favor of intersectsWith
            # https://docs.asf.alaska.edu/api/keywords/#geospatial-parameters
            intersectsWith=self.wkt,
            start=self.start,
            end=self.end,
            processingLevel="SLC",
            relativeOrbit=self.track,
            flightDirection=self.flight_direction,
            maxResults=2000,
            output="geojson",
            platform="S1",  # Currently only supporting S1 right now
            beamMode="IW",
            frame=frame_str,
        )
        params = {k: v for k, v in params.items() if v is not None}
        base_url = "https://api.daac.asf.alaska.edu/services/search/param?{params}"
        return base_url.format(params=urlencode(params))

    def query_results(self) -> dict:
        """Query the ASF API and save the results to a file."""
        return _query_url(self._url)

    @staticmethod
    def _get_urls(results: dict) -> List[str]:
        return [r["properties"]["url"] for r in results["features"]]

    @staticmethod
    def _file_names(results: dict) -> List[str]:
        return [r["properties"]["fileName"] for r in results["features"]]

    def _download_with_aria(self, urls, log_dir: Filename = Path(".")):
        url_filename = self.out_dir / "urls.txt"
        with open(self.out_dir / url_filename, "w") as f:
            for u in urls:
                f.write(u + "\n")

        log_filename = Path(log_dir) / "aria2c.log"
        aria_cmd = f'aria2c -i "{url_filename}" -d "{self.out_dir}" --continue=true'
        logger.info("Downloading with aria2c")
        logger.info(aria_cmd)
        with open(log_filename, "w") as f:
            subprocess.run(aria_cmd, shell=True, stdout=f, stderr=f, text=True)

    @log_runtime
    def download(self, log_dir: Filename = Path(".")) -> List[Path]:
        # Start by saving data available as geojson
        results = self.query_results()
        urls = self._get_urls(results)

        if not urls:
            raise ValueError("No results found for query")

        # Make the output directory
        logger.info(f"Saving to {self.out_dir}")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        file_names = [self.out_dir / f for f in self._file_names(results)]

        # NOTE: aria should skip already-downloaded files
        self._download_with_aria(urls, log_dir=log_dir)

        if self.unzip:
            # Change to .SAFE extension
            logger.info("Unzipping files...")
            file_names = unzip_all(self.out_dir, out_dir=self.out_dir)
        return file_names


@lru_cache(maxsize=10)
def _query_url(url: str) -> dict:
    """Query the ASF API and save the results to a file."""
    logger.info("Querying url:")
    print(url, file=sys.stderr)
    resp = requests.get(url)
    resp.raise_for_status()
    results = json.loads(resp.content.decode("utf-8"))
    return results


def cli():
    """Run the command line interface."""
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out-dir",
        "-o",
        help="Path to directory for saving output files (default=%(default)s)",
        default="./",
    )
    p.add_argument(
        "--bbox",
        nargs=4,
        metavar=("left", "bottom", "right", "top"),
        type=float,
        help=(
            "Bounding box of area of interest  (e.g. --bbox -106.1 30.1 -103.1 33.1 ). "
        ),
    )
    p.add_argument(
        "--wkt-file",
        help="Filename of a WKT polygon to search within",
    )
    p.add_argument(
        "--start",
        help="Starting date for query (recommended: YYYY-MM-DD)",
    )
    p.add_argument(
        "--end",
        help="Ending date for query (recommended: YYYY-MM-DD)",
    )
    p.add_argument(
        "--relativeOrbit",
        type=int,
        help="Limit to one path / relativeOrbit",
    )
    p.add_argument(
        "--flightDirection",
        type=str.upper,
        help="Satellite orbit direction during acquisition",
        choices=["A", "D", "ASCENDING", "DESCENDING"],
    )
    p.add_argument(
        "--maxResults",
        type=int,
        default=2000,
        help="Limit of number of products to download (default=%(default)s)",
    )
    p.add_argument(
        "--query-only",
        action="store_true",
        help="display available data in format of --query-file, no download",
    )
    args = p.parse_args()

    q = ASFQuery(**vars(args))
    if args.query_only:
        q.query_only()
    else:
        q.download_data()


def _unzip_one(filepath: Filename, pol: str = "vv", out_dir=Path(".")):
    """Unzip one Sentinel-1 zip file."""
    if pol is None:
        pol = ""
    with zipfile.ZipFile(filepath, "r") as zipref:
        # Get the list of files in the zip
        names_to_extract = [
            fp for fp in zipref.namelist() if pol.lower() in str(fp).lower()
        ]
        zipref.extractall(path=out_dir, members=names_to_extract)


def delete_tiffs_within_zip(data_path: Filename, pol: str = "vh"):
    """Delete (in place) the tiff files within a zip file matching `pol`."""
    cmd = f"""find {data_path} -name "S*.zip" | xargs -I{{}} -n1 -P4 zip -d {{}} '*-vh-*.tiff'"""  # noqa
    logger.info(cmd)
    subprocess.run(cmd, shell=True, check=True)


if __name__ == "__main__":
    cli()
