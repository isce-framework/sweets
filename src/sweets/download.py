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
from typing import Any, List, Optional
from urllib.parse import urlencode

import requests
from dateutil.parser import parse
from osgeo import gdal
from pydantic import BaseModel, Extra, Field, PrivateAttr, root_validator, validator
from shapely import wkt

from ._log import get_log, log_runtime
from ._types import Filename
from ._unzip import unzip_all

logger = get_log(__name__)

DIRNAME = os.path.dirname(os.path.abspath(__file__))


class ASFQuery(BaseModel):
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
    dem: Optional[str] = Field(
        None,
        description="Name of DEM filename (will parse bbox)",
    )
    wkt_file: Optional[str] = Field(
        None,
        description="Well Known Text (WKT) file",
    )
    start: datetime = Field(
        None,
        description=(
            "Starting time for search. Many acceptable inputs e.g. '3 months and a day"
            " ago' 'May 30, 2018' '2010-10-30T00:00:00Z'"
        ),
    )
    end: datetime = Field(
        None,
        description=(
            "Ending time for search. Many acceptable inputs e.g. '3 months and a day"
            " ago' 'May 30, 2018' '2010-10-30T00:00:00Z'"
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
    unzip: bool = Field(
        True,
        description="Unzip downloaded files into .SAFE directories",
    )
    orbit_dir: Path = Field(
        Path("orbits"),
        description="Directory for orbit files",
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
    def _check_bbox(cls, values):
        if values.get("dem") is not None:
            values["bbox"] = cls._get_dem_bbox(values["dem"])
        elif values.get("wkt_file") is not None:
            values["bbox"] = cls._get_wkt_bbox(values["wkt_file"])
        if values.get("bbox") is None:
            raise ValueError("Must provide a bbox or a dem or a wkt_file")

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
        params = dict(
            bbox=",".join(map(str, self.bbox)) if self.bbox else None,
            start=self.start,
            end=self.end,
            processingLevel="SLC",
            relativeOrbit=self.track,
            flightDirection=self.flight_direction,
            maxResults=2000,
            output="geojson",
            platform="S1",  # Currently only supporting S1 right now
            beamMode="IW",
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

    def _download_with_aria(self, urls):
        url_filename = self.out_dir / "urls.txt"
        with open(self.out_dir / url_filename, "w") as f:
            for u in urls:
                f.write(u + "\n")

        aria_cmd = f'aria2c -i "{url_filename}" -d "{self.out_dir}" --continue=true'
        subprocess.run(aria_cmd, shell=True)

    @log_runtime
    def download(self) -> List[Path]:
        # Start by saving data available as geojson
        results = self.query_results()

        # Make the output directory
        logger.info(f"Saving to {self.out_dir}")
        self.out_dir.mkdir(parents=True, exist_ok=True)

        urls = self._get_urls(results)
        file_names = [self.out_dir / f for f in self._file_names(results)]

        # Exclude already-downloaded files
        done_files = [f for f in file_names if f.exists()]
        to_do_files, to_do_urls = [], []
        for f, u in zip(file_names, urls):
            if f not in done_files:
                to_do_files.append(f)
                to_do_urls.append(u)
        msg = f"Missing {len(to_do_files)}/{len(file_names)} files. "

        if len(to_do_files) == 0:
            msg += "All files already downloaded."
        else:
            msg += "Downloading..."
        logger.info(msg)

        if to_do_urls:
            self._download_with_aria(to_do_urls)

        if self.unzip:
            # Change to .SAFE extension
            logger.info("Unzipping files...")
            file_names = unzip_all(self.out_dir)
        return file_names

    @staticmethod
    def _get_dem_bbox(fname):
        ds = gdal.Open(fname)
        left, xres, _, top, _, yres = ds.GetGeoTransform()
        right = left + (ds.RasterXSize * xres)
        bottom = top + (ds.RasterYSize * yres)
        return left, bottom, right, top

    @staticmethod
    def _get_wkt_bbox(fname):
        with open(fname) as f:
            return wkt.load(f).bounds


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
        "--dem",
        help="Filename of a (gdal-readable) DEM",
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
    if all(vars(args)[item] for item in ("bbox", "dem", "absoluteOrbit", "flightLine")):
        raise ValueError(
            "Need either --bbox or --dem options without flightLine/absoluteOrbit"
        )

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


if __name__ == "__main__":
    cli()
