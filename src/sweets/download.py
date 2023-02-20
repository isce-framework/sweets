#!/usr/bin/env python
"""
Script for downloading through https://asf.alaska.edu/api/

Base taken from
https://github.com/scottyhq/isce_notes/blob/master/BatchProcessing.md
https://github.com/scottstanie/apertools/blob/master/apertools/asfdownload.py


To download, you need aria2
yum install -y aria2

and either a .netrc:

# cat ~/.netrc
machine urs.earthdata.nasa.gov
    login CHANGE
    password CHANGE

or, an aria2 conf file

# $HOME/.aria2/asf.conf
http-user=CHANGE
http-passwd=CHANGE

max-concurrent-downloads=5
check-certificate=false

allow-overwrite=false
auto-file-renaming=false
always-resume=true
"""
from typing import Optional, Any
from pathlib import Path
import argparse
import datetime
import os
import subprocess
from collections import Counter
from osgeo import gdal

from shapely import wkt
from urllib.parse import urlencode

from pydantic import BaseModel, Field, PrivateAttr, root_validator
from datetime import datetime

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
    _url: str = PrivateAttr()
    _outname: str = PrivateAttr()
    _query_cmd: str = PrivateAttr()

    @root_validator
    def check_bbox(cls, values):
        if values.get("dem") is not None:
            values["bbox"] = _get_dem_bbox(values["dem"])
        elif values.get("wkt_file") is not None:
            values["bbox"] = _get_wkt_bbox(values["wkt_file"])
        if values.get("bbox") is None:
            raise ValueError("Must provide a bbox or a dem or a wkt_file")
        return values

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        # Form the url for the ASF query.
        self._url = self.form_url()
        self._outname = "asfquery.geojson"
        self._query_cmd = """ curl "{url}" > {outname} """.format(
            url=self._url, outname=self._outname
        )

    def form_url(self):
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

    def query_only(self):
        """Save files into correct output type."""
        print("Running command:")
        print(self._query_cmd)
        subprocess.check_call(self._query_cmd, shell=True)

    def download_data(self, query_filetype="metalink", out_dir=".", **kwargs):
        # Start by saving data available as a metalink file
        self.query_only(query_filetype=query_filetype, **kwargs)

        aria2_conf = os.path.expanduser("~/.aria2/asf.conf")
        download_cmd = (
            "aria2c --http-auth-challenge=true --continue=true "
            f"--conf-path={aria2_conf} --dir={out_dir} {self._outname}"
        )
        print("Running command:")
        print(download_cmd)
        subprocess.check_call(download_cmd, shell=True)


def _get_dem_bbox(fname):
    ds = gdal.Open(fname)
    left, xres, _, top, _, yres = ds.GetGeoTransform()
    right = left + (ds.RasterXSize * xres)
    bottom = top + (ds.RasterYSize * yres)
    return left, bottom, right, top


def _get_wkt_bbox(fname):
    with open(fname) as f:
        return wkt.load(f).bounds


def _parse_query_results(fname="asfquery.geojson"):
    """Extract the path number counts and date ranges from a geojson query result"""
    import geojson

    with open(fname) as f:
        results = geojson.load(f)

    features = results["features"]
    # In[128]: pprint(results["features"])
    # [{'geometry': {'coordinates': [[[-101.8248, 34.1248],...
    #            'type': 'Polygon'},
    # 'properties': {'beamModeType': 'IW', 'pathNumber': '85',
    # 'type': 'Feature'}, ...
    print(f"Found {len(features)} results:")
    if len(features) == 0:
        return Counter(), []

    # Include both the number and direction (asc/desc) in Counter key
    path_nums = Counter(
        [
            (f["properties"]["pathNumber"], f["properties"]["flightDirection"].lower())
            for f in features
        ]
    )
    print(f"Count by pathNumber: {path_nums.most_common()}")
    starts = Counter([f["properties"]["startTime"] for f in features])
    starts = [datetime.datetime.fromisoformat(s) for s in starts]
    print(f"Dates ranging from {min(starts)} to {max(starts)}")

    return path_nums, starts


def cli():
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


if __name__ == "__main__":
    cli()
