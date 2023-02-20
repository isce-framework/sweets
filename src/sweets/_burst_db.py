import gzip
import json
from pathlib import Path
from typing import Optional, Union

import requests

from .utils import get_cache_dir

URL = "https://github.com/scottstanie/burst_db/raw/test-json-gz/src/burst_db/data/burst_map_bbox_only.json.gz"  # noqa


def download_burst_db(
    url: str = URL, out_file: Optional[Union[str, Path]] = None
) -> Path:
    """Download the burst-db file.

    Parameters
    ----------
    url : str, optional
        URL to download from.
    out_file : Optional[Union[str, Path]], optional
        Location to save locally. If None, will save to
        `get_cache_dir() / "burst_dict.json.gz"`, by default None

    Returns
    -------
    Path
        Locally saved file.

    Raises
    ------
    ValueError
        If the download fails.
    """
    if out_file is None:
        out_file = get_cache_dir() / "burst_dict.json.gz"
    out_file = Path(out_file)
    if not out_file.exists():
        r = requests.get(url)
        if r.status_code != 200:
            raise ValueError(f"Could not download {url}")
        with open(out_file, "wb") as f:
            f.write(r.content)

    return out_file


def read_burst_json(fname: Union[str, Path]) -> dict:
    """Read a pre-downloaded burst-dict file into memory."""
    if str(fname).endswith(".gz"):
        with gzip.open(fname, "r") as f:
            return json.loads(f.read().decode("utf-8"))
    else:
        with open(fname) as f:
            return json.load(f)
