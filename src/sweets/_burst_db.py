import functools
import gzip
import json
from pathlib import Path
from typing import Dict, List, Optional

import requests

from ._log import get_log
from ._types import Filename
from .utils import get_cache_dir

URL = "https://github.com/scottstanie/burst_db/raw/test-json-gz/src/burst_db/data/burst_map_bbox_only.json.gz"  # noqa

logger = get_log()


def get_burst_db(
    url: str = URL, out_file: Optional[Filename] = None
) -> Dict[str, List[str]]:
    """Read or download the burst-db file.

    Parameters
    ----------
    url : str, optional
        URL to download from.
    out_file : Optional[Filename], optional
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

    if out_file.exists():
        logger.info(f"Using cached burst DB at {out_file}")
        return _read_burst_json(out_file)

    logger.info(f"Downloading {url}...")
    r = requests.get(url)
    r.raise_for_status()
    with open(out_file, "wb") as f:
        f.write(r.content)
    return _read_burst_json(out_file)


@functools.lru_cache(maxsize=2)
def _read_burst_json(fname: Filename) -> Dict[str, List[str]]:
    """Read a pre-downloaded burst-dict file into memory."""
    if str(fname).endswith(".gz"):
        with gzip.open(fname, "r") as f:
            return json.loads(f.read().decode("utf-8"))
    else:
        with open(fname) as f:
            return json.load(f)
