import zipfile
from pathlib import Path
from typing import Optional

import requests

from ._log import get_log
from ._types import Filename
from .utils import get_cache_dir

BURST_DB_URL = "https://github.com/scottstanie/burst_db/raw/frames-with-data/data/s1-frames-9frames-5min-10max-bbox-only.gpkg.zip"  # noqa: E501

logger = get_log(__name__)


def get_burst_db(url: str = BURST_DB_URL, out_file: Optional[Filename] = None) -> Path:
    """Read or download the burst-db file.

    TODO: getting a URL to a release version of the sqlite database.

    Parameters
    ----------
    url : str, optional
        Location to download from.
    out_file : Optional[Filename], optional
        Location to save locally. If None, will save to
        `get_cache_dir() / "burst_map_bbox_only.sqlite3"`, by default None

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
        out_file = get_cache_dir() / "burst_map_bbox_only.sqlite3"
    out_file = Path(out_file)

    if out_file.exists():
        logger.info(f"Using cached burst DB at {out_file}")
        return out_file

    logger.info(f"Copying {url} to local...")
    remote_filename = url.split("/")[-1]
    temp_file = out_file.parent / remote_filename
    print(temp_file)
    r = requests.get(url)
    r.raise_for_status()
    with open(temp_file, "wb") as f:
        f.write(r.content)
    if str(temp_file).endswith(".zip"):
        # extract and rename the result to out_file
        with zipfile.ZipFile(temp_file, "r") as zip_ref:
            zipinfo = zip_ref.infolist()[0]
            zip_ref.extract(zipinfo, out_file.parent)
            (out_file.parent / zipinfo.filename).rename(out_file)

    return out_file
