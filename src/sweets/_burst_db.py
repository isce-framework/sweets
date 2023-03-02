import shutil
from pathlib import Path
from typing import Optional

from ._log import get_log
from ._types import Filename
from .utils import get_cache_dir

# import requests


DEFAULT_BURST_DB_FILE = Path("/u/aurora-r0/staniewi/dev/burst_map_bbox_only.sqlite3")

logger = get_log(__name__)


def get_burst_db(
    uri: Filename = DEFAULT_BURST_DB_FILE, out_file: Optional[Filename] = None
) -> Path:
    """Read or download the burst-db file.

    TODO: getting a URL to a release version of the sqlite database.

    Parameters
    ----------
    uri : str, optional
        Location to download from.
        Note: this currently is a local file, by default DEFAULT_BURST_DB_FILE
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

    logger.info(f"Copying {uri}...")
    shutil.copy(uri, out_file)
    # r = requests.get(uri)
    # r.raise_for_status()
    # with open(out_file, "wb") as f:
    #     f.write(r.content)
    return out_file


# @functools.lru_cache(maxsize=2)
# def _read_burst_json(fname: Filename) -> Dict[str, List[str]]:
#     """Read a pre-downloaded burst-dict file into memory."""
#     if str(fname).endswith(".gz"):
#         with gzip.open(fname, "r") as f:
#             return json.loads(f.read().decode("utf-8"))
#     else:
#         with open(fname) as f:
#             return json.load(f)
