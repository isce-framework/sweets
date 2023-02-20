import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from sweets._log import get_log
from sweets._types import Filename

logger = get_log(__name__)


def unzip_one(filepath: Filename, pol: str = "vv", out_dir=Path(".")):
    """Unzip one Sentinel-1 zip file."""
    if pol is None:
        pol = ""
    with zipfile.ZipFile(filepath, "r") as zipref:
        # Get the list of files in the zip
        names_to_extract = [
            fp for fp in zipref.namelist() if pol.lower() in str(fp).lower()
        ]
        zipref.extractall(path=out_dir, members=names_to_extract)


def unzip_all(
    path: Filename = ".", pol: str = "vv", delete_zips: bool = False, n_workers: int = 4
):
    """Find all .zips and unzip them, skipping overwrites."""
    zip_files = list(Path(path).glob("S1[AB]_*IW*.zip"))
    logger.info(f"Found {len(zip_files)} zip files to unzip")

    safe_files = list(Path(path).glob("S1[AB]_*IW*.SAFE"))
    logger.info(f"Found {len(safe_files)} SAFE files already unzipped")

    # Skip if already unzipped
    files_to_unzip = [
        fp for fp in zip_files if fp.stem not in [sf.stem for sf in safe_files]
    ]
    logger.info(f"Unzipping {len(files_to_unzip)} zip files")
    # Unzip in parallel
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(unzip_one, fp, pol=pol) for fp in files_to_unzip]
        for future in as_completed(futures):
            future.result()

    if delete_zips:
        for fp in files_to_unzip:
            fp.unlink()
