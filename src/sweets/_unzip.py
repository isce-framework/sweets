import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List

from sweets._log import get_log
from sweets._types import Filename

logger = get_log(__name__)


def unzip_one(filepath: Filename, pol: str = "vv", out_dir=Path(".")) -> Path:
    """Unzip one Sentinel-1 zip file."""
    if pol is None:
        pol = ""

    # unzip all of these
    to_unzip = [pol.lower(), "preview", "support", "manifest.safe"]
    with zipfile.ZipFile(filepath, "r") as zipref:
        # Get the list of files in the zip
        names_to_extract = [
            fp
            for fp in zipref.namelist()
            if any(key in str(fp).lower() for key in to_unzip)
        ]
        zipref.extractall(path=out_dir, members=names_to_extract)
    # Return the path to the unzipped file
    return Path(filepath).with_suffix(".SAFE")


def unzip_all(
    path: Filename = ".", pol: str = "vv", delete_zips: bool = False, n_workers: int = 4
) -> List[Path]:
    """Find all .zips and unzip them, skipping overwrites."""
    zip_files = list(Path(path).glob("S1[AB]_*IW*.zip"))
    logger.info(f"Found {len(zip_files)} zip files to unzip")

    existing_safes = list(Path(path).glob("S1[AB]_*IW*.SAFE"))
    logger.info(f"Found {len(existing_safes)} SAFE files already unzipped")

    # Skip if already unzipped
    files_to_unzip = [
        fp for fp in zip_files if fp.stem not in [sf.stem for sf in existing_safes]
    ]
    logger.info(f"Unzipping {len(files_to_unzip)} zip files")
    # Unzip in parallel
    newly_unzipped = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(unzip_one, fp, pol=pol) for fp in files_to_unzip]
        for future in as_completed(futures):
            newly_unzipped.append(future.result())

    if delete_zips:
        for fp in files_to_unzip:
            fp.unlink()
    return newly_unzipped + existing_safes
