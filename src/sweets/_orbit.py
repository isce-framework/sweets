from pathlib import Path

from eof import download
from loguru import logger


def download_orbits(search_path: Path, save_dir: Path) -> list[Path]:
    """Download orbit files for a given search path."""
    logger.info(f"Orbit search_path: {search_path}, save_dir: {save_dir}")
    filenames = download.main(
        search_path=search_path,
        save_dir=save_dir,
    )
    if not filenames:
        return []
    return [Path(f) for f in filenames]
