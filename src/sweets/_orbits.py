from pathlib import Path
from typing import List

from eof import download


def download_orbits(search_path: Path, save_dir: Path) -> List[Path]:
    """Download orbit files for a given search path."""
    print(f"search_path: {search_path}")
    print(f"save_dir: {save_dir}")
    filenames = download.main(
        search_path=search_path,
        save_dir=save_dir,
    )
    if not filenames:
        return []
    return [Path(f) for f in filenames]
