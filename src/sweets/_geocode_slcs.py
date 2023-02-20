from pathlib import Path
from typing import List, Optional, Tuple

from compass import s1_cslc, s1_geocode_stack

from ._types import Filename

# TODO: do i ever care to change these?
X_SPAC = 5
Y_SPAC = 10
POL = "VV"


def run(run_config_path: Filename):
    """Run a single geocoding run config."""
    s1_cslc.run(run_config_path)


def _create_config_files(
    slc_dir: Filename,
    burst_db_file: Filename,
    dem_file: Filename,
    orbit_dir: Filename,
    bbox: Optional[Tuple[float, ...]] = None,
    out_dir: Filename = Path("gslcs"),
) -> List[Path]:
    s1_geocode_stack.run(
        slc_dir=slc_dir,
        dem_file=dem_file,
        orbit_dir=orbit_dir,
        work_dir=out_dir,
        burst_db_file=burst_db_file,
        bbox=bbox,
        pol=POL,
        x_spac=X_SPAC,
        y_spac=Y_SPAC,
    )
    return sorted((Path(out_dir) / "runconfigs").glob("*"))
