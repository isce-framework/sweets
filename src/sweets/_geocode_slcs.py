from pathlib import Path
from typing import List, Union

from compass import s1_geocode_stack

# TODO: do i ever care to change these?
X_SPAC = 5
Y_SPAC = 10
POL = "VV"


def _create_config_files(
    slc_dir: Union[str, Path],
    burst_db_file: Union[str, Path],
    dem_file: Union[str, Path],
    orbit_dir: Union[str, Path],
    out_dir: Union[str, Path] = Path("gslcs"),
) -> List[Path]:
    s1_geocode_stack.run(
        slc_dir=slc_dir,
        dem_file=dem_file,
        orbit_dir=orbit_dir,
        work_dir=out_dir,
        burst_db_file=burst_db_file,
        pol=POL,
        x_spac=X_SPAC,
        y_spac=Y_SPAC,
    )
    return sorted((Path(out_dir) / "runconfigs").glob("*"))
