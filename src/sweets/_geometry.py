from __future__ import annotations

from pathlib import Path
from typing import Optional

import rasterio as rio
from dolphin import io, stitching
from dolphin._types import Bbox
from opera_utils import group_by_burst

from ._log import get_log
from ._types import Filename

logger = get_log(__name__)


def stitch_geometry(
    geom_path_list: list[Filename],
    geom_dir: Path,
    dem_filename: Filename,
    looks: tuple[int, int],
    bbox: Optional[Bbox] = None,
    overwrite: bool = False,
):
    """Stitch the burst-wise geometry files.

    Parameters
    ----------
    geom_path_list : list[Filename]
        List of paths to individual HDF5 burst static_layers files.
    geom_dir : Path
        Directory where the stitched geometry file will be saved.
    dem_filename : Filename
        Digital elevation model (DEM) filename used for processing.
    looks : tuple[int, int]
        (row, column) looks for multi-looking the stitched geometry.
    bbox : Optional[Bbox], default=None
        Bounding box to subset the stitched geometry. If None, no subsetting is applied.
    overwrite : bool, default=False
        If True, overwrite the stitched geometry files if they exists.

    Returns
    -------
    stitched_geom_files : list[Path]
        The output stitched files.
    """
    geom_dir.mkdir(parents=True, exist_ok=True)

    file_list = []
    # TODO: do I want to handle this missing data problem differently?
    # if there's a burst with the 1st day missing so the static_layers_
    # file is a different date... is that a problem?
    for burst, files in group_by_burst(geom_path_list).items():
        if len(files) > 1:
            logger.warning(f"Found {len(files)} static_layers files for {burst}")
        file_list.append(files[0])
    logger.info(f"Stitching {len(file_list)} images.")

    # Convert row/col looks to strides for the right shape
    strides = {"x": looks[1], "y": looks[0]}
    stitched_geom_files = []
    # local_incidence_angle needed by anyone?
    datasets = ["los_east", "los_north", "layover_shadow_mask"]
    # Descriptions from:
    # https://github.com/opera-adt/Static_Layers_CSLC-S1_Specs/blob/main/XML/static_layers_cslc-s1.xml
    descriptions = [
        "East component of LOS unit vector from target to sensor",
        "North component of LOS unit vector from target to sensor",
        (
            "Layover shadow mask. 0=no layover, no shadow; 1=shadow; 2=layover;"
            " 3=shadow and layover."
        ),
    ]
    # layover_shadow_mask is Int8 with 127 meaning nodata
    nodatas = [0, 0, 127]
    for ds_name, nodata, desc in zip(datasets, nodatas, descriptions):
        outfile = geom_dir / f"{ds_name}.tif"
        logger.info(f"Creating {outfile}")
        stitched_geom_files.append(outfile)
        # Used to be:
        # /science/SENTINEL1/CSLC/grids/static_layers
        # we might also move this to dolphin if we do use the layers
        ds_path = f"/data/{ds_name}"
        cur_files = [io.format_nc_filename(f, ds_name=ds_path) for f in file_list]

        stitching.merge_images(
            cur_files,
            outfile=outfile,
            driver="GTiff",
            out_bounds=bbox,
            out_bounds_epsg=4326,
            target_aligned_pixels=True,
            in_nodata=nodata,
            out_nodata=nodata,
            strides=strides,
            resample_alg="nearest",
            overwrite=overwrite,
        )
        with rio.open(outfile, "r+") as src:
            src.set_band_description(1, desc)

    # Create the height (from dem) at the same resolution as the interferograms
    height_file = geom_dir / "height.tif"
    stitched_geom_files.append(height_file)

    if height_file.exists() and not overwrite:
        logger.info(f"{height_file} exists, skipping.")
    else:
        logger.info(f"Creating {height_file}")
        stitched_geom_files.append(height_file)
        stitching.warp_to_match(
            input_file=dem_filename,
            match_file=stitched_geom_files[0],
            output_file=height_file,
            resample_alg="cubic",
        )

    return stitched_geom_files
