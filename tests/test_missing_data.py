from pathlib import Path

import numpy as np
import pytest
from osgeo import gdal

try:
    from sweets._missing_data import Stats, get_bad_files, get_raster_stats, is_valid

    MISSING_DEPS = False
except ImportError:
    MISSING_DEPS = True


# Fixture for creating temporary files
@pytest.mark.skipif(MISSING_DEPS)
@pytest.fixture()
def slc_file_list(tmp_path):
    shape = (10, 100, 100)
    nodata = np.nan  # Set nodata value

    # Create random data with a certain percentage of nodata pixels
    slc_stack = np.random.random(shape)
    nodata_mask = np.random.choice(
        [True, False], shape, p=[0.2, 0.8]
    )  # 20% of pixels are nodata
    slc_stack[nodata_mask] = nodata

    # Write to a file
    driver = gdal.GetDriverByName("GTiff")
    start_date = 20220101
    d = tmp_path / "gtiffs"
    d.mkdir()
    name_template = d / "{date}.slc.tif"
    file_list = []
    for i in range(shape[0]):
        fname = str(name_template).format(date=str(start_date + i))
        file_list.append(Path(fname))
        ds = driver.Create(fname, shape[-1], shape[-2], 1, gdal.GDT_CFloat32)
        ds.GetRasterBand(1).WriteArray(slc_stack[i])
        ds.GetRasterBand(1).SetNoDataValue(nodata)
        ds = None

    return file_list


# Now we write test cases using the created fixture
@pytest.mark.skipif(MISSING_DEPS)
def test_get_bad_files(slc_file_list):
    bad_files, bad_stats = get_bad_files(str(slc_file_list[0].parent), max_jobs=1)
    assert isinstance(bad_files, list)
    assert isinstance(bad_stats, list)
    get_bad_files(str(slc_file_list[0].parent), max_jobs=5)


@pytest.mark.skipif(MISSING_DEPS)
def test_is_valid(slc_file_list):
    valid, reason = is_valid(str(slc_file_list[0]))
    assert valid
    assert reason == ""


@pytest.mark.skipif(MISSING_DEPS)
def test_get_raster_stats(slc_file_list):
    stats = get_raster_stats(str(slc_file_list[0]))
    assert isinstance(stats, Stats)
