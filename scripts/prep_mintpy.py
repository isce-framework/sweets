#!/usr/bin/env python
############################################################
# Program is part of MintPy                                #
# Copyright (c) 2013, Zhang Yunjun, Heresh Fattahi         #
# Author: Talib Oliver Cabrerra, Scott Staniewicz          #
############################################################


import argparse
import datetime
import glob
import itertools
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pyproj
from dolphin import io
from dolphin.utils import full_suffix, get_dates
from dolphin.workflows.config import OPERA_DATASET_ROOT
from mintpy.utils import arg_utils, isce_utils, ptime, readfile, writefile

####################################################################################
EXAMPLE = """example:

  python3 ./prep_sweets.py -u 'interferograms/stitched/*.unw'  -m gslcs/t064_135516_iw3/20190704/static_layers_t064_135516_iw3.h5

  ## example commands after prep_sweets.py
  reference_point.py timeseries.h5 -y 500 -x 1150
  generate_mask.py temporalCoherence.h5 -m 0.7 -o maskTempCoh.h5
  tropo_pyaps3.py -f timeseries.h5 -g inputs/geometryRadar.h5
  remove_ramp.py timeseries_ERA5.h5 -m maskTempCoh.h5 -s linear
  dem_error.py timeseries_ERA5_ramp.h5 -g inputs/geometryRadar.h5
  timeseries2velocity.py timeseries_ERA5_ramp_demErr.h5
  geocode.py velocity.h5 -l inputs/geometryRadar.h5
"""  # noqa: E501

# """
# Scott TODO:
# - dont need the ifg sample file, the unwrapped files should have everything
# - UTM_ZONE, EPSG from the stitched IFG (it won't work to get a single GSLC burst)
# - pixel size is wrong since we're taking range/azimuth size, instead of geocoded size
# - HEIGHT: do we wanna try to get that from the saved orbit info?
# -  cor_files  = [x.split(".unw")[0] + ".cor" for x in unw_files]
# -    This might be wrong for dolphin? to verify

# Scott did:
# - remove bbox
# - "set all data less than 0" thats a fringe specific thing to do -1, -2 for tcoh

# """


def _create_parser():
    parser = argparse.ArgumentParser(
        description="Prepare Sweets products for MintPy",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=EXAMPLE,
    )

    parser.add_argument(
        "-u",
        "--unw-file-glob",
        type=str,
        default="./interferograms/unwrapped/*.unw.tif",
        help="path pattern of unwrapped interferograms (default: %(default)s).",
    )
    parser.add_argument(
        "-c",
        "--cor-file-glob",
        type=str,
        default="./interferograms/stitched/*.cor",
        help="path pattern of unwrapped interferograms (default: %(default)s).",
    )
    parser.add_argument(
        "-g",
        "--geom-dir",
        default="./geometry",
        help="Geometry directory (default: %(default)s).\n",
    )
    parser.add_argument(
        "-m",
        "--meta-file",
        dest="metaFile",
        type=str,
        help="GSLC metadata file or directory",
    )
    parser.add_argument(
        "-b",
        "--baseline-dir",
        dest="baselineDir",
        type=str,
        default=None,
        help="baseline directory (default: %(default)s).",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        dest="outDir",
        type=str,
        default="./mintpy",
        help="output directory (default: %(default)s).",
    )
    parser.add_argument(
        "-r",
        "--range",
        dest="lks_x",
        type=int,
        default=1,
        help=(
            "number of looks in range direction, for multilooking applied after fringe"
            " processing.\nOnly impacts metadata. (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "-a",
        "--azimuth",
        dest="lks_y",
        type=int,
        default=1,
        help=(
            "number of looks in azimuth direction, for multilooking applied after"
            " fringe processing.\nOnly impacts metadata. (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--single-reference",
        action="store_true",
        help=(
            "Indicate that all the unwrapped ifgs are single reference, which allows"
            " use to create the timeseries.h5 file directly without inversion."
        ),
    )

    parser = arg_utils.add_subset_argument(parser, geo=False)

    return parser


def cmd_line_parse(iargs=None):
    """Create the command line parser."""
    parser = _create_parser()
    inps = parser.parse_args(args=iargs)

    # in case meta_file is input as wildcard
    inps.metaFile = sorted(glob.glob(inps.metaFile))[0]

    return inps


####################################################################################


def prepare_metadata(meta_file, int_file, nlks_x=1, nlks_y=1):
    """Get the metadata from the GSLC metadata file and the unwrapped interferogram."""
    print("-" * 50)

    cols, rows = io.get_raster_xysize(int_file)

    meta_compass = h5py.File(meta_file, "r")
    meta = {}

    geotransform = io.get_raster_gt(int_file)
    meta["LENGTH"] = rows
    meta["WIDTH"] = cols

    meta["X_FIRST"] = geotransform[0]
    meta["Y_FIRST"] = geotransform[3]
    meta["X_STEP"] = geotransform[1]
    meta["Y_STEP"] = geotransform[5]
    meta["X_UNIT"] = meta["Y_UNIT"] = "meters"

    crs = io.get_raster_crs(int_file)
    meta["EPSG"] = crs.to_epsg()

    processing_ds = f"{OPERA_DATASET_ROOT}/metadata/processing_information"
    burst_ds = f"{processing_ds}/input_burst_metadata"

    meta["WAVELENGTH"] = meta_compass[f"{burst_ds}/wavelength"][()]
    meta["RANGE_PIXEL_SIZE"] = meta_compass[f"{burst_ds}/range_pixel_spacing"][()]
    meta["AZIMUTH_PIXEL_SIZE"] = 14.1
    meta["EARTH_RADIUS"] = 6371000.0

    t0 = datetime.datetime.strptime(
        meta_compass[f"{burst_ds}/sensing_start"][()].decode("utf-8"),
        "%Y-%m-%d %H:%M:%S.%f",
    )
    t1 = datetime.datetime.strptime(
        meta_compass[f"{burst_ds}/sensing_stop"][()].decode("utf-8"),
        "%Y-%m-%d %H:%M:%S.%f",
    )
    t_mid = t0 + (t1 - t0) / 2.0
    meta["CENTER_LINE_UTC"] = (
        t_mid - datetime.datetime(t_mid.year, t_mid.month, t_mid.day)
    ).total_seconds()
    meta["HEIGHT"] = 750000.0
    meta["STARTING_RANGE"] = meta_compass[f"{burst_ds}/starting_range"][()]
    meta["PLATFORM"] = meta_compass[f"{burst_ds}/platform_id"][()].decode("utf-8")
    meta["ORBIT_DIRECTION"] = meta_compass[
        f"{OPERA_DATASET_ROOT}/metadata/orbit/orbit_direction"
    ][()].decode("utf-8")
    meta["ALOOKS"] = 1
    meta["RLOOKS"] = 1

    # apply optional user multilooking
    if nlks_x > 1:
        meta["RANGE_PIXEL_SIZE"] = str(float(meta["RANGE_PIXEL_SIZE"]) * nlks_x)
        meta["RLOOKS"] = str(float(meta["RLOOKS"]) * nlks_x)

    if nlks_y > 1:
        meta["AZIMUTH_PIXEL_SIZE"] = str(float(meta["AZIMUTH_PIXEL_SIZE"]) * nlks_y)
        meta["ALOOKS"] = str(float(meta["ALOOKS"]) * nlks_y)

    return meta


def _get_xy_arrays(atr):
    x0 = float(atr["X_FIRST"])
    y0 = float(atr["Y_FIRST"])
    x_step = float(atr["X_STEP"])
    y_step = float(atr["Y_STEP"])
    rows = int(atr["LENGTH"])
    cols = int(atr["WIDTH"])
    x_arr = x0 + x_step * np.arange(cols)
    y_arr = y0 + y_step * np.arange(rows)
    # Shift by half pixel to get the centers
    x_arr += x_step / 2
    y_arr += y_step / 2
    return x_arr, y_arr


def write_coordinate_system(
    filename, dset_name, xy_dim_names=("x", "y"), grid_mapping_dset="spatial_ref"
):
    """Write the coordinate system CF metadata to an existing HDF5 file."""
    x_dim_name, y_dim_name = xy_dim_names
    atr = readfile.read_attribute(filename)
    epsg = int(atr.get("EPSG", 4326))

    with h5py.File(filename, "a") as hf:
        crs = pyproj.CRS.from_user_input(epsg)
        dset = hf[dset_name]

        # Setup the dataset holding the SRS information
        srs_dset = hf.require_dataset(grid_mapping_dset, shape=(), dtype=int)
        srs_dset.attrs.update(crs.to_cf())
        dset.attrs["grid_mapping"] = grid_mapping_dset

        if "date" in hf:
            date_arr = [
                datetime.datetime.strptime(ds, "%Y%m%d")
                for ds in hf["date"][()].astype(str)
            ]
            days_since = [(d - date_arr[0]).days for d in date_arr]
            dt_dim = hf.create_dataset("time", data=days_since)
            dt_dim.make_scale()
            cf_attrs = dict(
                units=f"days since {str(date_arr[0])}", calendar="proleptic_gregorian"
            )
            dt_dim.attrs.update(cf_attrs)
            dset.dims[0].attach_scale(dt_dim)
            dset.dims[0].label = "time"
        else:
            dt_dim = date_arr = None
            # If we want to do something other than time as a 3rd dimension...
            #  We'll need to figure out what other valid dims there are
            # otherwise, we can just do `phony_dims="sort"` in xarray

        # add metadata to x,y coordinates
        is_projected = crs.is_projected
        is_geographic = crs.is_geographic
        x_arr, y_arr = _get_xy_arrays(atr)
        x_dim_dset = hf.create_dataset(x_dim_name, data=x_arr)
        x_dim_dset.make_scale(x_dim_name)
        y_dim_dset = hf.create_dataset(y_dim_name, data=y_arr)
        y_dim_dset.make_scale(y_dim_name)

        x_coord_attrs = {}
        x_coord_attrs["axis"] = "X"
        y_coord_attrs = {}
        y_coord_attrs["axis"] = "Y"
        if is_projected:
            units = "meter"
            # X metadata
            x_coord_attrs["long_name"] = "x coordinate of projection"
            x_coord_attrs["standard_name"] = "projection_x_coordinate"
            x_coord_attrs["units"] = units
            # Y metadata
            y_coord_attrs["long_name"] = "y coordinate of projection"
            y_coord_attrs["standard_name"] = "projection_y_coordinate"
            y_coord_attrs["units"] = units
        elif is_geographic:
            # X metadata
            x_coord_attrs["long_name"] = "longitude"
            x_coord_attrs["standard_name"] = "longitude"
            x_coord_attrs["units"] = "degrees_east"
            # Y metadata
            y_coord_attrs["long_name"] = "latitude"
            y_coord_attrs["standard_name"] = "latitude"
            y_coord_attrs["units"] = "degrees_north"
        y_dim_dset.attrs.update(y_coord_attrs)
        x_dim_dset.attrs.update(x_coord_attrs)

        ndim = dset.ndim
        dset.dims[ndim - 1].attach_scale(x_dim_dset)
        dset.dims[ndim - 2].attach_scale(y_dim_dset)
        dset.dims[ndim - 1].label = x_dim_name
        dset.dims[ndim - 2].label = y_dim_name


def _get_date_pairs(filenames):
    str_list = [Path(f).stem for f in filenames]
    return [str(f).replace(full_suffix(f), "") for f in str_list]


def prepare_timeseries(
    outfile,
    unw_files,
    metadata,
    processor,
    baseline_dir=None,
):
    """Prepare the timeseries file."""
    print("-" * 50)
    print("preparing timeseries file: {}".format(outfile))

    # copy metadata to meta
    meta = {key: value for key, value in metadata.items()}
    phase2range = float(meta["WAVELENGTH"]) / (4.0 * np.pi)

    # grab date list from the filename
    date12_list = _get_date_pairs(unw_files)
    num_file = len(unw_files)
    print("number of unwrapped interferograms: {}".format(num_file))

    date_pairs = [dl.split("_") for dl in date12_list]
    date_list = sorted(set(itertools.chain.from_iterable(date_pairs)))
    # ref_date = date12_list[0].split("_")[0]
    # date_list = [ref_date] + [date12.split("_")[1] for date12 in date12_list]
    num_date = len(date_list)
    print("number of acquisitions: {}\n{}".format(num_date, date_list))

    # baseline info
    pbase = np.zeros(num_date, dtype=np.float32)
    if baseline_dir is not None:
        # read baseline data
        # TODO: this isn't implemented yet
        raise NotImplementedError
        # baseline_dict = isce_utils.read_baseline_timeseries(
        #     baseline_dir, processor=processor, ref_date=ref_date
        # )
        # # dict to array
        # for i in range(num_date):
        #     pbase_top, pbase_bottom = baseline_dict[date_list[i]]
        #     pbase[i] = (pbase_top + pbase_bottom) / 2.0

    # size info
    cols, rows = io.get_raster_xysize(unw_files[0])

    # define dataset structure
    dates = np.array(date_list, dtype=np.string_)
    ds_name_dict = {
        "date": [dates.dtype, (num_date,), dates],
        "bperp": [np.float32, (num_date,), pbase],
        "timeseries": [np.float32, (num_date, rows, cols), None],
    }

    # initiate HDF5 file
    meta["FILE_TYPE"] = "timeseries"
    meta["UNIT"] = "m"
    # meta["REF_DATE"] = ref_date # might not be the first date!
    writefile.layout_hdf5(outfile, ds_name_dict, metadata=meta)

    # writing data to HDF5 file
    print("writing data to HDF5 file {} with a mode ...".format(outfile))
    with h5py.File(outfile, "a") as f:
        prog_bar = ptime.progressBar(maxValue=num_file)
        for i, unw_file in enumerate(unw_files):
            # read data using gdal
            data = io.load_gdal(unw_file)

            f["timeseries"][i + 1] = data * phase2range
            prog_bar.update(i + 1, suffix=date12_list[i])
        prog_bar.close()

        print("set value at the first acquisition to ZERO.")
        f["timeseries"][0] = 0.0

    print("finished writing to HDF5 file: {}".format(outfile))
    return outfile


def prepare_geometry(outfile, geom_dir, metadata, box, water_mask_file=None):
    """Prepare the geometry file."""
    print("-" * 50)
    print(f"preparing geometry file: {outfile}")

    geom_path = Path(geom_dir)
    # copy metadata to meta
    meta = {key: value for key, value in metadata.items()}
    meta["FILE_TYPE"] = "geometry"

    file_to_path = {
        "height": geom_path / "height.tif",
        "incidenceAngle": geom_path / "incidence_angle.tif",
        "azimuthAngle": geom_path / "heading_angle.tif",
        "shadowMask": geom_path / "layover_shadow_mask.tif",
    }

    if water_mask_file:
        file_to_path["waterMask"] = water_mask_file

    dsDict = {}
    for dsName, fname in file_to_path.items():
        dsDict[dsName] = readfile.read(fname, datasetName=dsName, box=box)[0]

    # write data to HDF5 file
    writefile.write(dsDict, outfile, metadata=meta)

    return outfile


def prepare_temporal_coherence(outfile, infile, metadata):
    """Prepare the temporal coherence file."""
    print("-" * 50)
    print("preparing temporal coherence file: {}".format(outfile))

    # copy metadata to meta
    meta = {key: value for key, value in metadata.items()}
    meta["FILE_TYPE"] = "temporalCoherence"
    meta["UNIT"] = "1"

    data = io.load_gdal(infile)

    print(data.shape)
    # write to HDF5 file
    writefile.write(data, outfile, metadata=meta)
    return outfile


def prepare_ps_mask(outfile, infile, metadata):
    """Prepare the PS mask file."""
    print("-" * 50)
    print("preparing PS mask file: {}".format(outfile))

    # copy metadata to meta
    meta = {key: value for key, value in metadata.items()}
    meta["FILE_TYPE"] = "mask"
    meta["UNIT"] = "1"

    # read data using gdal
    data = io.load_gdal(infile)

    # write to HDF5 file
    writefile.write(data, outfile, metadata=meta)
    return outfile


def prepare_stack(
    outfile,
    unw_files,
    cor_files,
    metadata,
    processor,
    baseline_dir=None,
):
    """Prepare the input unw stack."""
    print("-" * 50)
    print("preparing ifgramStack file: {}".format(outfile))
    # copy metadata to meta
    meta = {key: value for key, value in metadata.items()}

    # get list of *.unw file
    num_pair = len(unw_files)
    unw_ext = full_suffix(unw_files[0])

    print(unw_files)
    print(f"number of unwrapped interferograms: {num_pair}")
    print(f"number of correlation files: {len(cor_files)}")
    print(cor_files)

    # get list of *.unw.conncomp file
    cc_files = [str(x).replace(unw_ext, ".unw.conncomp") for x in unw_files]
    cc_files = [x for x in cc_files if Path(x).exists()]
    print(f"number of connected components files: {len(cc_files)}")

    if len(cc_files) != len(unw_files) or len(cor_files) != len(unw_files):
        print(
            "the number of *.unw and *.unw.conncomp or *.cor files are NOT consistent"
        )
        if len(unw_files) > len(cor_files):
            print("skip creating ifgramStack.h5 file.")
            return
        print("Keeping only cor files which match a unw file")
        unw_dates_set = set([tuple(get_dates(f)) for f in unw_files])
        cor_files = [f for f in cor_files if tuple(get_dates(f)) in unw_dates_set]

    # get date info: date12_list
    date12_list = _get_date_pairs(unw_files)

    pbase = np.zeros(num_pair, dtype=np.float32)
    # prepare baseline info
    if baseline_dir is not None:
        # read baseline timeseries
        baseline_dict = isce_utils.read_baseline_timeseries(
            baseline_dir, processor=processor
        )

        # calc baseline for each pair
        print("calc perp baseline pairs from time-series")
        # pbase = np.zeros(num_pair, dtype=np.float32)
        for i, date12 in enumerate(date12_list):
            [date1, date2] = date12.split("_")
            pbase[i] = np.subtract(baseline_dict[date2], baseline_dict[date1]).mean()

    # size info
    cols, rows = io.get_raster_xysize(unw_files[0])

    # define (and fill out some) dataset structure
    date12_arr = np.array([x.split("_") for x in date12_list], dtype=np.string_)
    drop_ifgram = np.ones(num_pair, dtype=np.bool_)
    ds_name_dict = {
        "date": [date12_arr.dtype, (num_pair, 2), date12_arr],
        "bperp": [np.float32, (num_pair,), pbase],
        "dropIfgram": [np.bool_, (num_pair,), drop_ifgram],
        "unwrapPhase": [np.float32, (num_pair, rows, cols), None],
        "coherence": [np.float32, (num_pair, rows, cols), None],
        "connectComponent": [
            np.float32,
            (num_pair, rows, cols),
            None,
        ],
    }

    # initiate HDF5 file
    meta["FILE_TYPE"] = "ifgramStack"
    writefile.layout_hdf5(outfile, ds_name_dict, metadata=meta)

    # writing data to HDF5 file
    print("writing data to HDF5 file {} with a mode ...".format(outfile))
    with h5py.File(outfile, "a") as f:
        prog_bar = ptime.progressBar(maxValue=num_pair)
        for i, (unw_file, cor_file, cc_file) in enumerate(
            zip(unw_files, cor_files, cc_files)
        ):
            # read/write *.unw file
            f["unwrapPhase"][i] = io.load_gdal(unw_file)

            # read/write *.cor file
            f["coherence"][i] = io.load_gdal(cor_file)

            # read/write *.unw.conncomp file
            f["connectComponent"][i] = io.load_gdal(cc_file)

            prog_bar.update(i + 1, suffix=date12_list[i])
        prog_bar.close()

    print("finished writing to HDF5 file: {}".format(outfile))
    return outfile


def main(iargs=None):
    """Run the preparation functions."""
    inps = cmd_line_parse(iargs)

    unw_files = sorted(glob.glob(inps.unw_file_glob))
    print(f"Found {len(unw_files)} unwrapped files")
    cor_files = sorted(glob.glob(inps.cor_file_glob))
    print(f"Found {len(cor_files)} correlation files")
    # translate input options
    processor = "sweets"  # isce_utils.get_processor(inps.metaFile)
    # metadata
    meta_file = Path(inps.metaFile)
    if meta_file.is_dir():
        # Search for the line of sight static_layers file
        try:
            # Grab the first one in in the directory
            meta_file = next(meta_file.rglob("static_*.h5"))
        except StopIteration:
            raise ValueError(f"No static layers file found in {meta_file}")

    meta = prepare_metadata(
        meta_file, unw_files[0], nlks_x=inps.lks_x, nlks_y=inps.lks_y
    )

    # output directory
    for dname in [inps.outDir, os.path.join(inps.outDir, "inputs")]:
        os.makedirs(dname, exist_ok=True)

    # output filename
    # tcoh_file    = os.path.join(inps.outDir, 'temporalCoherence.h5')
    # ps_mask_file = os.path.join(inps.outDir, 'maskPS.h5')
    stack_file = os.path.join(inps.outDir, "inputs/ifgramStack.h5")
    ts_file = os.path.join(inps.outDir, "timeseries.h5")

    if inps.single_reference:
        # 2 - time-series (if inputs are all single-reference)
        prepare_timeseries(
            outfile=ts_file,
            unw_files=unw_files,
            metadata=meta,
            processor=processor,
            baseline_dir=inps.baselineDir,
        )

    # 3 - temporal coherence and mask for PS (from fringe)
    # prepare_temporal_coherence(
    #     outfile=tcoh_file,
    #     infile=inps.cohFile,
    #     metadata=meta,
    #     box=None)

    # prepare_ps_mask(
    #     outfile=ps_mask_file,
    #     infile=inps.psMaskFile,
    #     metadata=meta,
    #     box=None)

    # 4 - prepare and ifgstack with connected components
    prepare_stack(
        outfile=stack_file,
        unw_files=unw_files,
        cor_files=cor_files,
        metadata=meta,
        processor=processor,
        baseline_dir=inps.baselineDir,
    )

    print("Done.")
    return


if __name__ == "__main__":
    main(sys.argv[1:])
