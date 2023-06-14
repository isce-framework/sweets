from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Union

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import ipywidgets
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from cartopy.io import shapereader
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
from dolphin import io
from matplotlib.colors import LinearSegmentedColormap
from numpy.typing import ArrayLike
from shapely.geometry import Polygon, box

from ._types import Filename
from .core import UNW_SUFFIX


def plot_ifg(
    img: Optional[ArrayLike] = None,
    filename: Optional[Filename] = None,
    phase_cmap: str = "dismph",
    ax: Optional[plt.Axes] = None,
    add_colorbar: bool = True,
    title: str = "",
    figsize: Optional[tuple[float, float]] = None,
    plot_cor: bool = False,
    subsample_factor: Union[int, tuple[int, int]] = 1,
    **kwargs,
):
    """Plot an interferogram.

    Parameters
    ----------
    img : np.ndarray
        Complex interferogram array.
    filename : str
        Filename of interferogram to load.
    phase_cmap : str
        Colormap to use for phase.
    ax : matplotlib.axes.Axes
        Axes to plot on.
    add_colorbar : bool
        If true, add a colorbar to the plot.
    title : str
        Title for the plot.
    figsize : tuple
        Figure size.
    plot_cor : bool
        If true, plot the correlation image as well.
    subsample_factor : int or tuple[int, int]
        If loading from `filename`, amount to downsample when loading.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure of the plot.
    ax : matplotlib.axes.Axes
        Axes of the plot containing the interferogram.
    """
    if img is None:
        img = io.load_gdal(filename, subsample_factor=subsample_factor)
    else:
        # check for accidentally passing a filename as positional
        if isinstance(img, (Path, str)):
            img = io.load_gdal(img, subsample_factor=subsample_factor)
    phase = np.angle(img) if np.iscomplexobj(img) else img
    if plot_cor:
        cor = np.abs(img)

    if ax is None:
        if plot_cor:
            fig, (ax, cor_ax) = plt.subplots(ncols=2, figsize=figsize)
        else:
            fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Note: other interpolations (besides nearest/None) make dismph/cyclic maps look weird
    axim = ax.imshow(
        phase, cmap=phase_cmap, interpolation="nearest", vmax=3.14, vmin=-3.14
    )
    if add_colorbar:
        fig.colorbar(axim, ax=ax)
    if plot_cor:
        axim = cor_ax.imshow(cor, cmap="plasma", vmax=1, vmin=0)
        fig.colorbar(axim, ax=cor_ax)
    if title:
        ax.set_title(title)
    return fig, ax


def browse_ifgs(
    sweets_path: Optional[Filename] = None,
    file_list: Optional[list[Filename]] = None,
    amp_image: Optional[ArrayLike] = None,
    figsize: tuple[int, int] = (7, 4),
    vm_unw: float = 10,
    vm_cor: float = 1,
    layout="horizontal",
    axes: Optional[plt.Axes] = None,
    ref_unw: Optional[tuple[float, float]] = None,
    subsample_factor: Union[int, tuple[int, int]] = 1,
):
    """Browse interferograms in a sweets directory.

    Creates an interactive plot with a slider to browse stitched interferograms.

    Parameters
    ----------
    sweets_path : str
        Path to sweets directory.
    file_list : list[Filename], optional
        Alternative to `sweets_path`, directly provide a list of ifg files.
    amp_image : ArrayLike, optional
        If provided, plots an amplitude image along with the ifgs for visual reference.
    figsize : tuple
        Figure size.
    vm_unw : float
        Value used as min/max cutoff for unwrapped phase plot.
    vm_cor : float
        Value used as min/max cutoff for correlation phase plot.
    layout : str
        Layout of the plot. Can be "horizontal" or "vertical".
    axes : matplotlib.pyplot.Axes
        If provided, use this array of axes to plot the images.
        Otherwise, creates a new figure.
    ref_unw : Optional[tuple[int, int]]
        Reference point for all .unw files.
        If not passed, subtracts the mean of each file.
    subsample_factor : int or tuple[int, int]
        Amount to downsample when loading images.
    """

    def apply_ref(unw):
        mask = unw == 0
        ref = unw[ref_unw] if ref_unw is not None else unw.mean()
        unw -= ref
        unw[mask] = 0
        return unw

    if file_list is None:
        if sweets_path is None:
            raise ValueError("Must provide `file_list` or `sweets_path`")
        ifg_path = Path(sweets_path) / "interferograms/stitched"
        file_list = sorted(ifg_path.glob("2*.int"))
    file_list = [Path(f) for f in file_list]
    print(f"Browsing {len(file_list)} ifgs.")

    num_panels = 2  # ifg, cor

    # Check if we have unwrapped images
    unw_list = [
        Path(str(i).replace("stitched", "unwrapped")).with_suffix(UNW_SUFFIX)
        for i in file_list
    ]
    num_existing_unws = sum(f.exists() for f in unw_list)
    if num_existing_unws > 0:
        print(f"Found {num_existing_unws} {UNW_SUFFIX} files")
        num_panels += 1

    if amp_image is not None:
        num_panels += 1

    if axes is None:
        subplots_dict = dict(figsize=figsize, sharex=True, sharey=True)
        if layout == "horizontal":
            subplots_dict["ncols"] = num_panels
        else:
            subplots_dict["nrows"] = num_panels
        fig, axes = plt.subplots(**subplots_dict)
    else:
        fig = axes[0].figure

    # imgs = np.stack([io.load_gdal(f) for f in file_list])
    img = io.load_gdal(file_list[0], subsample_factor=subsample_factor)
    phase = np.angle(img)
    cor = np.abs(img)
    titles = [f.stem for f in file_list]  # type: ignore

    # plot once with colorbar
    plot_ifg(img=phase, add_colorbar=True, ax=axes[0])
    axim_ifg = axes[0].images[0]

    # vm_cor = np.nanpercentile(cor.ravel(), 99)
    axim_cor = axes[1].imshow(cor, cmap="plasma", vmin=0, vmax=vm_cor)
    fig.colorbar(axim_cor, ax=axes[1])

    if unw_list[0].exists():
        unw = apply_ref(io.load_gdal(unw_list[0], subsample_factor=subsample_factor))
        axim_unw = axes[2].imshow(unw, cmap="RdBu", vmin=-vm_unw, vmax=vm_unw)
        fig.colorbar(axim_unw, ax=axes[2])

    if amp_image is not None:
        vm_amp = np.nanpercentile(amp_image.ravel(), 99)
        axim_amp = axes[-1].imshow(amp_image, cmap="gray", vmax=vm_amp)
        fig.colorbar(axim_amp, ax=axes[2])

    @ipywidgets.interact(idx=(0, len(file_list) - 1))
    def browse_plot(idx=0):
        phase = np.angle(
            io.load_gdal(file_list[idx], subsample_factor=subsample_factor)
        )
        cor = np.abs(io.load_gdal(file_list[idx], subsample_factor=subsample_factor))
        axim_ifg.set_data(phase)
        axim_cor.set_data(cor)
        if unw_list[idx].exists():
            unw = apply_ref(
                io.load_gdal(unw_list[idx], subsample_factor=subsample_factor)
            )
            axim_unw.set_data(unw)
        fig.suptitle(titles[idx])


def _make_dismph_colors():
    """Create a cyclic colormap for insar phase."""
    red, green, blue = [], [], []
    for i in range(120):
        red.append(i * 2.13 * 155.0 / 255.0 + 100)
        green.append((119.0 - i) * 2.13 * 155.0 / 255.0 + 100.0)
        blue.append(255)
    for i in range(120):
        red.append(255)
        green.append(i * 2.13 * 155.0 / 255.0 + 100.0)
        blue.append((119 - i) * 2.13 * 155.0 / 255.0 + 100.0)
    for i in range(120):
        red.append((119 - i) * 2.13 * 155.0 / 255.0 + 100.0)
        green.append(255)
        blue.append(i * 2.13 * 155.0 / 255.0 + 100.0)
    return np.vstack((red, green, blue))


try:
    plt.get_cmap("dismph")
except:
    DISMPH = LinearSegmentedColormap.from_list("dismph", _make_dismph_colors().T / 256)
    plt.register_cmap(cmap=DISMPH)


# # @lru_cache(maxsize=30)
# def io.load_gdal(filename: , subsample_factor=subsample_factorFilename):
#     return io.load_gdal(filename)


def plot_area_of_interest(
    state: Optional[str] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    area_coordinates: Optional[List[Tuple[float, float]]] = None,
    buffer: float = 0.0,
    grid_step: Optional[float] = 1.0,
    ax=None,
) -> Tuple:
    """Make a basic map to highlight an area of interest.

    Parameters
    ----------
    state : str
        The name of the state to center on the plot
    bbox : tuple of float, optional
        A tuple representing the bounding box (minx, miny, maxx, maxy) of the AOI
    area_coordinates : list of tuple of float, optional
        A list of tuples representing the coordinates of the area of interest.
    buffer : float, optional
        The buffer distance for the state's geometry. Default is 0.0, meaning no buffer.
    grid_step : float
        Frequency to plot the grid in degrees. if `None`, skips the grid
    ax : matplotlib.axes._subplots.AxesSubplot, optional
        An Axes object to plot on. If None, a new figure and axes are created.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created matplotlib Figure instance.
    ax : matplotlib.axes._subplots.AxesSubplot
        The created or used Axes instance.
    """
    # Create a GeoAxes object if one wasn't provided
    if ax is None:
        # Use cartopy's GeoAxes
        fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
    else:
        fig = ax.figure

    # Add a background
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.COASTLINE)
    if state is not None:
        # Load the US States geometry
        states_shp = shapereader.natural_earth(
            resolution="110m",
            category="cultural",
            name="admin_1_states_provinces_lakes",
        )
        us_states = gpd.read_file(states_shp)

        # Filter for the state of interest
        state_geo = us_states[us_states.name.str.lower() == state.lower()]
        g = state_geo.geometry.iloc[0]
        buffered_state = g.buffer(buffer)
        left, bottom, right, top = buffered_state.bounds
        extent = (left, right, bottom, top)
        # Filter the GeoDataFrame to only include states that intersect
        intersecting_states = us_states[us_states.geometry.intersects(buffered_state)]
        # Plot the states
        intersecting_states.plot(ax=ax, color="none", edgecolor="black")
    else:
        states_provinces = cfeature.NaturalEarthFeature(
            category="cultural",
            name="admin_1_states_provinces_lines",
            scale="110m",
            facecolor="none",
        )
        buffered_bounds = box(*bbox).buffer(buffer).bounds
        left, bottom, right, top = buffered_bounds
        extent = (left, right, bottom, top)
        ax.add_feature(states_provinces, edgecolor="gray")

    ax.set_extent(extent, crs=ccrs.PlateCarree())
    # Set extent using the correct coordinate system

    if grid_step is not None:
        gl = ax.gridlines(draw_labels=True, alpha=0.2)

        gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, grid_step))
        gl.ylocator = mticker.FixedLocator(np.arange(-90, 90, grid_step))

        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.right_labels = False
        gl.top_labels = False

    aoi = None
    # Create a polygon for the area of interest
    if bbox is not None:
        aoi = box(*bbox)
    elif area_coordinates is not None:
        aoi = Polygon(area_coordinates)
    if aoi is not None:
        # Create a GeoDataFrame for the area of interest
        area_gdf = gpd.GeoDataFrame([1], geometry=[aoi], crs=state_geo.crs)

        # Plot the area of interest
        area_gdf.plot(ax=ax, edgecolor="red", facecolor="None", lw=2)
    return fig, ax
