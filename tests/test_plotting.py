"""Tests for sweets.plotting.

Covers the oil_slick colormap registration (import-time side effect) and
a minimal plot_ifg invocation on a synthetic array, no raster I/O.
"""

from __future__ import annotations

import inspect

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from sweets import plotting


def test_oil_slick_registered_with_matplotlib() -> None:
    assert "oil_slick" in mpl.colormaps
    cm = mpl.colormaps["oil_slick"]
    assert cm.name == "oil_slick"


def test_oil_slick_cmap_properties() -> None:
    assert plotting.OIL_SLICK_CMAP.name == "oil_slick"
    assert len(plotting._OIL_SLICK_NODES) == 11
    # First and last nodes must match so the colormap is seamless at 2pi wrap.
    assert np.allclose(plotting._OIL_SLICK_NODES[0], plotting._OIL_SLICK_NODES[-1])


def test_plot_ifg_default_is_oil_slick() -> None:
    sig = inspect.signature(plotting.plot_ifg)
    assert sig.parameters["phase_cmap"].default == "oil_slick"


def test_plot_ifg_accepts_complex_array() -> None:
    y, x = np.mgrid[0:32, 0:32]
    phase = 0.1 * x - 0.05 * y
    ifg = np.exp(1j * phase).astype(np.complex64)
    fig, ax = plotting.plot_ifg(img=ifg, figsize=(3, 3))
    assert fig is not None and ax is not None
    assert len(ax.images) == 1
    plt.close("all")


def test_plot_ifg_accepts_real_phase_array() -> None:
    phase = np.linspace(-np.pi, np.pi, 64).reshape(8, 8)
    fig, ax = plotting.plot_ifg(img=phase, figsize=(3, 3), add_colorbar=False)
    assert fig is not None and ax is not None
    assert len(ax.images) == 1
    plt.close("all")
