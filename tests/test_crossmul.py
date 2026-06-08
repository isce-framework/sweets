"""Tests for sweets._crossmul.

Pure math helpers are tested with synthetic numpy arrays (no I/O).
run_crossmul is tested with small GeoTIFF SLCs written to tmp_path.
"""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from sweets._crossmul import (
    CrossmulOptions,
    _boxcar_multilook,
    _compute_coherence,
    _gaussian_kernel_1d,
    _gaussian_multilook,
    _output_geotransform,
    run_crossmul,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(0)


def _make_slc(shape: tuple[int, int]) -> np.ndarray:
    """Unit-amplitude complex SLC with random phase."""
    phase = RNG.uniform(-np.pi, np.pi, shape).astype(np.float32)
    return np.exp(1j * phase).astype(np.complex64)


@pytest.fixture
def slc_geotiffs(tmp_path):
    """Two 40-row × 80-col complex64 GeoTIFFs for run_crossmul integration tests."""
    transform = from_bounds(0, 0, 8000, 4000, 80, 40)
    crs = rasterio.CRS.from_epsg(32610)

    # ifg = ref * conj(sec) = |ref|^2 * exp(j*(phi_ref - phi_sec))
    # sec leads by known_phase → ifg phase = -known_phase
    known_phase = 0.7
    ref_data = _make_slc((40, 80))
    sec_data = ref_data * np.exp(1j * known_phase).astype(np.complex64)

    paths = {}
    for name, data in (("ref.tif", ref_data), ("sec.tif", sec_data)):
        p = tmp_path / name
        with rasterio.open(
            p,
            "w",
            driver="GTiff",
            height=40,
            width=80,
            count=1,
            dtype="complex64",
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(data, 1)
        paths[name] = p

    return paths["ref.tif"], paths["sec.tif"], known_phase


# ---------------------------------------------------------------------------
# _gaussian_kernel_1d
# ---------------------------------------------------------------------------


def test_gaussian_kernel_sums_to_one():
    k = _gaussian_kernel_1d(sigma=3.0)
    assert abs(float(k.sum()) - 1.0) < 1e-6


def test_gaussian_kernel_is_symmetric():
    k = _gaussian_kernel_1d(sigma=5.0)
    np.testing.assert_allclose(k, k[::-1], atol=1e-7)


def test_gaussian_kernel_length():
    # radius = int(4.0 * sigma + 0.5); length = 2*radius + 1
    sigma = 4.0
    radius = int(4.0 * sigma + 0.5)
    k = _gaussian_kernel_1d(sigma)
    assert len(k) == 2 * radius + 1


# ---------------------------------------------------------------------------
# _boxcar_multilook
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("looks", [(4, 8), (2, 2), (3, 5)])
def test_boxcar_output_shape(looks):
    az, rg = looks
    arr = np.ones((40, 80))
    out = _boxcar_multilook(arr, looks)
    assert out.shape == (40 // az, 80 // rg)


def test_boxcar_constant_real():
    arr = np.full((20, 40), 3.5)
    out = _boxcar_multilook(arr, (4, 8))
    np.testing.assert_allclose(out, 3.5, rtol=1e-6)


def test_boxcar_constant_complex():
    val = 1.0 + 2.0j
    arr = np.full((20, 40), val, dtype=np.complex64)
    out = _boxcar_multilook(arr, (4, 8))
    np.testing.assert_allclose(out, val, rtol=1e-5)


def test_boxcar_non_divisible_truncates():
    # 101 cols with rg_looks=40 → floor(101/40)=2 output cols, not ceil
    arr = np.ones((40, 101))
    out = _boxcar_multilook(arr, (4, 40))
    assert out.shape == (10, 2)


# ---------------------------------------------------------------------------
# _gaussian_multilook
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("looks", [(4, 8), (2, 4)])
def test_gaussian_output_shape_matches_boxcar(looks):
    arr = RNG.standard_normal((40, 80)).astype(np.float32)
    assert _gaussian_multilook(arr, looks).shape == _boxcar_multilook(arr, looks).shape


def test_gaussian_smooths_constant_real():
    # Zero-padding at edges attenuates edge pixels, so only check interior.
    arr = np.full((80, 160), 2.5, dtype=np.float32)
    out = _gaussian_multilook(arr, (4, 8))
    interior = out[3:-3, 3:-3]
    np.testing.assert_allclose(interior, 2.5, atol=0.05)


def test_gaussian_smooths_constant_complex():
    val = np.complex64(1.0 + 0.5j)
    arr = np.full((80, 160), val, dtype=np.complex64)
    out = _gaussian_multilook(arr, (4, 8))
    interior = out[3:-3, 3:-3]
    np.testing.assert_allclose(interior.real, val.real, atol=0.05)
    np.testing.assert_allclose(interior.imag, val.imag, atol=0.05)


def test_gaussian_non_divisible_truncates():
    # Regression: P2 fix — 101 cols with rg=40 must not produce 3 output cols.
    arr = np.ones((40, 101), dtype=np.float32)
    out = _gaussian_multilook(arr, (4, 40))
    assert out.shape == (10, 2)


# ---------------------------------------------------------------------------
# _output_geotransform — regression for P1 (no half-window origin shift)
# ---------------------------------------------------------------------------


def test_output_geotransform_origin_unchanged():
    gt = [100.0, 10.0, 0.0, 500.0, 0.0, -10.0]
    out = _output_geotransform(gt, az_looks=10, rg_looks=40)
    assert out[0] == 100.0, "x0 must not shift"
    assert out[3] == 500.0, "y0 must not shift"


def test_output_geotransform_spacing_scaled():
    gt = [0.0, 10.0, 0.0, 0.0, 0.0, -10.0]
    out = _output_geotransform(gt, az_looks=10, rg_looks=40)
    assert out[1] == 400.0  # dx * rg_looks
    assert out[5] == -100.0  # dy * az_looks


# ---------------------------------------------------------------------------
# _compute_coherence
# ---------------------------------------------------------------------------


def test_coherence_identical_slcs():
    # ref == sec → phase cancels, coherence == 1
    slc = _make_slc((10, 20))
    power = (slc.real**2 + slc.imag**2).astype(np.float32)
    ifg = slc * np.conj(slc)
    coh = _compute_coherence(ifg, power, power)
    np.testing.assert_allclose(coh, 1.0, atol=1e-5)


def test_coherence_zero_power():
    ifg = np.zeros((4, 4), dtype=np.complex64)
    power = np.zeros((4, 4), dtype=np.float32)
    coh = _compute_coherence(ifg, power, power)
    assert (coh == 0.0).all()


def test_coherence_clipped_to_unit_interval():
    # Feed deliberately large values; output must stay in [0, 1].
    ifg = np.full((4, 4), 1e10 + 0j, dtype=np.complex64)
    power = np.ones((4, 4), dtype=np.float32)
    coh = _compute_coherence(ifg, power, power)
    assert (coh >= 0.0).all() and (coh <= 1.0).all()


# ---------------------------------------------------------------------------
# run_crossmul — integration (real GeoTIFF SLCs, subdataset="")
# ---------------------------------------------------------------------------


def test_run_crossmul_output_files_created(slc_geotiffs, tmp_path):
    ref, sec, _ = slc_geotiffs
    opts = CrossmulOptions(looks=(4, 8), lines_per_block=5)
    ifg_p, coh_p = run_crossmul(
        ref,
        sec,
        tmp_path / "pair",
        date1="20200101",
        date2="20200113",
        options=opts,
        subdataset="",
    )
    assert ifg_p.exists()
    assert coh_p.exists()


def test_run_crossmul_output_shape(slc_geotiffs, tmp_path):
    ref, sec, _ = slc_geotiffs
    opts = CrossmulOptions(looks=(4, 8), lines_per_block=5)
    ifg_p, coh_p = run_crossmul(
        ref,
        sec,
        tmp_path / "pair",
        date1="20200101",
        date2="20200113",
        options=opts,
        subdataset="",
    )
    with rasterio.open(ifg_p) as ds:
        assert ds.height == 40 // 4
        assert ds.width == 80 // 8

    with rasterio.open(coh_p) as ds:
        assert ds.height == 40 // 4
        assert ds.width == 80 // 8


def test_run_crossmul_geotransform_no_origin_shift(slc_geotiffs, tmp_path):
    ref, sec, _ = slc_geotiffs
    opts = CrossmulOptions(looks=(4, 8))
    ifg_p, _ = run_crossmul(
        ref,
        sec,
        tmp_path / "pair",
        date1="20200101",
        date2="20200113",
        options=opts,
        subdataset="",
    )
    with rasterio.open(ref) as src:
        ref_origin = (src.transform.c, src.transform.f)
    with rasterio.open(ifg_p) as dst:
        out_origin = (dst.transform.c, dst.transform.f)
    assert ref_origin == pytest.approx(
        out_origin
    ), "origin must not shift after multilooking"


def test_run_crossmul_phase_recovery(slc_geotiffs, tmp_path):
    ref, sec, known_phase = slc_geotiffs
    opts = CrossmulOptions(looks=(4, 8), lines_per_block=5)
    ifg_p, coh_p = run_crossmul(
        ref,
        sec,
        tmp_path / "pair",
        date1="20200101",
        date2="20200113",
        options=opts,
        subdataset="",
    )
    from dolphin.io import load_gdal

    ifg = load_gdal(ifg_p)
    coh = load_gdal(coh_p)

    phase = np.angle(ifg[coh > 0.5])
    # run_crossmul computes ref * conj(sec), so phase = phi_ref - phi_sec = -known_phase
    np.testing.assert_allclose(phase, -known_phase, atol=0.05)


def test_run_crossmul_skip_existing(slc_geotiffs, tmp_path):
    ref, sec, _ = slc_geotiffs
    opts = CrossmulOptions(looks=(4, 8))
    pair_dir = tmp_path / "pair"

    ifg_p, coh_p = run_crossmul(
        ref,
        sec,
        pair_dir,
        date1="20200101",
        date2="20200113",
        options=opts,
        subdataset="",
    )
    mtime_ifg = ifg_p.stat().st_mtime
    mtime_coh = coh_p.stat().st_mtime

    # Second call without overwrite must not touch the files.
    run_crossmul(
        ref,
        sec,
        pair_dir,
        date1="20200101",
        date2="20200113",
        options=opts,
        subdataset="",
        overwrite=False,
    )
    assert ifg_p.stat().st_mtime == mtime_ifg
    assert coh_p.stat().st_mtime == mtime_coh


def test_run_crossmul_overwrite_refreshes(slc_geotiffs, tmp_path):
    ref, sec, _ = slc_geotiffs
    opts = CrossmulOptions(looks=(4, 8))
    pair_dir = tmp_path / "pair"

    ifg_p, _ = run_crossmul(
        ref,
        sec,
        pair_dir,
        date1="20200101",
        date2="20200113",
        options=opts,
        subdataset="",
    )
    mtime_first = ifg_p.stat().st_mtime

    import time

    time.sleep(0.05)  # ensure mtime can change

    run_crossmul(
        ref,
        sec,
        pair_dir,
        date1="20200101",
        date2="20200113",
        options=opts,
        subdataset="",
        overwrite=True,
    )
    assert ifg_p.stat().st_mtime > mtime_first
