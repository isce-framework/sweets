"""Tests for sweets._burst_alignment.

Pure math helpers are tested without I/O.
Integration tests use synthetic overlapping GeoTIFFs written to tmp_path.
"""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from sweets._burst_alignment import (
    BurstCorrection,
    _aggregate_pair,
    _apply_correction,
    _solve_offset_lsq,
    _valid_mask,
    _weighted_median,
    _wrap,
    _wrap_scalar,
    align_bursts,
    apply_burst_offsets,
    estimate_burst_offsets,
)

# ---------------------------------------------------------------------------
# helpers / fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)
_CRS = rasterio.CRS.from_epsg(32610)

# 100m pixels; A covers y=[0,10000], B covers y=[-5000,5000].
# Overlap y=[0,5000] → bottom 50 rows of A and top 50 rows of B.
_TRANSFORM_A = from_bounds(0, 0, 10_000, 10_000, 100, 100)
_TRANSFORM_B = from_bounds(0, -5_000, 10_000, 5_000, 100, 100)
_KNOWN_OFFSET = 0.35  # radians (well below pi, suitable for wrapped data too)


def _write_tif(path, data, transform, crs=_CRS, nodata=None):
    dtype = "complex64" if np.iscomplexobj(data) else "float32"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data, 1)


@pytest.fixture
def burst_tifs_real(tmp_path):
    """Two overlapping unwrapped-phase GeoTIFFs with a known constant offset in B."""
    data_a = np.ones((100, 100), dtype=np.float32)
    data_b = np.ones((100, 100), dtype=np.float32)
    # Top 50 rows of B spatially overlap bottom 50 rows of A.
    data_b[:50, :] = data_a[50:, :] + _KNOWN_OFFSET

    pa = tmp_path / "burst_a.tif"
    pb = tmp_path / "burst_b.tif"
    _write_tif(pa, data_a, _TRANSFORM_A)
    _write_tif(pb, data_b, _TRANSFORM_B)
    return pa, pb


@pytest.fixture
def burst_tifs_complex(tmp_path):
    """Two overlapping wrapped-phase GeoTIFFs with a known constant phase offset in B."""
    data_a = np.ones((100, 100), dtype=np.complex64)
    data_b = np.ones((100, 100), dtype=np.complex64)
    data_b[:50, :] = data_a[50:, :] * np.exp(1j * _KNOWN_OFFSET).astype(np.complex64)

    pa = tmp_path / "burst_a.tif"
    pb = tmp_path / "burst_b.tif"
    _write_tif(pa, data_a, _TRANSFORM_A)
    _write_tif(pb, data_b, _TRANSFORM_B)
    return pa, pb


# ---------------------------------------------------------------------------
# BurstCorrection
# ---------------------------------------------------------------------------


def test_burst_correction_evaluate_constant():
    corr = BurstCorrection(offset=0.3, x_ref=0.0, y_ref=0.0)
    x = np.array([1.0, 2.0])
    y = np.array([3.0, 4.0])
    np.testing.assert_allclose(corr.evaluate(x, y), 0.3)


def test_burst_correction_evaluate_planar():
    corr = BurstCorrection(offset=0.1, cx=0.01, cy=0.02, x_ref=0.0, y_ref=0.0)
    assert corr.evaluate(np.array([100.0]), np.array([0.0]))[0] == pytest.approx(
        0.1 + 0.01 * 100.0
    )


def test_burst_correction_is_planar():
    assert not BurstCorrection(offset=1.0).is_planar
    assert BurstCorrection(cx=0.001).is_planar
    assert BurstCorrection(cy=0.001).is_planar


# ---------------------------------------------------------------------------
# _wrap_scalar / _wrap
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "x, expected",
    [
        (0.0, 0.0),
        (np.pi, -np.pi),  # boundary wraps to -pi
        (np.pi + 0.1, -np.pi + 0.1),
        (-np.pi - 0.1, np.pi - 0.1),
        (2 * np.pi, 0.0),
    ],
)
def test_wrap_scalar(x, expected):
    assert _wrap_scalar(x) == pytest.approx(expected, abs=1e-10)


def test_wrap_array_in_range():
    x = np.linspace(-4 * np.pi, 4 * np.pi, 200)
    w = _wrap(x)
    assert (w >= -np.pi).all() and (w < np.pi).all()


# ---------------------------------------------------------------------------
# _weighted_median
# ---------------------------------------------------------------------------


def test_weighted_median_equal_weights():
    # _weighted_median uses continuous CDF interpolation, which gives the
    # midpoint between the two central values for an even-count uniform sample.
    v = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    w = np.ones(5)
    result = _weighted_median(v, w)
    # Interp at CDF=0.5 lands between the 2nd (0.4) and 3rd (0.6) values → 2.5
    assert result == pytest.approx(2.5, abs=0.1)


def test_weighted_median_heavy_tail():
    v = np.array([1.0, 2.0, 10.0])
    w = np.array([1.0, 1.0, 100.0])
    assert _weighted_median(v, w) > 5.0  # pulled toward the heavy-weight sample


# ---------------------------------------------------------------------------
# _aggregate_pair
# ---------------------------------------------------------------------------


def test_aggregate_pair_mean_real():
    v = np.array([0.2, 0.3, 0.4])
    w = np.ones(3)
    assert _aggregate_pair(v, w, method="mean", is_complex=False) == pytest.approx(0.3)


def test_aggregate_pair_mean_complex_recovers_phase():
    phase = 0.5
    v = np.array([phase, phase, phase])
    w = np.ones(3)
    assert _aggregate_pair(v, w, method="mean", is_complex=True) == pytest.approx(
        phase, abs=1e-5
    )


def test_aggregate_pair_median_real():
    v = np.array([0.1, 0.3, 10.0])  # outlier at 10.0
    w = np.ones(3)
    result = _aggregate_pair(v, w, method="median", is_complex=False)
    assert abs(result - 0.3) < 0.5  # median ignores the outlier


# ---------------------------------------------------------------------------
# _valid_mask
# ---------------------------------------------------------------------------


def test_valid_mask_real_nodata():
    arr = np.array([1.0, np.nan, 0.0, 2.0])
    v = _valid_mask(arr, nodata=0.0)
    assert list(v) == [True, False, False, True]


def test_valid_mask_complex_excludes_zero():
    arr = np.array([1 + 0j, 0 + 0j, 1 + 1j, np.nan + 0j], dtype=np.complex64)
    v = _valid_mask(arr, nodata=None)
    assert list(v) == [True, False, True, False]


# ---------------------------------------------------------------------------
# _solve_offset_lsq
# ---------------------------------------------------------------------------


def test_solve_offset_lsq_two_bursts():
    offset = 0.4
    # Burst 1 is +offset ahead of burst 0.
    pair_eqs: list[tuple[int, int, float, float]] = [(0, 1, offset, 1.0)]
    deltas = _solve_offset_lsq(pair_eqs, n_bursts=2, anchor_index=0, is_complex=False)
    assert deltas[0] == pytest.approx(0.0, abs=1e-6)
    assert deltas[1] == pytest.approx(offset, abs=1e-4)


def test_solve_offset_lsq_three_bursts_chain():
    # 0 → 1 (offset 0.3), 1 → 2 (offset -0.2). Anchor = 0.
    pair_eqs: list[tuple[int, int, float, float]] = [
        (0, 1, 0.3, 1.0),
        (1, 2, -0.2, 1.0),
    ]
    deltas = _solve_offset_lsq(pair_eqs, n_bursts=3, anchor_index=0, is_complex=False)
    assert deltas[0] == pytest.approx(0.0, abs=1e-5)
    assert deltas[1] == pytest.approx(0.3, abs=1e-4)
    assert deltas[2] == pytest.approx(0.1, abs=1e-4)


def test_solve_offset_lsq_no_equations_returns_zeros():
    deltas = _solve_offset_lsq([], n_bursts=3, anchor_index=0, is_complex=False)
    np.testing.assert_array_equal(deltas, 0.0)


# ---------------------------------------------------------------------------
# _apply_correction
# ---------------------------------------------------------------------------


def test_apply_correction_real_removes_offset():
    arr = np.full((10, 10), 2.0, dtype=np.float32)
    corr = BurstCorrection(offset=0.5)
    transform = _TRANSFORM_A
    out = _apply_correction(arr, corr, nodata=None, transform=transform)
    np.testing.assert_allclose(out, 1.5, atol=1e-6)


def test_apply_correction_complex_removes_phase():
    phase = 0.4
    arr = np.exp(1j * phase).astype(np.complex64) * np.ones(
        (10, 10), dtype=np.complex64
    )
    corr = BurstCorrection(offset=phase)
    out = _apply_correction(arr, corr, nodata=None, transform=_TRANSFORM_A)
    np.testing.assert_allclose(np.angle(out), 0.0, atol=1e-5)


def test_apply_correction_preserves_nodata():
    arr = np.array([[1.0, 0.0, 2.0]], dtype=np.float32)  # 0.0 is nodata
    corr = BurstCorrection(offset=0.5)
    out = _apply_correction(arr, corr, nodata=0.0, transform=_TRANSFORM_A)
    assert out[0, 1] == 0.0  # nodata pixel unchanged


# ---------------------------------------------------------------------------
# estimate_burst_offsets — integration
# ---------------------------------------------------------------------------


def test_estimate_offsets_real_recovers_constant(burst_tifs_real):
    pa, pb = burst_tifs_real
    corrections = estimate_burst_offsets([pa, pb], degree=0)
    assert corrections[pa].offset == pytest.approx(0.0, abs=1e-4)
    assert corrections[pb].offset == pytest.approx(_KNOWN_OFFSET, abs=0.05)


def test_estimate_offsets_complex_recovers_phase(burst_tifs_complex):
    pa, pb = burst_tifs_complex
    corrections = estimate_burst_offsets([pa, pb], degree=0)
    assert corrections[pa].offset == pytest.approx(0.0, abs=1e-4)
    assert corrections[pb].offset == pytest.approx(_KNOWN_OFFSET, abs=0.05)


def test_estimate_offsets_single_file_returns_zero(burst_tifs_real):
    pa, _ = burst_tifs_real
    corrections = estimate_burst_offsets([pa], degree=0)
    assert corrections[pa].offset == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# apply_burst_offsets — integration
# ---------------------------------------------------------------------------


def test_apply_offsets_zeroes_seam(burst_tifs_real, tmp_path):
    pa, pb = burst_tifs_real
    corrections = estimate_burst_offsets([pa, pb], degree=0)
    out_dir = tmp_path / "aligned"
    apply_burst_offsets([pa, pb], corrections, out_dir)

    pa_out = out_dir / "burst_a.aligned.tif"
    pb_out = out_dir / "burst_b.aligned.tif"
    assert pa_out.exists()
    assert pb_out.exists()

    # Read the corrected data and verify the overlap values match.
    with rasterio.open(pa_out) as src:
        a_corrected = src.read(1)
    with rasterio.open(pb_out) as src:
        b_corrected = src.read(1)

    # Bottom 50 rows of A and top 50 rows of B overlap spatially.
    # After correction their values should be close.
    np.testing.assert_allclose(a_corrected[50:, :], b_corrected[:50, :], atol=0.1)


# ---------------------------------------------------------------------------
# align_bursts — end-to-end
# ---------------------------------------------------------------------------


def test_align_bursts_returns_corrected_paths(burst_tifs_real, tmp_path):
    pa, pb = burst_tifs_real
    out_paths, corrections = align_bursts([pa, pb], tmp_path / "aligned")
    assert len(out_paths) == 2
    assert all(p.exists() for p in out_paths)
    assert corrections[pb].offset == pytest.approx(_KNOWN_OFFSET, abs=0.05)
