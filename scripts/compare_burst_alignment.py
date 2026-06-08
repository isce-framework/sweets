"""Stitch per-burst IFGs with and without burst alignment, then plot a comparison.

Writes:
    <base>/interferograms/stitched_no_align/   — plain merge_by_date
    <base>/interferograms/stitched_aligned/    — with sweets burst alignment
    <base>/burst_align_comparison.png          — full-frame side-by-side wrapped phase
    <base>/burst_align_seam_zoom.png           — zoomed seam + difference panel
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from dolphin.io import load_gdal
from dolphin.stitching import merge_by_date
from opera_utils import group_by_date

from sweets._burst_alignment import align_bursts

BASE = Path(
    "/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/sweets-testing/sweets-f16941"
)
IFG_DIR = BASE / "interferograms"
# Number of pairs to show in the full-frame comparison figure.
N_SHOW = 4
# Half-height (rows) of the seam zoom window.
SEAM_HALF_HEIGHT = 60


def _collect_burst_ifgs() -> list[Path]:
    return sorted(p for p in IFG_DIR.glob("*/*/*_ifg.tif") if "stitched" not in p.parts)


def _stitched_tifs(out_dir: Path) -> list[Path]:
    return sorted(p for p in out_dir.glob("*.tif") if "burst_aligned" not in str(p))


def stitch_no_align(ifg_files: list[Path], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    merge_by_date(ifg_files, output_dir=out_dir, driver="GTiff")
    return _stitched_tifs(out_dir)


def stitch_aligned(ifg_files: list[Path], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    align_dir = out_dir / "burst_aligned"
    align_dir.mkdir(exist_ok=True)

    aligned: list[Path] = []
    for _dates, group in group_by_date(ifg_files).items():
        out, _ = align_bursts(group, align_dir, degree=0)
        aligned.extend(out)

    merge_by_date(aligned, output_dir=out_dir, driver="GTiff")
    return _stitched_tifs(out_dir)


def _wrapped_phase(ifg_path: Path) -> np.ndarray:
    arr = load_gdal(ifg_path)
    phase = np.angle(arr)
    phase[arr == 0] = np.nan
    return phase


def _wrap(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2 * np.pi) - np.pi


def _find_seam_rows(diff: np.ndarray, n_seams: int = 3) -> list[int]:
    """Return row indices with the highest column-wise phase variance in diff."""
    valid = np.where(~np.isnan(diff), diff, 0.0)
    # Row-wise absolute mean — seams appear as sudden jumps so use abs mean.
    row_signal = np.nanmean(np.abs(valid), axis=1)
    # Smooth to avoid picking two adjacent rows from the same seam.
    kernel = np.ones(SEAM_HALF_HEIGHT) / SEAM_HALF_HEIGHT
    smoothed = np.convolve(row_signal, kernel, mode="same")
    # Find local maxima separated by at least SEAM_HALF_HEIGHT rows.
    peaks = []
    min_sep = SEAM_HALF_HEIGHT * 2
    for _ in range(n_seams):
        r = int(np.argmax(smoothed))
        peaks.append(r)
        lo = max(0, r - min_sep)
        hi = min(len(smoothed), r + min_sep)
        smoothed[lo:hi] = 0.0
    return sorted(peaks)


def plot_comparison(
    no_align_ifgs: list[Path],
    aligned_ifgs: list[Path],
    out_png: Path,
    n_show: int = N_SHOW,
) -> None:
    indices = np.linspace(0, len(no_align_ifgs) - 1, n_show, dtype=int)
    pairs_no = [no_align_ifgs[i] for i in indices]
    stem_to_aligned = {p.stem: p for p in aligned_ifgs}
    pairs_al = [stem_to_aligned[p.stem] for p in pairs_no if p.stem in stem_to_aligned]
    pairs_no = [p for p in pairs_no if p.stem in stem_to_aligned]

    fig, axes = plt.subplots(
        n_show, 2, figsize=(12, 3.5 * n_show), constrained_layout=True
    )
    if n_show == 1:
        axes = axes[np.newaxis, :]

    for row, (p_no, p_al) in enumerate(zip(pairs_no, pairs_al)):
        dates = p_no.stem
        for col, (path, label) in enumerate(
            [(p_no, "no alignment"), (p_al, "burst aligned")]
        ):
            phase = _wrapped_phase(path)
            ax = axes[row, col]
            row_im = ax.imshow(
                phase, cmap="RdBu", vmin=-np.pi, vmax=np.pi, aspect="auto"
            )
            ax.set_title(f"{dates}  [{label}]", fontsize=9)
            ax.axis("off")
            if col == 1:
                fig.colorbar(row_im, ax=axes[row, :], label="phase (rad)", shrink=0.6)

    fig.suptitle("Burst alignment comparison — wrapped phase", fontsize=11)
    fig.savefig(out_png, dpi=150)
    print(f"Saved {out_png}")


def plot_seam_zoom(
    no_align_ifgs: list[Path],
    aligned_ifgs: list[Path],
    out_png: Path,
    pair_index: int = 0,
    n_seams: int = 3,
) -> None:
    """Plot zoomed windows around detected burst seams for one pair.

    Each seam row shows: no-align | aligned | wrapped difference (aligned - no-align).
    """
    stem_to_aligned = {p.stem: p for p in aligned_ifgs}
    p_no = no_align_ifgs[pair_index]
    p_al = stem_to_aligned[p_no.stem]
    dates = p_no.stem

    phase_no = _wrapped_phase(p_no)
    phase_al = _wrapped_phase(p_al)
    diff = _wrap(phase_al - phase_no)

    seam_rows = _find_seam_rows(diff, n_seams=n_seams)
    n_rows = len(seam_rows)
    nrows_img, ncols_img = phase_no.shape

    fig, axes = plt.subplots(
        n_rows, 3, figsize=(14, 4 * n_rows), constrained_layout=True
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, sr in enumerate(seam_rows):
        r0 = max(0, sr - SEAM_HALF_HEIGHT)
        r1 = min(nrows_img, sr + SEAM_HALF_HEIGHT)

        crops = {
            "no alignment": phase_no[r0:r1, :],
            "burst aligned": phase_al[r0:r1, :],
            "difference (aligned - no-align)": diff[r0:r1, :],
        }
        cmaps = ["RdBu", "RdBu", "PiYG"]
        vlims = [(-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)]

        for col, (label, crop) in enumerate(crops.items()):
            ax = axes[row, col]
            im = ax.imshow(
                crop,
                cmap=cmaps[col],
                vmin=vlims[col][0],
                vmax=vlims[col][1],
                aspect="auto",
            )
            ax.set_title(label, fontsize=9)
            ax.axhline(SEAM_HALF_HEIGHT, color="yellow", lw=0.8, ls="--", alpha=0.7)
            ax.axis("off")
            fig.colorbar(im, ax=ax, label="rad", shrink=0.7, pad=0.02)

        axes[row, 0].set_ylabel(f"seam ~row {sr}", fontsize=8)

    fig.suptitle(
        f"Seam zoom — {dates}  (rows ±{SEAM_HALF_HEIGHT} around seam centre)",
        fontsize=11,
    )
    fig.savefig(out_png, dpi=150)
    print(f"Saved {out_png}")


if __name__ == "__main__":
    ifg_files = _collect_burst_ifgs()
    print(
        f"Found {len(ifg_files)} per-burst IFGs across {len(group_by_date(ifg_files))} pairs"
    )

    no_align_dir = IFG_DIR / "stitched_no_align"
    aligned_dir = IFG_DIR / "stitched_aligned"

    print("Stitching without alignment...")
    no_align_ifgs = stitch_no_align(ifg_files, no_align_dir)
    print(f"  -> {len(no_align_ifgs)} stitched IFGs in {no_align_dir}")

    print("Stitching with burst alignment...")
    aligned_ifgs = stitch_aligned(ifg_files, aligned_dir)
    print(f"  -> {len(aligned_ifgs)} stitched IFGs in {aligned_dir}")

    plot_comparison(no_align_ifgs, aligned_ifgs, BASE / "burst_align_comparison.png")
    plot_seam_zoom(no_align_ifgs, aligned_ifgs, BASE / "burst_align_seam_zoom.png")
