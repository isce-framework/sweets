"""Stitch per-burst IFGs with and without burst alignment, then plot a comparison.

Writes:
    <base>/interferograms/stitched_no_align/   — plain merge_by_date
    <base>/interferograms/stitched_aligned/    — with sweets burst alignment
    <base>/burst_align_comparison.png          — side-by-side wrapped phase figure
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
# Number of pairs to show in the comparison figure (picks evenly-spaced pairs).
N_SHOW = 4


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


def plot_comparison(
    no_align_ifgs: list[Path],
    aligned_ifgs: list[Path],
    out_png: Path,
    n_show: int = N_SHOW,
) -> None:
    # Pick n_show evenly-spaced pairs from the sorted list.
    indices = np.linspace(0, len(no_align_ifgs) - 1, n_show, dtype=int)
    pairs_no = [no_align_ifgs[i] for i in indices]
    # Match by filename stem (date key).
    stem_to_aligned = {p.stem: p for p in aligned_ifgs}
    pairs_al = [stem_to_aligned[p.stem] for p in pairs_no if p.stem in stem_to_aligned]
    pairs_no = [p for p in pairs_no if p.stem in stem_to_aligned]

    fig, axes = plt.subplots(
        n_show, 2, figsize=(12, 3.5 * n_show), constrained_layout=True
    )
    if n_show == 1:
        axes = axes[np.newaxis, :]

    for row, (p_no, p_al) in enumerate(zip(pairs_no, pairs_al)):
        dates = p_no.stem.replace("_ifg", "")
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
