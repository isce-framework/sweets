# Running the IFG Workflow: Test Guide

Quick reference for testing the `IfgWorkflow` on OPERA CSLC data.

## 1. Create a config YAML

The minimum config for an OPERA CSLC run:

```yaml
work_dir: /path/to/output/dir
bbox:
  - -118.2295   # xmin (west)
  - 34.7939     # ymin (south)
  - -116.5642   # xmax (east)
  - 35.8195     # ymax (north)

search:
  kind: opera-cslc
  out_dir: data
  bbox: [-118.2295, 34.7939, -116.5642, 35.8195]
  start: "2025-04-01T00:00:00"
  end: "2025-06-01T00:00:00"
  relativeOrbit: 64
  # Optional: pin burst IDs instead of deriving from bbox each run
  burst_ids:
    - t064_135521_iw1
    - t064_135521_iw2
    # ... (use opera_utils.get_burst_ids_for_frame(frame_id) to list them)

network:
  max_bandwidth: 2   # nearest-2 network (each date connects to 2 neighbors)

crossmul:
  lines_per_block: 512

stitch:
  run_stitch: true
  crop_to_bbox: true
  run_burst_align: true   # remove inter-burst seam artifacts
  burst_align_degree: 0   # 0 = constant offset per burst; 1 = planar ramp

unwrap:
  run_unwrap: false

overwrite: false
```

To get the burst IDs for a known OPERA DISP frame:

```python
from opera_utils import get_burst_ids_for_frame
print(get_burst_ids_for_frame(16941))
```

## 2. Run the workflow

```bash
python -m sweets ifg-run sweets_ifg_config.yaml
```

### Re-running from a specific step

```
Step 1 — DEM/watermask download + CSLC download
Step 2 — Geometry stitching (LOS east/north/incidence angles)
Step 3 — Crossmul (per-burst complex IFGs + coherence)
Step 4 — Stitch (burst alignment + merge_by_date + wrapped phase)
Step 5 — Unwrap (snaphu, if enabled)
```

To skip download and re-run from crossmul onward:

```bash
python -m sweets ifg-run sweets_ifg_config.yaml --starting-step 3
```

To re-run only the stitch step (crossmul outputs already exist):

```bash
python -m sweets ifg-run sweets_ifg_config.yaml --starting-step 4
```

When re-running the stitch step, delete the existing stitched outputs first
so `overwrite: false` does not silently reuse them:

```bash
find <work_dir>/interferograms/stitched -maxdepth 1 -name "*.tif" -delete
```

## 3. Output layout

```
<work_dir>/
  dem.tif
  watermask.tif
  geometry/
    los_east.tif  los_north.tif  incidence_angle.tif  ...
  data/
    <CSLCs>
    static_layers/
  interferograms/
    t064_135521_iw1/
      20250403_20250415/
        20250403_20250415_ifg.tif          # complex64 crossmul output
        20250403_20250415_coherence.tif
    ...
    stitched/
      20250403_20250415_ifg.tif            # burst-aligned complex
      20250403_20250415_coherence.tif
      20250403_20250415_wrapped_phase.tif  # np.angle(ifg), derived after stitch
      burst_aligned/
        t064_135521_iw1__20250403_20250415_ifg.aligned.tif
        ...
```

Key design decisions:
- Crossmul saves **complex64** `_ifg.tif` (not wrapped phase) so that
  `merge_by_date` interpolates real+imag independently, avoiding ±π
  discontinuities at burst seams.
- Wrapped phase is derived with `np.angle()` **after** stitching.
- Burst alignment runs **per date pair** (27 bursts each), not across all
  pairs at once — passing all pairs together caused same-geography bursts
  from different epochs to be compared as "overlapping", producing garbage
  corrections.

## 4. Plot and QA

```python
from sweets.plotting import plot_ifg_pairs, save_ifg_qa_metrics
from pathlib import Path

stitch_dir = Path("<work_dir>/interferograms/stitched")

# Thumbnail grid of wrapped phase + coherence
plot_ifg_pairs(stitch_dir, max_pairs=13, output_path="ifgs.png")

# Per-pair coherence metrics + JSON sidecar
save_ifg_qa_metrics(stitch_dir, output_path="qa.png")
```

## 5. Test dataset (frame 16941)

- **Frame**: OPERA DISP frame 16941, T064 ascending, S California / Mojave
- **AOI**: `POLYGON((-118.2295 34.7939,-116.5642 34.7939,-116.5642 35.8195,-118.2295 35.8195,-118.2295 34.7939))`
- **Dates**: 2025-04-01 to 2025-06-01 (13 acquisitions, 6-day repeat)
- **Bursts**: 27 (9 frames × 3 IW subswaths)
- **Data on**: `/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/sweets-testing/sweets-f16941/`
- **Config**: `sweets_ifg_config.yaml` in the same directory
