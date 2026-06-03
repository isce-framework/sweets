"""Generate docs/ifg_workflow.ipynb from source strings."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

OUT = Path(__file__).parent.parent / "docs" / "ifg_workflow.ipynb"


def cell_id() -> str:
    return uuid.uuid4().hex[:16]


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": cell_id(),
        "metadata": {},
        "source": source,
    }


def code(source: str) -> dict:
    # Strip any accidental trailing backslash from raw-string cell sources.
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id(),
        "metadata": {},
        "outputs": [],
        "source": source.rstrip("\\"),
    }


# ── cell sources ──────────────────────────────────────────────────────────────

TITLE = """\
# sweets IFG Workflow — Interactive Notebook

Interactive configuration and step-by-step execution of the `IfgWorkflow`
pipeline, designed for JupyterHub and standalone Jupyter environments.

**What this notebook does**:

1. Let you set up an AOI, date range, and processing options via widgets
2. Build and save a `sweets_ifg_config.yaml`
3. Run each workflow step (download → geometry → crossmul → stitch → unwrap)
4. Display the stitched interferogram thumbnails

Default values are pre-loaded for **OPERA DISP frame 16941** (T064 ascending,
S California / Mojave, April–June 2025).

---
> **Requirements**: `sweets`, `ipywidgets`, `folium` (all in the pixi `default`
> environment). Run with:
> ```
> pixi run jupyter lab
> ```\
"""

IMPORTS = """\
from __future__ import annotations

import io
import sys
from pathlib import Path
from datetime import date

import ipywidgets as w
import folium
from IPython.display import display, clear_output

# Make sure sweets is importable (editable install or src on sys.path)
_src = Path.cwd().parent / "src"
if _src.exists() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))\
"""

SECTION_CONFIGURE = """\
## 1. Configure the Workflow

The tabs below cover every option.  Values are pre-filled for frame 16941 —
change what you need, then proceed to **Section 2** to build the config.\
"""

WIDGETS = """\
# ── Frame 16941 defaults (change _WORK_DIR to your own output directory) ──────
import os as _os
_WORK_DIR = _os.path.join(_os.path.expanduser("~"), "sweets-output", "frame-16941")
_BBOX     = (-118.2295, 34.7939, -116.5642, 35.8195)
_START    = date(2025, 4, 1)
_END      = date(2025, 6, 1)
_TRACK    = 64
_BURST_IDS = "\\n".join([
    "t064_135521_iw1", "t064_135521_iw2", "t064_135521_iw3",
    "t064_135522_iw1", "t064_135522_iw2", "t064_135522_iw3",
    "t064_135523_iw1", "t064_135523_iw2", "t064_135523_iw3",
    "t064_135524_iw1", "t064_135524_iw2", "t064_135524_iw3",
    "t064_135525_iw1", "t064_135525_iw2", "t064_135525_iw3",
    "t064_135526_iw1", "t064_135526_iw2", "t064_135526_iw3",
    "t064_135527_iw1", "t064_135527_iw2", "t064_135527_iw3",
    "t064_135528_iw1", "t064_135528_iw2", "t064_135528_iw3",
    "t064_135529_iw1", "t064_135529_iw2", "t064_135529_iw3",
])

# ── Shared widget styles ──────────────────────────────────────────────────────
_S  = {"description_width": "130px"}
_L  = w.Layout(width="320px")
_LW = w.Layout(width="460px")

# ── Tab 1: AOI ────────────────────────────────────────────────────────────────
w_west  = w.BoundedFloatText(value=_BBOX[0], min=-180, max=180, step=0.0001,
                             description="West (lon)", style=_S, layout=_L)
w_south = w.BoundedFloatText(value=_BBOX[1], min=-90,  max=90,  step=0.0001,
                             description="South (lat)", style=_S, layout=_L)
w_east  = w.BoundedFloatText(value=_BBOX[2], min=-180, max=180, step=0.0001,
                             description="East (lon)", style=_S, layout=_L)
w_north = w.BoundedFloatText(value=_BBOX[3], min=-90,  max=90,  step=0.0001,
                             description="North (lat)", style=_S, layout=_L)

_map_out = w.Output(layout=w.Layout(height="360px"))

def _render_map(*_):
    lat_c = (w_south.value + w_north.value) / 2
    lon_c = (w_west.value  + w_east.value)  / 2
    m = folium.Map(location=[lat_c, lon_c], zoom_start=7)
    folium.Rectangle(
        bounds=[[w_south.value, w_west.value], [w_north.value, w_east.value]],
        color="#4285F4", fill=True, fill_opacity=0.15,
        tooltip=(f"W={w_west.value:.4f}  S={w_south.value:.4f}  "
                 f"E={w_east.value:.4f}  N={w_north.value:.4f}"),
    ).add_to(m)
    with _map_out:
        clear_output(wait=True)
        display(m)

for _wgt in (w_west, w_south, w_east, w_north):
    _wgt.observe(_render_map, names="value")
_render_map()

_blank = w.Label("")
_aoi_compass = w.GridBox(
    [_blank,  w_north, _blank,
     w_west,  _blank,  w_east,
     _blank,  w_south, _blank],
    layout=w.Layout(
        grid_template_rows="auto auto auto",
        grid_template_columns="1fr 1fr 1fr",
        grid_gap="6px",
    ),
)
tab_aoi = w.VBox([
    w.HTML("<h4 style='margin:4px 0'>Bounding box (decimal degrees)</h4>"),
    _aoi_compass,
    w.HTML("<h4 style='margin:8px 0 4px'>Map preview</h4>"),
    _map_out,
])

# ── Tab 2: Search & Download ──────────────────────────────────────────────────
w_source = w.Dropdown(
    options=["opera-cslc", "safe", "nisar-gslc"], value="opera-cslc",
    description="Source type", style=_S, layout=_L,
)
w_start = w.DatePicker(value=_START, description="Start date", style=_S, layout=_L)
w_end   = w.DatePicker(value=_END,   description="End date",   style=_S, layout=_L)
w_track = w.BoundedIntText(
    value=_TRACK, min=1, max=175, step=1,
    description="Rel. orbit", style=_S, layout=_L,
)
w_burst_ids = w.Textarea(
    value=_BURST_IDS,
    description="Burst IDs",
    placeholder="One burst ID per line.  Leave empty to derive from bbox.",
    style={"description_width": "90px"},
    layout=w.Layout(width="500px", height="160px"),
)
w_data_dir = w.Text(value="data", description="Data dir", style=_S, layout=_L)

tab_search = w.VBox([
    w.HTML("<h4 style='margin:4px 0'>Source</h4>"),
    w_source,
    w.HTML("<h4 style='margin:8px 0 4px'>Date range</h4>"),
    w_start, w_end,
    w.HTML("<h4 style='margin:8px 0 4px'>Track & burst selection</h4>"),
    w_track, w_burst_ids,
    w.HTML("<h4 style='margin:8px 0 4px'>Download directory</h4>"),
    w_data_dir,
])

# ── Tab 3: Processing options ─────────────────────────────────────────────────
# Network
w_bandwidth = w.BoundedIntText(
    value=2, min=0, max=20,
    description="Max bandwidth",
    style=_S, layout=_L,
)
w_ref_date = w.Text(
    value="", placeholder="YYYY-MM-DD  (optional)",
    description="Ref. date", style=_S, layout=_L,
)
w_max_tbl = w.BoundedFloatText(
    value=0.0, min=0.0, step=6.0,
    description="Max Δt (days)", style=_S, layout=_L,
)

# Crossmul
w_az_looks = w.BoundedIntText(value=10, min=1, description="Az. looks", style=_S, layout=_L)
w_rg_looks = w.BoundedIntText(value=40, min=1, description="Rg. looks", style=_S, layout=_L)

# Stitch
w_run_stitch  = w.Checkbox(value=True,  description="Run stitch")
w_crop_bbox   = w.Checkbox(value=True,  description="Crop to bbox")
w_burst_align = w.Checkbox(value=True,  description="Burst alignment")
w_align_deg   = w.Dropdown(
    options=[(0, 0), (1, 1)], value=0,
    description="Align degree",
    style=_S, layout=w.Layout(width="240px"),
)

# Unwrap
w_run_unwrap    = w.Checkbox(value=False, description="Run unwrap")
w_unwrap_method = w.Dropdown(
    options=["snaphu", "spurt", "whirlwind"], value="snaphu",
    description="Method", style=_S, layout=_L,
)
w.dlink((w_run_unwrap, "value"), (w_unwrap_method, "disabled"), lambda v: not v)
w_unwrap_method.disabled = True

tab_proc = w.VBox([
    w.HTML("<h4 style='margin:4px 0'>Interferogram network</h4>"),
    w_bandwidth, w_ref_date, w_max_tbl,
    w.HTML("<h4 style='margin:8px 0 4px'>Crossmul (multilooking)</h4>"),
    w.HBox([w_az_looks, w_rg_looks]),
    w.HTML("<h4 style='margin:8px 0 4px'>Stitch</h4>"),
    w.HBox([w_run_stitch, w_crop_bbox, w_burst_align]),
    w_align_deg,
    w.HTML("<h4 style='margin:8px 0 4px'>Unwrap</h4>"),
    w.HBox([w_run_unwrap, w_unwrap_method]),
])

# ── Tab 4: Output ─────────────────────────────────────────────────────────────
w_work_dir = w.Text(
    value=_WORK_DIR,
    description="Work dir",
    style={"description_width": "80px"},
    layout=w.Layout(width="640px"),
)
w_overwrite = w.Checkbox(value=False, description="Overwrite existing outputs")
tab_output = w.VBox([
    w.HTML("<h4 style='margin:4px 0'>Working directory</h4>"),
    w_work_dir, w_overwrite,
])

# ── Tabbed layout ─────────────────────────────────────────────────────────────
_tabs = w.Tab([tab_aoi, tab_search, tab_proc, tab_output])
for _i, _t in enumerate(["AOI", "Search & Download", "Processing", "Output"]):
    _tabs.set_title(_i, _t)

display(_tabs)\
"""

SECTION_CONFIG = """\
## 2. Build & Save Config

Click **Build config** to validate the settings and preview the YAML.
Click **Save YAML** to write `sweets_ifg_config.yaml` inside your work directory.\
"""

CONFIG_BUILDER = """\
import pydantic
from sweets.ifg import IfgWorkflow

_out_cfg = w.Output()
_btn_build = w.Button(description="Build config", button_style="info",  icon="check")
_btn_save  = w.Button(description="Save YAML",    button_style="success", icon="save")

workflow: IfgWorkflow | None = None  # shared state; set by on_build


def _collect() -> dict:
    burst_ids = [b.strip() for b in w_burst_ids.value.strip().splitlines() if b.strip()]
    cfg: dict = {
        "work_dir": w_work_dir.value,
        "bbox": [w_west.value, w_south.value, w_east.value, w_north.value],
        "search": {
            "kind":    w_source.value,
            "out_dir": w_data_dir.value,
            "start":   str(w_start.value),
            "end":     str(w_end.value),
        },
        "network": {"max_bandwidth": w_bandwidth.value},
        "crossmul": {"looks": [w_az_looks.value, w_rg_looks.value]},
        "stitch": {
            "run_stitch":        w_run_stitch.value,
            "crop_to_bbox":      w_crop_bbox.value,
            "run_burst_align":   w_burst_align.value,
            "burst_align_degree": int(w_align_deg.value),
        },
        "unwrap": {
            "run_unwrap":    w_run_unwrap.value,
            "unwrap_method": w_unwrap_method.value,
        },
        "overwrite": w_overwrite.value,
    }
    if w_track.value:
        cfg["search"]["track"] = w_track.value
    if burst_ids:
        cfg["search"]["burst_ids"] = burst_ids
    if w_ref_date.value.strip():
        cfg["network"]["reference_date"] = w_ref_date.value.strip()
    if w_max_tbl.value > 0:
        cfg["network"]["max_temporal_baseline"] = float(w_max_tbl.value)
    return cfg


def _on_build(b):
    global workflow
    _out_cfg.clear_output(wait=True)
    cfg = _collect()
    try:
        workflow = IfgWorkflow(**cfg)
        buf = io.StringIO()
        workflow.to_yaml(buf)
        yaml_str = buf.getvalue()
        with _out_cfg:
            print("Config valid!  YAML preview:\\n" + "─" * 60)
            print(yaml_str)
    except pydantic.ValidationError as exc:
        with _out_cfg:
            print(f"Validation error:\\n{exc}")


def _on_save(b):
    _out_cfg.clear_output(wait=True)
    if workflow is None:
        with _out_cfg:
            print("Click 'Build config' first.")
        return
    cfg_path = Path(w_work_dir.value) / "sweets_ifg_config.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    workflow.save(cfg_path)
    with _out_cfg:
        print(f"Saved: {cfg_path}")


_btn_build.on_click(_on_build)
_btn_save.on_click(_on_save)

# Auto-build on first load so the YAML preview is visible without clicking
_on_build(None)

display(w.HBox([_btn_build, _btn_save]), _out_cfg)\
"""

SECTION_RUN = """\
## 3. Run the Workflow

Select a starting step and click **Run**.  All steps from the selected point
onward will execute in sequence.

| Step | What happens |
|------|-------------|
| 1 — Download | DEM, water mask, OPERA CSLCs |
| 2 — Geometry | Stitch CSLC-STATIC look-angle layers |
| 3 — Crossmul | Per-burst multilooked interferograms |
| 4 — Stitch | Burst-align + merge to single frame |
| 5 — Unwrap | SNAPHU/SPURT (only if `run_unwrap=True`) |

> **Tip**: If data already exists, start from step 3 or 4 to skip the download.\
"""

RUN_CELL = """\
_out_run = w.Output(layout=w.Layout(
    border="1px solid #ccc",
    height="400px",
    overflow_y="scroll",
    padding="6px",
))
_step_sel = w.BoundedIntText(
    value=4, min=1, max=5, step=1,
    description="Starting step",
    style={"description_width": "110px"},
    layout=w.Layout(width="240px"),
)
_btn_run = w.Button(
    description="Run workflow",
    button_style="danger",
    icon="play",
    layout=w.Layout(width="160px"),
)

def _on_run(b):
    _out_run.clear_output(wait=True)
    if workflow is None:
        with _out_run:
            print("Build and save config first (Section 2).")
        return
    _btn_run.disabled = True
    _btn_run.description = "Running..."
    try:
        with _out_run:
            workflow.run(starting_step=_step_sel.value)
    finally:
        _btn_run.disabled = False
        _btn_run.description = "Run workflow"

_btn_run.on_click(_on_run)

display(
    w.VBox([
        w.HBox([_step_sel, _btn_run]),
        w.HTML("<b>Output log</b>"),
        _out_run,
    ])
)\
"""

SECTION_RESULTS = """\
## 4. View Results

After the workflow finishes, run this cell to display wrapped-phase and
coherence thumbnails for all stitched date pairs.\
"""

RESULTS_CELL = """\
import matplotlib
matplotlib.use("Agg")          # safe for JupyterHub environments without a GUI display
import matplotlib.pyplot as plt
from IPython.display import display as _ipy_display
from sweets.plotting import plot_ifg_pairs

_stitch_dir = Path(w_work_dir.value) / "interferograms" / "stitched"

if not _stitch_dir.exists():
    print(f"Stitch directory not found: {_stitch_dir}")
    print("Run the workflow (step 4) first.")
else:
    fig = plot_ifg_pairs(_stitch_dir, max_pairs=13, subsample=3)
    _ipy_display(fig)          # embeds the figure inline (works with Agg backend)
    plt.close(fig)
    print(f"Plotted from: {_stitch_dir}")\
"""

SECTION_ADVANCED = """\
## Appendix: Programmatic Usage

You can also create and run the workflow entirely without the widgets:\
"""

ADVANCED_CELL = """\
# Construct directly from a YAML file saved in Section 2
# (or hand-craft the dict here)

from sweets.ifg import IfgWorkflow

wf = IfgWorkflow.load(
    Path(w_work_dir.value) / "sweets_ifg_config.yaml"
)

# Inspect any field
print("Pairs in network:", wf.network.max_bandwidth, "nearest-neighbor(s)")
print("Crossmul looks:", wf.crossmul.looks)
print("Burst alignment:", wf.stitch.run_burst_align)

# Run from stitch step
# wf.run(starting_step=4)\
"""

# ── assemble notebook ─────────────────────────────────────────────────────────

cells = [
    md(TITLE),
    code(IMPORTS),
    md(SECTION_CONFIGURE),
    code(WIDGETS),
    md(SECTION_CONFIG),
    code(CONFIG_BUILDER),
    md(SECTION_RUN),
    code(RUN_CELL),
    md(SECTION_RESULTS),
    code(RESULTS_CELL),
    md(SECTION_ADVANCED),
    code(ADVANCED_CELL),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11.0",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(f"Wrote {OUT}")
