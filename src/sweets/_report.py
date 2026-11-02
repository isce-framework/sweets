"""Single-file HTML report generator for a finished sweets run.

Walks a work directory's ``dolphin/`` output tree and renders a
self-contained HTML document with:

- A summary table (AOI, date range, source, track/frame, wall time)
- The velocity raster
- The temporal-coherence raster (dolphin's quality map)
- The longest-baseline cumulative-displacement raster
- A histogram of per-pair coherence means
- A file inventory of everything in ``dolphin/``

All raster images are rendered to PNG with matplotlib and embedded as
base64 data URIs so the resulting HTML is fully portable — no external
image files, no JS, no network fetches at view time.
"""

from __future__ import annotations

import base64
import html
import io
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from sweets.core import Workflow

__all__ = ["build_report"]


# ---------------------------------------------------------------------------
# Section dataclass + top-level entry point
# ---------------------------------------------------------------------------


@dataclass
class _Section:
    title: str
    body_html: str


def build_report(
    config_file: Path,
    output: Path | None = None,
) -> Path:
    """Render an HTML report for a completed sweets run.

    Parameters
    ----------
    config_file
        Path to the ``sweets_config.yaml`` used for the run. A work
        directory containing a ``sweets_config.yaml`` is also accepted
        and resolved to the yaml inside it.
    output
        Where to write the report. Defaults to
        ``<work_dir>/sweets_report.html``.

    Returns
    -------
    Path
        The report path.
    """
    from sweets.core import Workflow

    config_file = Path(config_file).resolve()
    if config_file.is_dir():
        config_file = config_file / "sweets_config.yaml"
    assert config_file.is_file(), f"Config file not found: {config_file}"

    workflow = Workflow.from_yaml(config_file)
    work_dir = Path(workflow.work_dir).resolve()
    dolphin_dir = work_dir / "dolphin"
    assert dolphin_dir.exists(), f"No dolphin/ under {work_dir}"

    if output is None:
        output = work_dir / "sweets_report.html"

    sections: list[_Section] = []
    sections.append(_build_header(workflow, config_file))
    sections.append(_build_raster_section(dolphin_dir, "velocity"))
    sections.append(_build_raster_section(dolphin_dir, "temporal_coherence"))
    sections.append(_build_raster_section(dolphin_dir, "longest_displacement"))
    sections.append(_build_coherence_histogram(dolphin_dir))
    sections.append(_build_inventory(dolphin_dir))

    title = f"sweets report — {work_dir.name}"
    output.write_text(_render_html(title, sections))
    logger.info(f"Wrote sweets report to {output}")
    return output


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------


def _build_header(workflow: Workflow, config_path: Path) -> _Section:
    rows: list[tuple[str, str]] = [
        ("Work directory", str(workflow.work_dir)),
        ("Config file", str(config_path)),
        ("Generated", datetime.now().isoformat(timespec="seconds")),
    ]

    if workflow.bbox:
        rows.append(("AOI (bbox)", _fmt_bbox(workflow.bbox)))
    elif workflow.wkt:
        rows.append(("AOI (WKT)", _truncate(str(workflow.wkt), 90)))

    search = workflow.search
    rows.append(("Source", search.kind))
    # `model_dump(exclude_none=True)` gives us a uniform view across the
    # discriminated-union variants so we don't have to hard-code which
    # optional fields each Search subclass carries.
    search_dict = search.model_dump(exclude_none=True)
    for k in ("track", "frame", "frequency", "polarizations", "swaths", "burst_ids"):
        v = search_dict.get(k)
        if v is not None:
            rows.append((f"search.{k}", str(v)))

    rows.append(("Date range", f"{search.start.date()} to {search.end.date()}"))

    wall = _wall_time(workflow.work_dir)
    if wall is not None:
        rows.append(("Wall time (est.)", f"{wall}"))

    dolphin_version = _dolphin_version()
    if dolphin_version:
        rows.append(("dolphin version", dolphin_version))

    table = "<table class='kv'>\n"
    for k, v in rows:
        table += f"  <tr><th>{html.escape(k)}</th>" f"<td>{html.escape(v)}</td></tr>\n"
    table += "</table>"
    return _Section("Summary", table)


def _build_raster_section(dolphin_dir: Path, kind: str) -> _Section:
    """Build a section for one of: velocity, temporal_coherence, longest_displacement."""
    ts_dir = dolphin_dir / "timeseries"
    ifg_dir = dolphin_dir / "interferograms"

    path: Path | None = None
    title = kind
    cbar_label = ""
    cmap = "RdBu_r"
    diverging = True
    clip_nodata_zero = False
    fixed_vlim: tuple[float, float] | None = None

    if kind == "velocity":
        path = ts_dir / "velocity.tif"
        title = "Velocity"
        cbar_label = "m / yr"
        clip_nodata_zero = True
    elif kind == "temporal_coherence":
        # There's one per compressed-SLC ministack; take the first.
        tc_paths = sorted(ifg_dir.glob("temporal_coherence_*.tif"))
        path = tc_paths[0] if tc_paths else None
        title = "Temporal coherence"
        cbar_label = "coherence"
        cmap = "viridis"
        diverging = False
        # Coherence is bounded to [0, 1]; pin the color range so a
        # mostly-high-coherence AOI doesn't collapse the colorbar.
        fixed_vlim = (0.0, 1.0)
    elif kind == "longest_displacement":
        longest = _longest_timeseries_pair(ts_dir)
        if longest:
            path, d1, d2 = longest
            title = f"Cumulative displacement: {d1.strftime('%Y-%m-%d')} → {d2.strftime('%Y-%m-%d')}"
            cbar_label = "m"

    if path is None or not path.exists():
        return _Section(
            title.replace("_", " ").title(),
            _note(f"No raster found for `{kind}`; skipping."),
        )

    try:
        data, bounds, nodata, unit = _read_raster(path, default_unit=cbar_label)
    except Exception as e:
        return _Section(title, _note(f"Could not read {path.name}: {e}"))

    import numpy as np

    if clip_nodata_zero and nodata is not None and nodata == 0:
        data = np.where(data == 0, np.nan, data)

    png = _render_raster_png(
        data,
        bounds=bounds,
        cmap=cmap,
        diverging=diverging,
        cbar_label=unit,
        title=path.name,
        fixed_vlim=fixed_vlim,
    )
    stats = _summary_stats(data)
    body = (
        _img(png, alt=title)
        + "<p class='caption'>"
        + f"<b>{html.escape(path.name)}</b>"
        + f" — min {stats['min']:.4f}, max {stats['max']:.4f},"
        + f" mean {stats['mean']:.4f}, std {stats['std']:.4f} ({html.escape(unit)}),"
        + f" {stats['n_valid']} valid pixels"
        + "</p>"
    )
    return _Section(title, body)


def _build_coherence_histogram(dolphin_dir: Path) -> _Section:
    ifg_dir = dolphin_dir / "interferograms"
    cor_paths = sorted(ifg_dir.glob("*.int.cor.tif"))
    if not cor_paths:
        return _Section(
            "Per-pair coherence",
            _note("No per-pair coherence rasters found; skipping."),
        )

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return _Section(
            "Per-pair coherence",
            _note("matplotlib not available; cannot render."),
        )

    means: list[tuple[str, float]] = []
    for p in cor_paths:
        try:
            arr, _, _, _ = _read_raster(p)
        except Exception as e:
            logger.warning(f"Could not read {p.name}: {e}; skipping")
            continue
        vals = np.asarray(arr).ravel()
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if vals.size == 0:
            continue
        label = p.name.replace(".int.cor.tif", "")
        means.append((label, float(vals.mean())))

    if not means:
        return _Section(
            "Per-pair coherence",
            _note("All coherence rasters were empty or masked."),
        )

    labels = [m[0] for m in means]
    values = [m[1] for m in means]

    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.35 * len(labels) + 1.5)))
    y = np.arange(len(labels))
    ax.barh(y, values, color="#4c72b0")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("mean coherence")
    ax.set_title(f"{len(labels)} interferogram pairs")
    ax.axvline(0.3, color="#999", linewidth=0.8, linestyle="--", label="0.3")
    ax.axvline(0.5, color="#666", linewidth=0.8, linestyle="--", label="0.5")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    png = _fig_to_png(fig)
    plt.close(fig)

    table = "<table class='pairs'>\n<tr><th>pair</th><th>mean coherence</th></tr>\n"
    for label, v in means:
        table += f"<tr><td>{html.escape(label)}</td><td>{v:.3f}</td></tr>\n"
    table += "</table>"
    return _Section("Per-pair coherence", _img(png, alt="coherence bar chart") + table)


def _build_inventory(dolphin_dir: Path) -> _Section:
    interesting_globs = [
        ("dolphin/timeseries/", "*.tif"),
        ("dolphin/interferograms/", "*.tif"),
        ("dolphin/unwrapped/", "*.tif"),
    ]
    rows = []
    for rel_dir, pattern in interesting_globs:
        sub = dolphin_dir / Path(rel_dir).name
        if not sub.exists():
            continue
        for p in sorted(sub.glob(pattern)):
            rows.append((p.relative_to(dolphin_dir.parent), p.stat().st_size))

    if not rows:
        return _Section(
            "Output inventory",
            _note("No raster outputs found under dolphin/."),
        )

    # Group by parent dir for readability
    out = "<table class='inv'>\n"
    out += "<tr><th>file</th><th align='right'>size</th></tr>\n"
    current_parent: str | None = None
    for rel, size in rows:
        parent = str(rel.parent)
        if parent != current_parent:
            out += (
                f"<tr class='parent'><td colspan='2'>{html.escape(parent)}/"
                "</td></tr>\n"
            )
            current_parent = parent
        out += (
            f"<tr><td>&nbsp;&nbsp;{html.escape(rel.name)}</td>"
            f"<td align='right'>{_fmt_bytes(size)}</td></tr>\n"
        )
    out += "</table>"
    return _Section(f"Output inventory ({len(rows)} files)", out)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_PAIR_RE = re.compile(r"^(\d{8})_(\d{8})\.tif$")


def _read_raster(
    path: Path, *, default_unit: str = ""
) -> tuple[Any, tuple[float, float, float, float], float | None, str]:
    """Read a single-band raster via GDAL and return (data, bounds, nodata, unit).

    Uses GDAL directly because rasterio's ``_band_dtype`` map doesn't
    handle ``GDT_Float16`` (code 15), which dolphin uses for a few
    outputs (notably the temporal coherence raster).
    """
    import numpy as np
    from osgeo import gdal

    gdal.UseExceptions()
    ds = gdal.Open(str(path))
    if ds is None:
        msg = f"gdal.Open returned None for {path}"
        raise RuntimeError(msg)
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()
    if arr is None:
        msg = f"ReadAsArray returned None for {path}"
        raise RuntimeError(msg)
    data = np.asarray(arr, dtype=np.float64)
    # Mask the file's nodata to NaN (if declared)
    nodata = band.GetNoDataValue()
    if nodata is not None and np.isfinite(nodata):
        data[data == nodata] = np.nan
    gt = ds.GetGeoTransform()
    xs = ds.RasterXSize
    ys = ds.RasterYSize
    # Corner convention: (x_origin, dx, 0, y_origin, 0, dy) with dy typically negative.
    left = gt[0]
    top = gt[3]
    right = left + gt[1] * xs
    bottom = top + gt[5] * ys
    if bottom > top:
        bottom, top = top, bottom
    unit = (band.GetUnitType() or default_unit or "").strip()
    ds = None
    return data, (left, bottom, right, top), nodata, unit


def _longest_timeseries_pair(ts_dir: Path) -> tuple[Path, datetime, datetime] | None:
    """Return the `YYYYMMDD_YYYYMMDD.tif` timeseries step with the widest baseline."""
    best: tuple[Path, datetime, datetime] | None = None
    best_span = -1
    for p in ts_dir.glob("*.tif"):
        m = _PAIR_RE.match(p.name)
        if not m:
            continue
        d1 = datetime.strptime(m.group(1), "%Y%m%d")
        d2 = datetime.strptime(m.group(2), "%Y%m%d")
        span = (d2 - d1).days
        if span > best_span:
            best_span = span
            best = (p, d1, d2)
    return best


def _fmt_bbox(bbox: Any) -> str:
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        return f"({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]})"
    return str(bbox)


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 3] + "..."


def _wall_time(work_dir: Path) -> str | None:
    """Estimate wall time from log mtimes in work_dir / dolphin / ."""
    dolphin_dir = work_dir / "dolphin"
    candidates = list(dolphin_dir.rglob("*.tif"))
    if not candidates:
        return None
    mtimes = [p.stat().st_mtime for p in candidates]
    span = max(mtimes) - min(mtimes)
    if span < 60:
        return f"{span:.0f} s"
    if span < 3600:
        return f"{span / 60:.1f} min"
    return f"{span / 3600:.1f} h"


def _dolphin_version() -> str | None:
    try:
        import dolphin

        return getattr(dolphin, "__version__", None)
    except Exception:
        return None


def _summary_stats(arr: Any) -> dict[str, float]:
    import numpy as np

    vals = np.asarray(arr).ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {
            "min": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "std": float("nan"),
            "n_valid": 0,
        }
    return {
        "min": float(vals.min()),
        "max": float(vals.max()),
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "n_valid": int(vals.size),
    }


def _render_raster_png(
    data: Any,
    bounds: tuple[float, float, float, float],
    *,
    cmap: str,
    diverging: bool,
    cbar_label: str,
    title: str,
    fixed_vlim: tuple[float, float] | None = None,
) -> str:
    import matplotlib.pyplot as plt
    import numpy as np

    vmin: float | None
    vmax: float | None
    if fixed_vlim is not None:
        vmin, vmax = fixed_vlim
    else:
        finite = np.asarray(data).ravel()
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            vmin = vmax = None
        elif diverging:
            lim = float(np.nanpercentile(np.abs(finite), 98))
            vmin, vmax = -lim, lim
        else:
            vmin = float(np.nanpercentile(finite, 2))
            vmax = float(np.nanpercentile(finite, 98))

    fig, ax = plt.subplots(figsize=(7, 5.5), constrained_layout=True)
    im = ax.imshow(
        data,
        extent=(bounds[0], bounds[2], bounds[1], bounds[3]),
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        aspect="equal",
    )
    ax.set_xlabel("easting (m)")
    ax.set_ylabel("northing (m)")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.85, label=cbar_label)
    png = _fig_to_png(fig)
    plt.close(fig)
    return png


def _fig_to_png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _img(png_b64: str, alt: str = "") -> str:
    return f'<img src="data:image/png;base64,{png_b64}"' f' alt="{html.escape(alt)}"/>'


def _note(text: str) -> str:
    return f"<p class='note'>{html.escape(text)}</p>"


def _fmt_bytes(size: int) -> str:
    amount = float(size)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if amount < 1024:
            return f"{amount:.1f} {unit}"
        amount /= 1024
    return f"{amount:.1f} TiB"


# ---------------------------------------------------------------------------
# HTML render
# ---------------------------------------------------------------------------

_STYLE = """
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
                 Helvetica, Arial, sans-serif;
    max-width: 900px;
    margin: 2rem auto;
    padding: 0 1rem;
    color: #222;
    line-height: 1.45;
}
h1 { font-size: 1.6rem; margin-bottom: 0.2rem; }
h2 {
    font-size: 1.2rem;
    border-bottom: 1px solid #ccc;
    padding-bottom: 0.3rem;
    margin-top: 2rem;
}
table { border-collapse: collapse; margin: 0.5rem 0; }
table.kv th { text-align: left; padding: 0.2rem 0.8rem 0.2rem 0; color: #555; }
table.kv td { font-family: monospace; padding: 0.2rem 0; }
table.pairs, table.inv {
    font-size: 0.85rem;
    font-family: monospace;
    margin-top: 0.5rem;
}
table.pairs th, table.pairs td,
table.inv   th, table.inv   td {
    padding: 0.15rem 0.6rem 0.15rem 0;
}
table.pairs th, table.inv th { color: #555; border-bottom: 1px solid #ccc; }
tr.parent td { color: #666; padding-top: 0.5rem; }
img { max-width: 100%; height: auto; }
p.caption { font-size: 0.85rem; color: #555; margin-top: 0.2rem; }
p.note { color: #a33; font-style: italic; }
footer {
    margin-top: 3rem;
    color: #888;
    font-size: 0.8rem;
    border-top: 1px solid #eee;
    padding-top: 0.6rem;
}
"""


def _render_html(title: str, sections: list[_Section]) -> str:
    esc_title = html.escape(title)
    parts: list[str] = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8"/>',
        f"<title>{esc_title}</title>",
        f"<style>{_STYLE}</style>",
        "</head>",
        "<body>",
        f"<h1>{esc_title}</h1>",
    ]
    for sec in sections:
        parts.append(f"<h2>{html.escape(sec.title)}</h2>")
        parts.append(sec.body_html)
    parts.append(
        "<footer>Generated by <code>sweets report</code>"
        f" at {datetime.now().isoformat(timespec='seconds')}.</footer>"
    )
    parts.append("</body></html>")
    return "\n".join(parts)
