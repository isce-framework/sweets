"""Granule search endpoint.

Returns a GeoJSON ``FeatureCollection`` of granule outlines for the requested
source, AOI, and date range. The frontend overlays these on the Leaflet map
so the user can see exactly which bursts / frames their search will pull in
before kicking off a download.

Three sources are supported, each backed by a small wrapper around an upstream
search call (no new CMR code in sweets):

- ``safe``       : Sentinel-1 SLC bursts via ``asf_search.search``.
- ``opera-cslc`` : OPERA L2_CSLC-S1 bursts via ``opera_utils.download.search_cslcs``.
- ``nisar-gslc`` : NISAR L2_GSLC frames via ``opera_utils.nisar.search``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()


SourceKind = Literal["safe", "opera-cslc", "nisar-gslc"]


class SearchRequest(BaseModel):
    """User-side search parameters from the map / search panel."""

    source: SourceKind
    bbox: tuple[float, float, float, float] = Field(
        ..., description="(left, bottom, right, top) in decimal degrees."
    )
    start: str = Field(..., description="ISO date or datetime.")
    end: str = Field(..., description="ISO date or datetime.")
    track: Optional[int] = None
    frame: Optional[int] = None


def _feature(
    name: str,
    geometry: dict,
    *,
    date: str = "",
    track: Optional[int] = None,
    burst_id: Optional[str] = None,
    frame: Optional[int] = None,
    url: Optional[str] = None,
    flight_direction: Optional[str] = None,
) -> dict:
    return {
        "type": "Feature",
        "properties": {
            "name": name,
            "date": date,
            "track": track,
            "burst_id": burst_id,
            "frame": frame,
            "url": url,
            "flight_direction": _norm_direction(flight_direction),
            # Filled in later by _annotate_coverage; default to True so
            # sources without burst-level coverage (NISAR) still render.
            "in_coverage": True,
        },
        "geometry": geometry,
    }


def _norm_direction(v: Any) -> Optional[str]:
    """Normalize various CMR/ASF flight-direction encodings to ASC/DESC."""
    if v is None:
        return None
    s = str(v).strip().upper()
    if not s:
        return None
    if s.startswith("A"):
        return "ASCENDING"
    if s.startswith("D"):
        return "DESCENDING"
    return s


def _annotate_coverage(features: list[dict]) -> dict[str, Any]:
    """Run the same missing-data analysis sweets uses on actual downloads.

    Mirrors :meth:`sweets.core.Workflow._apply_missing_data_filter` so the
    map preview shows the user, *before* the download, which bursts will
    end up in the chosen consistent subset and which will be dropped. The
    optimization is "maximize total bursts s.t. every chosen burst has
    every chosen date" — see :func:`opera_utils.missing_data.get_missing_data_options`.

    Returns a small summary dict describing the chosen option, or an
    empty dict when coverage analysis isn't applicable (e.g. NISAR, or
    every feature already covers every date).
    """
    from datetime import datetime as _dt

    tuples: list[tuple[str, _dt]] = []
    for f in features:
        p = f["properties"]
        bid = p.get("burst_id")
        date = (p.get("date") or "")[:10]
        if not bid or not date:
            continue
        try:
            tuples.append((bid, _dt.fromisoformat(date)))
        except ValueError:
            continue

    if len(tuples) < 2:
        return {}

    try:
        from opera_utils.missing_data import get_missing_data_options
    except ImportError:
        return {}

    options = get_missing_data_options(burst_id_date_tuples=tuples)
    if not options:
        return {}

    top = options[0]
    chosen_bursts = set(top.burst_ids)
    chosen_dates = {d.date().isoformat() for d in top.dates}

    in_coverage = 0
    for f in features:
        p = f["properties"]
        bid = p.get("burst_id")
        date = (p.get("date") or "")[:10]
        if bid in chosen_bursts and date in chosen_dates:
            p["in_coverage"] = True
            in_coverage += 1
        else:
            p["in_coverage"] = False

    return {
        "num_bursts": len(chosen_bursts),
        "num_dates": len(chosen_dates),
        "num_features_in_coverage": in_coverage,
        "num_features_excluded": len(features) - in_coverage,
        "num_options": len(options),
    }


def _safe_search(req: SearchRequest) -> list[dict]:
    import asf_search as asf

    opts = asf.ASFSearchOptions(
        platform=asf.PLATFORM.SENTINEL1,
        processingLevel=asf.PRODUCT_TYPE.BURST,
        start=req.start,
        end=req.end,
        intersectsWith=_bbox_to_wkt(req.bbox),
    )
    if req.track is not None:
        opts.relativeOrbit = req.track
    results = asf.search(opts=opts)
    out: list[dict] = []
    for r in results:
        geom = r.geometry
        props = r.properties
        out.append(
            _feature(
                name=props.get("sceneName") or props.get("fileID", "?"),
                geometry=geom,
                date=props.get("startTime", ""),
                track=props.get("pathNumber"),
                burst_id=str(props.get("burst", {}).get("fullBurstID", "") or "")
                or None,
                url=props.get("url"),
                flight_direction=props.get("flightDirection"),
            )
        )
    return out


def _opera_cslc_search(req: SearchRequest) -> list[dict]:
    from opera_utils.download import search_cslcs

    results = search_cslcs(
        start=_to_dt(req.start),
        end=_to_dt(req.end),
        bounds=req.bbox,
        track=req.track,
    )
    out: list[dict] = []
    for r in results:
        props = getattr(r, "properties", {}) or {}
        geom = getattr(r, "geometry", None)
        if geom is None:
            continue
        burst_id = props.get("operaBurstID") or props.get("burstID")
        out.append(
            _feature(
                name=props.get("sceneName") or props.get("fileID", burst_id or "?"),
                geometry=geom,
                date=props.get("startTime", ""),
                track=props.get("pathNumber"),
                burst_id=burst_id,
                url=props.get("url"),
                flight_direction=props.get("flightDirection"),
            )
        )
    return out


def _nisar_search(req: SearchRequest) -> list[dict]:
    from opera_utils.nisar import search

    results = search(
        bbox=req.bbox,
        relative_orbit_number=req.track,
        track_frame_number=req.frame,
        start_datetime=_to_dt(req.start),
        end_datetime=_to_dt(req.end),
    )
    out: list[dict] = []
    for r in results:
        # opera_utils.nisar.search may return either CMR JSON-like dicts or
        # asf_search ASFProduct objects depending on opera_utils version.
        # Pull a polygon out of whichever shape we get.
        geom, props = _extract_geom_and_props(r)
        if geom is None:
            continue
        name = props.get("granule_ur") or props.get("sceneName") or "?"
        out.append(
            _feature(
                name=name,
                geometry=geom,
                date=props.get("startTime", "") or props.get("time_start", ""),
                track=props.get("track") or props.get("pathNumber"),
                frame=props.get("frame"),
                url=props.get("url"),
                flight_direction=props.get("flightDirection")
                or props.get("ascendingFlag"),
            )
        )
    return out


def _extract_geom_and_props(r: Any) -> tuple[Optional[dict], dict]:
    """Adapt asf_search.ASFProduct, dict, or umm-json hit to (geometry, properties)."""
    if hasattr(r, "geometry") and hasattr(r, "properties"):
        return r.geometry, dict(getattr(r, "properties") or {})
    if isinstance(r, dict):
        # umm-json hit shape: {"umm": {"SpatialExtent": ...}}
        umm = r.get("umm", {})
        spatial = (
            umm.get("SpatialExtent", {})
            .get("HorizontalSpatialDomain", {})
            .get("Geometry", {})
        )
        polys = spatial.get("GPolygons", [])
        if polys:
            pts = polys[0].get("Boundary", {}).get("Points", [])
            coords = [[p["Longitude"], p["Latitude"]] for p in pts]
            if coords and coords[0] != coords[-1]:
                coords.append(coords[0])
            return {"type": "Polygon", "coordinates": [coords]}, _flatten_umm(umm)
        return None, _flatten_umm(umm)
    return None, {}


def _flatten_umm(umm: dict) -> dict:
    out: dict[str, Any] = {}
    out["granule_ur"] = umm.get("GranuleUR")
    t = umm.get("TemporalExtent", {}).get("RangeDateTime", {})
    out["startTime"] = t.get("BeginningDateTime")
    for attr in umm.get("AdditionalAttributes", []) or []:
        name = attr.get("Name")
        vals = attr.get("Values") or []
        if name and vals:
            out[name] = vals[0]
    return out


def _bbox_to_wkt(bbox: tuple[float, float, float, float]) -> str:
    left, bottom, right, top = bbox
    return (
        f"POLYGON(({left} {bottom}, {right} {bottom}, "
        f"{right} {top}, {left} {top}, {left} {bottom}))"
    )


def _to_dt(s: str) -> datetime:
    from dateutil.parser import parse as parse_date

    return parse_date(s)


@router.post("")
def search_granules(req: SearchRequest) -> dict:
    """Return a GeoJSON FeatureCollection of matching granule outlines.

    Each feature is annotated with ``properties.in_coverage`` so the map
    can show the user which bursts will actually be processed (i.e. the
    largest consistent (burst, date) subset) vs which would be filtered
    out as partial-coverage during the missing-data step. NISAR is granule-
    per-frame and skips this analysis.
    """
    if req.source == "safe":
        features = _safe_search(req)
    elif req.source == "opera-cslc":
        features = _opera_cslc_search(req)
    elif req.source == "nisar-gslc":
        features = _nisar_search(req)
    else:  # pragma: no cover - exhaustive by type
        raise HTTPException(400, f"Unknown source: {req.source}")

    coverage = _annotate_coverage(features) if req.source != "nisar-gslc" else {}

    return {
        "type": "FeatureCollection",
        "features": features,
        "count": len(features),
        "source": req.source,
        "coverage": coverage,
        "tracks": _summarize_tracks(features),
    }


def _summarize_tracks(features: list[dict]) -> list[dict]:
    """Group features by (track, flight_direction) with a granule count.

    The frontend uses this to render a track picker — sweets only downloads
    one track at a time, and the user typically wants to see which tracks
    actually have data over the AOI before narrowing.
    """
    by_key: dict[tuple[Optional[int], Optional[str]], int] = {}
    for f in features:
        p = f["properties"]
        key = (p.get("track"), p.get("flight_direction"))
        by_key[key] = by_key.get(key, 0) + 1
    return [
        {"track": t, "flight_direction": d, "count": n}
        for (t, d), n in sorted(
            by_key.items(),
            key=lambda kv: (-kv[1], kv[0][0] if kv[0][0] is not None else -1),
        )
        if t is not None
    ]
