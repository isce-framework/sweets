"""Sensor source models: raw S1 bursts, pre-made OPERA CSLCs, or NISAR GSLCs.

Four source classes are exposed; all are :class:`YamlModel` Pydantic models
with a similar external shape (an AOI, an optional track, ``out_dir``) so
:class:`sweets.core.Workflow` can swap between them:

- :class:`BurstSearch` — wraps :func:`burst2safe.burst2stack.burst2stack`
  to download burst-trimmed ``.SAFE`` directories that the rest of the
  workflow then geocodes via COMPASS. Default; works anywhere S1 flies.
- :class:`LocalSafeSearch` — points at a user-supplied directory of
  pre-downloaded full-frame Sentinel-1 ``.SAFE`` directories or ``.zip``
  archives (e.g. fetched directly from ASF Vertex). Skips the download
  step entirely and feeds the files straight into COMPASS, picking
  ``using_zipped`` automatically based on what's on disk.
- :class:`OperaCslcSearch` — wraps :func:`opera_utils.download.download_cslcs`
  + :func:`opera_utils.download.download_cslc_static_layers` to grab
  pre-geocoded OPERA CSLC HDF5s + their static layers from ASF DAAC. Skips
  COMPASS entirely; locked to OPERA's 5 m x 10 m posting; CONUS-friendly
  but coverage depends on what OPERA has actually produced for the AOI.
- :class:`NisarGslcSearch` — wraps :func:`opera_utils.nisar.run_download`
  to grab pre-geocoded NISAR GSLC HDF5s (L-band, UTM, 5x10 m posting)
  with CMR-based search and optional bbox-level subsetting. Skips COMPASS
  and static layer stitching (NISAR GSLCs have no separate static layers
  product). Coverage and availability depend on NISAR's acquisition plan.

Authentication for any network-fetching source relies on a ``~/.netrc``
entry for ``urs.earthdata.nasa.gov``.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Literal, Optional, TypeVar

from dateutil.parser import parse as parse_date
from dolphin.workflows.config import YamlModel
from pydantic import ConfigDict, Field, field_validator, model_validator
from shapely import wkt as shp_wkt
from shapely.geometry import Polygon, box

from loguru import logger

from ._log import log_runtime

_T = TypeVar("_T")


def _call_off_running_loop(fn: Callable[..., _T], *args: Any, **kwargs: Any) -> _T:
    """Call `fn` even when this thread already owns a running event loop.

    `burst2safe` downloads via `asyncio.run()`, which raises `RuntimeError`
    when invoked from a thread that already has a running loop (a Jupyter or
    IPython kernel, `jupyter execute`, etc.). In that case `fn` is run in a
    dedicated worker thread so its `asyncio.run()` gets a fresh loop. With no
    running loop (the normal script/CLI path) `fn` is called directly.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return fn(*args, **kwargs)

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(fn, *args, **kwargs).result()


FlightDirection = Literal["ASCENDING", "DESCENDING"]


class BurstSearch(YamlModel):
    """Sentinel-1 burst search/download configuration.

    Wraps :func:`burst2safe.burst2stack.burst2stack` so the user can pin a
    small AOI (bbox or WKT polygon) plus a date range and a track number,
    and get back ``.SAFE`` directories containing only the bursts that
    intersect the AOI.
    """

    kind: Literal["safe"] = Field(
        default="safe",
        description="Discriminator for the source type. Always `safe`.",
    )
    out_dir: Path = Field(
        Path("data"),
        description="Directory where SAFE directories will be written.",
        validate_default=True,
    )
    bbox: Optional[tuple[float, float, float, float]] = Field(
        None,
        description=(
            "Area of interest as (left, bottom, right, top) in decimal degrees."
            " Either `bbox` or `wkt` must be set."
        ),
    )
    wkt: Optional[str] = Field(
        None,
        description=(
            "Area of interest as a WKT polygon string (or path to a `.wkt` file)."
            " Takes precedence over `bbox` if both are provided."
        ),
    )
    start: datetime = Field(
        ...,
        description="Search start time (parsed by `dateutil.parser`).",
    )
    end: datetime = Field(
        default_factory=datetime.now,
        description="Search end time. Defaults to now.",
    )
    track: Optional[int] = Field(
        None,
        alias="relativeOrbit",
        description="Sentinel-1 relative orbit / track number.",
    )
    flight_direction: Optional[FlightDirection] = Field(
        None,
        alias="flightDirection",
        description="Restrict to ASCENDING or DESCENDING acquisitions.",
    )
    polarizations: list[str] = Field(
        default_factory=lambda: ["VV"],
        description="Polarizations to include (e.g. ['VV'], ['VV', 'VH']).",
    )
    swaths: Optional[list[str]] = Field(
        None,
        description=(
            "Restrict to specific subswaths (e.g. ['IW2']). If None, all swaths"
            " covering the AOI are downloaded."
        ),
    )
    min_bursts: int = Field(
        1,
        description="Minimum number of bursts a SAFE must contain to be kept.",
        ge=1,
    )
    all_anns: bool = Field(
        True,
        description=(
            "Include annotations for all swaths in the produced SAFE files."
            " Required by `s1-reader` / COMPASS, which always reads the IW2"
            " annotation regardless of the subswath being processed."
        ),
    )

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("start", "end", mode="before")
    @classmethod
    def _parse_datetime(cls, v: Any) -> datetime:
        if isinstance(v, datetime):
            return v
        if isinstance(v, date):
            return datetime.combine(v, datetime.min.time())
        return parse_date(str(v))

    @field_validator("out_dir")
    @classmethod
    def _absolute_out_dir(cls, v: Path) -> Path:
        return Path(v).expanduser().resolve()

    @field_validator("flight_direction", mode="before")
    @classmethod
    def _normalize_flight_direction(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        s = str(v).upper()
        if s.startswith("A"):
            return "ASCENDING"
        if s.startswith("D"):
            return "DESCENDING"
        msg = f"Unrecognized flight direction: {v!r}"
        raise ValueError(msg)

    @field_validator("polarizations")
    @classmethod
    def _upper_pols(cls, v: list[str]) -> list[str]:
        return [p.upper() for p in v]

    @field_validator("swaths")
    @classmethod
    def _upper_swaths(cls, v: Optional[list[str]]) -> Optional[list[str]]:
        return [s.upper() for s in v] if v else v

    @model_validator(mode="after")
    def _check_aoi_and_dates(self) -> "BurstSearch":
        if not self.wkt and not self.bbox:
            msg = "Must provide either `bbox` or `wkt`"
            raise ValueError(msg)
        if self.wkt and Path(self.wkt).exists():
            self.wkt = Path(self.wkt).read_text().strip()
        if self.wkt:
            try:
                shp_wkt.loads(self.wkt)
            except Exception as e:
                msg = f"Invalid WKT polygon: {e}"
                raise ValueError(msg) from e
        if self.end < self.start:
            msg = f"`end` ({self.end}) must be after `start` ({self.start})"
            raise ValueError(msg)
        return self

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def aoi(self) -> Polygon:
        """Return the search AOI as a shapely Polygon."""
        if self.wkt:
            return shp_wkt.loads(self.wkt)
        assert self.bbox is not None  # enforced in validator
        return box(*self.bbox)

    def summary(self) -> str:
        """Return a human-readable summary of the planned search."""
        bounds = self.aoi.bounds
        return (
            "BurstSearch:\n"
            f"  AOI bounds : {bounds}\n"
            f"  Dates      : {self.start.date()} -> {self.end.date()}\n"
            f"  Track      : {self.track}\n"
            f"  Direction  : {self.flight_direction or 'any'}\n"
            f"  Pols       : {self.polarizations}\n"
            f"  Swaths     : {self.swaths or 'any'}\n"
            f"  Output     : {self.out_dir}"
        )

    @log_runtime
    def download(self) -> list[Path]:
        """Download bursts covering the AOI as SAFE directories.

        Returns
        -------
        list[Path]
            Paths of the produced ``.SAFE`` directories.

        """
        # Imported lazily so importing this module is cheap and so users
        # without burst2safe still get a clear error.
        from burst2safe.burst2stack import burst2stack

        self.out_dir.mkdir(parents=True, exist_ok=True)
        logger.info(self.summary())

        result = _call_off_running_loop(
            burst2stack,
            rel_orbit=self.track,
            start_date=self.start,
            end_date=self.end,
            extent=self.aoi,
            polarizations=self.polarizations,
            swaths=self.swaths,
            min_bursts=self.min_bursts,
            all_anns=self.all_anns,
            work_dir=self.out_dir,
        )
        safes = sorted(Path(p) for p in result)
        logger.info(f"Downloaded {len(safes)} SAFE directories to {self.out_dir}")
        if self.flight_direction is not None:
            safes = _filter_by_flight_direction(safes, self.flight_direction)
        return safes

    def existing_safes(self) -> list[Path]:
        """Return any SAFEs already present in `out_dir` (does not query ASF)."""
        return sorted(self.out_dir.glob("S1[AB]_*.SAFE"))


def _filter_by_flight_direction(
    safes: list[Path], flight_direction: FlightDirection
) -> list[Path]:
    """Drop SAFEs whose first manifest does not match `flight_direction`.

    burst2safe does not expose a flight-direction filter directly. We can
    cheaply infer it from the manifest.safe inside the .SAFE bundle.
    """
    import xml.etree.ElementTree as ET

    keep: list[Path] = []
    for s in safes:
        manifest = s / "manifest.safe"
        if not manifest.exists():
            keep.append(s)
            continue
        try:
            tree = ET.parse(manifest)
        except ET.ParseError as e:
            logger.warning(f"Could not parse {manifest}: {e}; keeping SAFE.")
            keep.append(s)
            continue
        text = ET.tostring(tree.getroot(), encoding="unicode")
        upper = flight_direction.upper()
        if upper in text.upper():
            keep.append(s)
        else:
            logger.info(f"Dropping {s.name}: not {upper}")
    return keep


# ----------------------------------------------------------------------------
# Local SAFE / SAFE-zip source (no download)
# ----------------------------------------------------------------------------


class LocalSafeSearch(YamlModel):
    """Source for pre-downloaded Sentinel-1 ``.SAFE`` directories or ``.zip`` archives.

    Use this when the user already has full-frame S1 SAFE products on disk
    (e.g. fetched directly from ASF Vertex, copied off another machine,
    etc.) and wants sweets to skip the download step and feed the files
    straight into COMPASS. Both unzipped ``.SAFE`` directories and zipped
    ``.zip`` archives are accepted; sweets inspects :attr:`out_dir` and
    picks whichever format is present (preferring ``.SAFE`` when both
    are, which is the faster path for COMPASS / s1-reader).

    No date range is required — sweets uses whatever's in :attr:`out_dir`
    as-is. Provide :attr:`bbox` (or :attr:`wkt`) so the AOI can be cropped
    to the user's study area during geocoding.
    """

    kind: Literal["local"] = Field(
        default="local",
        description="Discriminator for the source type. Always `local`.",
    )
    out_dir: Path = Field(
        ...,
        description=(
            "Directory containing pre-downloaded ``.SAFE`` directories or"
            " ``.zip`` archives. Must already exist and contain at least"
            " one ``S1[AB]_*.SAFE`` or ``S1[AB]_*.zip`` file."
        ),
    )
    bbox: Optional[tuple[float, float, float, float]] = Field(
        None,
        description=(
            "Area of interest as (left, bottom, right, top) in decimal degrees."
            " Either `bbox` or `wkt` must be set."
        ),
    )
    wkt: Optional[str] = Field(
        None,
        description=(
            "Area of interest as a WKT polygon string (or path to a `.wkt` file)."
            " Takes precedence over `bbox` if both are provided."
        ),
    )

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("out_dir")
    @classmethod
    def _absolute_out_dir(cls, v: Path) -> Path:
        return Path(v).expanduser().resolve()

    @model_validator(mode="after")
    def _check_aoi(self) -> "LocalSafeSearch":
        if not self.wkt and not self.bbox:
            msg = "Must provide either `bbox` or `wkt`"
            raise ValueError(msg)
        if self.wkt and Path(self.wkt).exists():
            self.wkt = Path(self.wkt).read_text().strip()
        if self.wkt:
            try:
                shp_wkt.loads(self.wkt)
            except Exception as e:
                msg = f"Invalid WKT polygon: {e}"
                raise ValueError(msg) from e
        return self

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def aoi(self) -> Polygon:
        """Return the AOI as a shapely Polygon."""
        if self.wkt:
            return shp_wkt.loads(self.wkt)
        assert self.bbox is not None
        return box(*self.bbox)

    def existing_safes(self) -> list[Path]:
        """Return ``.SAFE`` dirs in `out_dir`, or ``.zip`` files if no SAFEs.

        ``.SAFE`` directories are preferred when both formats are present
        — COMPASS reads them slightly faster than zips and both formats in
        the same directory typically have matching stems (e.g. leftovers
        from an earlier unzip), so picking the zip in that case would just
        re-read the same product.
        """
        safes = sorted(self.out_dir.glob("S1[AB]_*.SAFE"))
        if safes:
            return safes
        return sorted(self.out_dir.glob("S1[AB]_*.zip"))

    def summary(self) -> str:
        """Return a human-readable summary of the configured source."""
        bounds = self.aoi.bounds
        return (
            "LocalSafeSearch:\n"
            f"  AOI bounds : {bounds}\n"
            f"  Source dir : {self.out_dir}"
        )


# ----------------------------------------------------------------------------
# OPERA CSLC source
# ----------------------------------------------------------------------------


class OperaCslcSearch(YamlModel):
    """Pre-made OPERA CSLC search/download configuration.

    Wraps :func:`opera_utils.download.download_cslcs` and
    :func:`opera_utils.download.download_cslc_static_layers` to fetch
    pre-geocoded OPERA CSLC HDF5s + their per-burst static layers from the
    ASF DAAC. Posting is whatever OPERA produced (currently 5 m x 10 m for
    Sentinel-1 OPERA CSLCs); use :class:`BurstSearch` instead if you need
    a custom posting.
    """

    kind: Literal["opera-cslc"] = Field(
        default="opera-cslc",
        description="Discriminator for the source type. Always `opera-cslc`.",
    )
    out_dir: Path = Field(
        Path("data"),
        description=(
            "Directory where the OPERA CSLC HDF5s and static layers will be"
            " written. Static layers go into a `static_layers/` subdirectory."
        ),
        validate_default=True,
    )
    bbox: Optional[tuple[float, float, float, float]] = Field(
        None,
        description=(
            "Area of interest as (left, bottom, right, top) in decimal degrees."
            " Either `bbox` or `wkt` must be set."
        ),
    )
    wkt: Optional[str] = Field(
        None,
        description=(
            "Area of interest as a WKT polygon string (or path to a `.wkt` file)."
        ),
    )
    start: datetime = Field(
        ...,
        description="Search start time (parsed by `dateutil.parser`).",
    )
    end: datetime = Field(
        default_factory=datetime.now,
        description="Search end time. Defaults to now.",
    )
    track: Optional[int] = Field(
        None,
        alias="relativeOrbit",
        description="Sentinel-1 relative orbit / track number.",
    )
    burst_ids: Optional[list[str]] = Field(
        None,
        description=(
            "Restrict to specific OPERA burst IDs (e.g. ['t078_165573_iw2']);"
            " if None, ASF returns whichever bursts intersect the AOI."
        ),
    )
    max_jobs: int = Field(
        3,
        ge=1,
        description="Concurrent download jobs.",
    )

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    # ------------------------------------------------------------------
    # Validators (mirrors BurstSearch shape)
    # ------------------------------------------------------------------

    @field_validator("start", "end", mode="before")
    @classmethod
    def _parse_datetime(cls, v: Any) -> datetime:
        if isinstance(v, datetime):
            return v
        if isinstance(v, date):
            return datetime.combine(v, datetime.min.time())
        return parse_date(str(v))

    @field_validator("out_dir")
    @classmethod
    def _absolute_out_dir(cls, v: Path) -> Path:
        return Path(v).expanduser().resolve()

    @model_validator(mode="after")
    def _check_aoi_and_dates(self) -> "OperaCslcSearch":
        if not self.wkt and not self.bbox:
            msg = "Must provide either `bbox` or `wkt`"
            raise ValueError(msg)
        if self.wkt and Path(self.wkt).exists():
            self.wkt = Path(self.wkt).read_text().strip()
        if self.wkt:
            try:
                shp_wkt.loads(self.wkt)
            except Exception as e:
                msg = f"Invalid WKT polygon: {e}"
                raise ValueError(msg) from e
        if self.end < self.start:
            msg = f"`end` ({self.end}) must be after `start` ({self.start})"
            raise ValueError(msg)
        return self

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def aoi(self) -> Polygon:
        """Return the search AOI as a shapely Polygon."""
        if self.wkt:
            return shp_wkt.loads(self.wkt)
        assert self.bbox is not None
        return box(*self.bbox)

    @property
    def static_layers_dir(self) -> Path:
        """Return the directory where CSLC-STATIC HDF5s live."""
        return self.out_dir / "static_layers"

    def summary(self) -> str:
        """Return a human-readable summary of the planned search."""
        return (
            "OperaCslcSearch:\n"
            f"  AOI bounds : {self.aoi.bounds}\n"
            f"  Dates      : {self.start.date()} -> {self.end.date()}\n"
            f"  Track      : {self.track}\n"
            f"  Burst IDs  : {self.burst_ids or 'auto (from AOI)'}\n"
            f"  Output     : {self.out_dir}"
        )

    def _resolve_burst_ids(self) -> list[str]:
        """Get the list of OPERA burst IDs covering the AOI.

        If the user supplied burst_ids explicitly, use them. Otherwise query
        ASF with the AOI + track to discover them. Querying without an
        explicit list returns one result *per acquisition*, so we
        deduplicate to unique burst IDs before passing to download_cslcs
        (which expects burst IDs and applies the date filter itself).
        """
        if self.burst_ids:
            return list(self.burst_ids)
        from opera_utils.download import search_cslcs

        bounds: tuple[float, float, float, float] = tuple(self.aoi.bounds)  # type: ignore[assignment]
        results = search_cslcs(
            start=self.start,
            end=self.end,
            bounds=bounds,
            track=self.track,
        )
        seen: dict[str, None] = {}
        for r in results:  # type: ignore[union-attr]
            props = getattr(r, "properties", {})
            bid = props.get("operaBurstID") or props.get("burstID")
            if bid:
                seen[bid.lower().replace("-", "_")] = None
        burst_ids = sorted(seen)
        if not burst_ids:
            msg = (
                "No OPERA CSLCs found for the requested AOI / track / dates."
                " Coverage may be missing — fall back to BurstSearch + COMPASS."
            )
            raise RuntimeError(msg)
        logger.info(f"Resolved {len(burst_ids)} OPERA burst IDs from ASF")
        return burst_ids

    @log_runtime
    def download(self) -> list[Path]:
        """Download OPERA CSLC HDF5 files into `out_dir`.

        Returns
        -------
        list[Path]
            Paths to the downloaded ``.h5`` files (one per burst per date).

        """
        from opera_utils.download import download_cslcs

        self.out_dir.mkdir(parents=True, exist_ok=True)
        logger.info(self.summary())

        burst_ids = self._resolve_burst_ids()
        files = download_cslcs(
            burst_ids=burst_ids,
            output_dir=self.out_dir,
            start=self.start,
            end=self.end,
            max_jobs=self.max_jobs,
        )
        files = sorted(Path(f) for f in files)
        logger.info(f"Downloaded {len(files)} OPERA CSLC files to {self.out_dir}")
        return files

    @log_runtime
    def download_static_layers(self) -> list[Path]:
        """Download the CSLC-STATIC HDF5 files for the resolved burst IDs."""
        from opera_utils.download import download_cslc_static_layers

        self.static_layers_dir.mkdir(parents=True, exist_ok=True)
        burst_ids = self._resolve_burst_ids()
        files = download_cslc_static_layers(
            burst_ids=burst_ids,
            output_dir=self.static_layers_dir,
            max_jobs=self.max_jobs,
        )
        files = sorted(Path(f) for f in files)
        logger.info(
            f"Downloaded {len(files)} CSLC-STATIC files to {self.static_layers_dir}"
        )
        return files

    def existing_cslcs(self) -> list[Path]:
        """Return any OPERA CSLC HDF5s already present in `out_dir`."""
        return sorted(self.out_dir.glob("OPERA_L2_CSLC-S1_*.h5"))

    def existing_static_layers(self) -> list[Path]:
        """Return any CSLC-STATIC HDF5s already present in `static_layers_dir`."""
        return sorted(self.static_layers_dir.glob("OPERA_L2_CSLC-S1-STATIC_*.h5"))

    # Mirror BurstSearch.existing_safes for symmetry — used by Workflow to
    # check whether the download step can be skipped.
    def existing_files(self) -> list[Path]:
        return self.existing_cslcs()


# ----------------------------------------------------------------------------
# NISAR GSLC source
# ----------------------------------------------------------------------------


class NisarGslcSearch(YamlModel):
    """Pre-made NISAR GSLC search/download configuration.

    Wraps :func:`opera_utils.nisar.run_download` to search CMR for NISAR
    GSLC products covering the AOI + date range, fetch the matching HDF5s,
    and optionally subset each one to the AOI in a single pass. NISAR
    GSLCs are already geocoded (UTM projection) and have no separate
    "static layers" product, so the downstream workflow skips both COMPASS
    and the geometry stitching step.
    """

    kind: Literal["nisar-gslc"] = Field(
        default="nisar-gslc",
        description="Discriminator for the source type. Always `nisar-gslc`.",
    )
    out_dir: Path = Field(
        Path("data"),
        description="Directory where the NISAR GSLC HDF5s will be written.",
        validate_default=True,
    )
    bbox: Optional[tuple[float, float, float, float]] = Field(
        None,
        description=(
            "Area of interest as (left, bottom, right, top) in decimal degrees."
            " Used for both the CMR query and the bbox subset. Either `bbox`"
            " or `wkt` must be set."
        ),
    )
    wkt: Optional[str] = Field(
        None,
        description=(
            "Area of interest as a WKT polygon string (or path to a `.wkt`"
            " file). Converted to a bbox by `run_download`."
        ),
    )
    start: datetime = Field(
        ...,
        description="Search start time (parsed by `dateutil.parser`).",
    )
    end: datetime = Field(
        default_factory=datetime.now,
        description="Search end time. Defaults to now.",
    )
    track: Optional[int] = Field(
        None,
        alias="relative_orbit_number",
        description=(
            "NISAR relative orbit / track number — the `Track` field on ASF"
            " Vertex, the `RRR` digits in the granule filename. Constant"
            " across repeat passes. Combined with `frame` it pins a single"
            " repeat-pass stack."
        ),
    )
    frame: Optional[int] = Field(
        None,
        alias="track_frame_number",
        description=(
            "NISAR track-frame number — the `Frame` field on ASF Vertex, the"
            " `TTT` digits in the granule filename (e.g. `71`). Constant"
            " across repeat passes."
        ),
    )
    frequency: Optional[Literal["A", "B"]] = Field(
        default=None,
        description=(
            "NISAR frequency band: `A` (L-band primary) or `B`. If left as"
            " the default (None), sweets peeks at the first matching CMR"
            " hit and uses whichever frequency is actually present in the"
            " HDF5. Different NISAR product releases ship different bands"
            " (early BETA was A; recent PR products are B), so guessing is"
            " usually wrong."
        ),
    )
    polarizations: Optional[list[str]] = Field(
        None,
        description=(
            "Polarizations to keep (e.g. ['HH']). If left as the default"
            " (None), sweets uses every polarization present under the"
            " resolved frequency in the first matching CMR hit."
        ),
    )
    short_name: str = Field(
        default="NISAR_L2_GSLC_BETA_V1",
        description="CMR collection short-name to query.",
    )
    num_workers: int = Field(
        default=4,
        ge=1,
        description="Concurrent download jobs.",
    )

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("start", "end", mode="before")
    @classmethod
    def _parse_datetime(cls, v: Any) -> datetime:
        if isinstance(v, datetime):
            return v
        if isinstance(v, date):
            return datetime.combine(v, datetime.min.time())
        return parse_date(str(v))

    @field_validator("out_dir")
    @classmethod
    def _absolute_out_dir(cls, v: Path) -> Path:
        return Path(v).expanduser().resolve()

    @field_validator("polarizations")
    @classmethod
    def _upper_pols(cls, v: Optional[list[str]]) -> Optional[list[str]]:
        return [p.upper() for p in v] if v else v

    @model_validator(mode="after")
    def _check_aoi_and_dates(self) -> "NisarGslcSearch":
        if not self.wkt and not self.bbox:
            msg = "Must provide either `bbox` or `wkt`"
            raise ValueError(msg)
        if self.wkt and Path(self.wkt).exists():
            self.wkt = Path(self.wkt).read_text().strip()
        if self.wkt:
            try:
                shp_wkt.loads(self.wkt)
            except Exception as e:
                msg = f"Invalid WKT polygon: {e}"
                raise ValueError(msg) from e
        if self.end < self.start:
            msg = f"`end` ({self.end}) must be after `start` ({self.start})"
            raise ValueError(msg)
        return self

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def aoi(self) -> Polygon:
        if self.wkt:
            return shp_wkt.loads(self.wkt)
        assert self.bbox is not None
        return box(*self.bbox)

    def hdf5_subdataset(self) -> str:
        """Return the dolphin `input_options.subdataset` path for this config.

        NISAR GSLCs put the complex data at
        ``/science/LSAR/GSLC/grids/frequency{A,B}/{POL}``. If `frequency`
        and `polarizations` are unset, this peeks at the first cached HDF5
        in `out_dir` (or, if there isn't one yet, the first matching CMR
        hit) to learn what's actually in the product.
        """
        freq, pols = self._resolve_frequency_and_pols()
        return f"/science/LSAR/GSLC/grids/frequency{freq}/{pols[0]}"

    def summary(self) -> str:
        return (
            "NisarGslcSearch:\n"
            f"  AOI bounds       : {self.aoi.bounds}\n"
            f"  Dates            : {self.start.date()} -> {self.end.date()}\n"
            f"  Track            : {self.track or 'any'}\n"
            f"  Frame            : {self.frame or 'any'}\n"
            f"  Frequency        : {self.frequency or 'auto'}\n"
            f"  Polarizations    : {self.polarizations or 'auto'}\n"
            f"  CMR short_name   : {self.short_name}\n"
            f"  Output           : {self.out_dir}"
        )

    def _resolve_frequency_and_pols(self) -> tuple[str, list[str]]:
        """Pick the actual `frequency` + polarizations to feed dolphin / run_download.

        Order of preference:

        1. Already-downloaded HDF5 in ``out_dir`` — peek inside.
        2. First CMR hit — open it remotely.

        Returns the user's overrides where they make sense (e.g. they
        asked for `HH` and the file does have it), otherwise falls back
        to whichever frequency / polarization is actually present in
        the HDF5. Logs a warning when the user-requested values don't
        match what's available.
        """
        local = self.existing_files()
        if local:
            freq, pols = _peek_nisar_grid(local[0])
        else:
            from opera_utils.nisar import search

            results = search(
                bbox=tuple(self.aoi.bounds),  # type: ignore[arg-type]
                relative_orbit_number=self.track,
                track_frame_number=self.frame,
                start_datetime=self.start,
                end_datetime=self.end,
                short_name=self.short_name,
            )
            if not results:
                msg = (
                    "No NISAR GSLC products found for the requested AOI /"
                    " track / frame / dates. Cannot resolve frequency."
                )
                raise RuntimeError(msg)
            with results[0]._open() as hf:
                freq, pols = _peek_nisar_grid_from_handle(hf)
        return self._reconcile(freq, pols)

    def _reconcile(
        self, available_freq: str, available_pols: list[str]
    ) -> tuple[str, list[str]]:
        """Reconcile user request against what's actually in the file."""
        if self.frequency and self.frequency != available_freq:
            logger.warning(
                f"NISAR: requested frequency={self.frequency!r} but the"
                f" product only has frequency{available_freq!r}; using"
                f" frequency{available_freq!r}."
            )
        freq = available_freq
        if self.polarizations:
            kept = [p for p in self.polarizations if p in available_pols]
            dropped = [p for p in self.polarizations if p not in available_pols]
            if dropped:
                logger.warning(
                    f"NISAR: requested polarizations {dropped} not present"
                    f" in product (available: {available_pols}); using"
                    f" {kept or available_pols} instead."
                )
            pols = kept or available_pols
        else:
            pols = available_pols
        return freq, pols

    @log_runtime
    def download(self) -> list[Path]:
        """Search + download + bbox-subset NISAR GSLC HDF5s into `out_dir`.

        Per-product peeking: each search result is opened remotely to
        learn its actual ``(frequency, polarizations)`` signature. We
        keep only the largest signature group so dolphin gets a stack
        with a single shared subdataset path; mismatched cycles get
        skipped with a warning. Row/col slices are computed per product
        (not once for the whole stack) so cycles whose grid origins
        don't perfectly line up still get a valid subset.
        """
        from opera_utils.nisar import search
        from opera_utils.nisar._download import process_file

        self.out_dir.mkdir(parents=True, exist_ok=True)
        logger.info(self.summary())

        results = search(
            bbox=tuple(self.aoi.bounds),  # type: ignore[arg-type]
            relative_orbit_number=self.track,
            track_frame_number=self.frame,
            start_datetime=self.start,
            end_datetime=self.end,
            short_name=self.short_name,
        )
        if not results:
            msg = (
                "No NISAR GSLC products found for the requested AOI /"
                " track / frame / dates."
            )
            raise RuntimeError(msg)

        groups = _group_nisar_results_by_signature(results)
        logger.info(
            f"NISAR found {len(results)} hit(s) across "
            f"{len(groups)} (frequency, polarization) signature(s):"
        )
        for sig, items in groups.items():
            logger.info(f"  frequency{sig[0]}/{sorted(sig[1])}: {len(items)} cycle(s)")

        ranked = self._rank_signatures(groups)
        bounds: tuple[float, float, float, float] = tuple(self.aoi.bounds)  # type: ignore[assignment]

        downloaded: list[Path] = []
        attempted: list[str] = []
        for sig in ranked:
            chosen = groups[sig]
            chosen_freq, chosen_pols = self._reconcile(sig[0], sorted(sig[1]))
            logger.info(
                f"NISAR trying frequency={chosen_freq},"
                f" polarizations={chosen_pols} ({len(chosen)} cycle(s));"
                f" {len(results) - len(chosen)} cycle(s) belong to other"
                " signatures."
            )
            attempted.append(f"frequency{chosen_freq}/{chosen_pols}")

            group_outputs = self._download_group(
                chosen=chosen,
                chosen_freq=chosen_freq,
                chosen_pols=chosen_pols,
                bounds=bounds,
                process_file=process_file,
            )
            if group_outputs:
                downloaded.extend(group_outputs)
                break
            logger.warning(
                f"NISAR: signature frequency{chosen_freq}/{chosen_pols} produced"
                " no usable GeoTIFFs (bbox likely outside the actual data"
                " extent); falling back to next signature."
            )

        downloaded.sort()
        if not downloaded:
            msg = (
                f"NISAR: no usable GeoTIFFs for AOI {bounds} after trying"
                f" signatures {attempted}. Either the bbox doesn't actually"
                " intersect any product's grid extent, or the pinned"
                " frequency/polarizations don't match what's available."
                " Try widening the date range or removing the `frequency` /"
                " `polarizations` pins from the config to let sweets"
                " auto-detect."
            )
            raise RuntimeError(msg)
        logger.info(f"Wrote {len(downloaded)} NISAR GSLC GeoTIFFs to {self.out_dir}")
        return downloaded

    def _download_group(
        self,
        chosen: list,  # noqa: ANN001
        chosen_freq: str,
        chosen_pols: list[str],
        bounds: tuple[float, float, float, float],
        process_file,  # noqa: ANN001
    ) -> list[Path]:
        """Download + GeoTIFF-convert every product in one signature group."""
        outputs: list[Path] = []
        for product in chosen:
            short = Path(product.filename).name
            try:
                rows, cols = _get_per_product_rowcol_slice(product, bounds, chosen_freq)
            except Exception as e:
                logger.warning(
                    f"NISAR: failed to compute row/col slice for {short}:"
                    f" {e}; skipping."
                )
                continue
            try:
                h5_path = Path(
                    process_file(
                        url=product.filename,
                        rows=rows,
                        cols=cols,
                        output_dir=self.out_dir,
                        frequency=chosen_freq,
                        polarizations=chosen_pols,
                    )
                )
            except Exception as e:
                logger.warning(f"NISAR: download failed for {short}: {e}; skipping.")
                continue
            # NISAR HDF5s store coordinates as separate xCoordinates /
            # yCoordinates 1D arrays. Complex data makes CF-compliance hard.
            # GDAL's HDF5 driver reads an identity geotransform for the
            # subdataset. Wrap each polarization in a tiny VRT that injects
            # the real geotransform + SRS on top of the HDF5 subdataset —
            # dolphin opens the VRT natively and the HDF5 stays the single
            # source of truth for the pixel values.
            try:
                vrt_paths = _nisar_h5_to_vrts(
                    h5_path=h5_path,
                    frequency=chosen_freq,
                    polarizations=chosen_pols,
                )
            except Exception as e:
                logger.warning(
                    f"NISAR: VRT wrap failed for {h5_path.name}:" f" {e}; skipping."
                )
                continue
            outputs.extend(vrt_paths)
        return outputs

    def _rank_signatures(
        self,
        groups: dict[tuple[str, frozenset[str]], list],
    ) -> list[tuple[str, frozenset[str]]]:
        """Rank (frequency, polarization-set) groups from best to worst.

        Sort key (descending):

        1. Stack size — more cycles is always better for InSAR.
        2. User polarization overlap — pols are what dolphin actually
           consumes, so a VV pin matters more than a frequency pin.
        3. User frequency match — soft preference; sweets will override
           it if the pinned frequency has no data in the AOI.

        The caller iterates this list and stops at the first signature
        whose products actually yield usable GeoTIFFs.
        """
        user_pols = set(self.polarizations or [])
        user_freq = self.frequency

        def key(item: tuple[tuple[str, frozenset[str]], list]) -> tuple[int, int, int]:
            sig, prods = item
            n_cycles = len(prods)
            pol_match = int(bool(user_pols and (user_pols & sig[1])))
            freq_match = int(user_freq is not None and sig[0] == user_freq)
            return (n_cycles, pol_match, freq_match)

        return [sig for sig, _ in sorted(groups.items(), key=key, reverse=True)]

    def existing_files(self) -> list[Path]:
        """Return on-disk NISAR GSLC VRTs (the dolphin-ready wrappers).

        sweets stores the raw subset HDF5s (written by opera-utils'
        `process_file`) alongside a tiny per-polarization VRT wrapper
        that injects the real geotransform + SRS on top of the HDF5
        subdataset. dolphin consumes the VRTs.
        """
        return sorted(self.out_dir.glob("NISAR_L2_*GSLC*.*.vrt"))

    def wavelength(self) -> float:
        """Radar wavelength (m) for the downloaded stack.

        Three-tier strategy, from most authoritative to most generic:

        1. **Runtime read from the HDF5.** NISAR D-102269 §4 specifies a
           Float64 scalar ``centerFrequency`` under each
           ``/science/LSAR/GSLC/grids/frequency{A,B}`` group, carrying
           the exact carrier of the processed image in Hz. When present
           we return ``C / centerFrequency`` directly.
        2. **Filename MODE lookup.** Current BETA PR products don't
           populate ``centerFrequency``, but they do embed the
           acquisition MODE code in slot 8 of the NISAR D-102269 §3.4
           granule filename (e.g. ``4005`` = 40+5 MHz split). Combined
           with the chosen frequency letter (detected from which
           ``frequency{A,B}`` subgroup exists in the downloaded subset),
           ``NISAR_L_MODE_CENTERS_HZ`` from ``dolphin.constants`` gives
           us the exact center frequency for that mode, per Figure 3-1.
        3. **Coarse band fallback.** If neither of the above resolves
           (unrecognized MODE, or the granule doesn't start with
           ``NISAR_L`` / ``NISAR_S``), fall back to the generic band
           constants. ``NISAR_L_WAVELENGTH`` matches the full-band
           77 MHz mode exactly, which is not in the MODE table on
           purpose — the fallback is correct for it.

        The caller forwards the result to
        ``build_displacement_config(wavelength=...)`` so dolphin writes
        timeseries outputs in meters instead of radians.
        """
        import h5py

        from dolphin import constants
        from dolphin.constants import SPEED_OF_LIGHT

        h5_files = sorted(self.out_dir.glob("NISAR_*GSLC*.h5"))
        candidates = h5_files if h5_files else self.existing_files()
        assert candidates, f"No NISAR files in {self.out_dir}; run download() first"

        chosen_letter: Optional[str] = None
        if h5_files:
            with h5py.File(h5_files[0], "r") as hf:
                for freq_letter in ("A", "B"):
                    cf_path = (
                        f"/science/LSAR/GSLC/grids/frequency{freq_letter}"
                        "/centerFrequency"
                    )
                    if cf_path in hf:
                        return SPEED_OF_LIGHT / float(hf[cf_path][()])
                # No centerFrequency → remember which frequency letter is
                # actually in the subset so the MODE-table lookup below
                # can pick the right column.
                for freq_letter in ("A", "B"):
                    if f"/science/LSAR/GSLC/grids/frequency{freq_letter}" in hf:
                        chosen_letter = freq_letter
                        break

        name = candidates[0].name
        parts = name.split("_")
        # NISAR_IL_PT_PROD_CYL_REL_P_FRM_MODE_POLE_... -> MODE is slot 8.
        if len(parts) >= 9 and chosen_letter is not None and parts[1].startswith("L"):
            mode = parts[8]
            centers = constants.NISAR_L_MODE_CENTERS_HZ.get(mode)
            if centers is not None:
                center_hz = centers[0 if chosen_letter == "A" else 1]
                if center_hz is not None:
                    return SPEED_OF_LIGHT / center_hz
                logger.warning(
                    f"NISAR MODE {mode!r} has no frequency{chosen_letter}"
                    " entry in Figure 3-1; falling back to the generic"
                    " L-band wavelength."
                )

        stem = name.upper()
        if stem.startswith("NISAR_L"):
            return constants.NISAR_L_WAVELENGTH
        if stem.startswith("NISAR_S"):
            return constants.NISAR_S_WAVELENGTH
        msg = f"Cannot infer NISAR band from filename {name}"
        raise RuntimeError(msg)


def _group_nisar_results_by_signature(
    results,  # noqa: ANN001
) -> dict[tuple[str, frozenset[str]], list]:
    """Group NISAR search results by (frequency_letter, frozenset(pols)).

    Each result is opened remotely to inspect its actual grid layout.
    Products that fail to peek (e.g. transient network error) are skipped
    with a warning.
    """
    groups: dict[tuple[str, frozenset[str]], list] = {}
    for prod in results:
        short = Path(prod.filename).name
        try:
            with prod._open() as hf:
                freq, pols = _peek_nisar_grid_from_handle(hf)
        except Exception as e:
            logger.warning(f"NISAR: failed to peek {short}: {e}; skipping.")
            continue
        sig = (freq, frozenset(pols))
        groups.setdefault(sig, []).append(prod)
    return groups


def _get_per_product_rowcol_slice(
    product,  # noqa: ANN001
    bbox: tuple[float, float, float, float],
    frequency: str,
) -> tuple[Optional[slice], Optional[slice]]:
    """Compute row/col slices for `bbox` against this product's own grid.

    opera-utils' default `_get_rowcol_slice` uses results[0]'s grid for
    the whole stack, which gives wrong indices when cycles have slightly
    different grid origins. Recomputing per-product is safer.
    """
    west, south, east, north = bbox
    row_start, col_start = product.lonlat_to_rowcol(west, north, frequency)
    row_stop, col_stop = product.lonlat_to_rowcol(east, south, frequency)
    if row_start > row_stop:
        row_start, row_stop = row_stop, row_start
    if col_start > col_stop:
        col_start, col_stop = col_stop, col_start
    return slice(row_start, row_stop), slice(col_start, col_stop)


def _peek_nisar_grid(h5path: Path) -> tuple[str, list[str]]:
    """Open a NISAR GSLC HDF5 and return (frequency_letter, polarizations)."""
    import h5py

    with h5py.File(h5path, "r") as hf:
        return _peek_nisar_grid_from_handle(hf)


def _nisar_h5_to_vrts(
    h5_path: Path,
    frequency: str,
    polarizations: list[str],
) -> list[Path]:
    """Wrap each NISAR GSLC HDF5 polarization in a tiny georeferenced VRT.

    NISAR HDF5s store their grid as separate ``xCoordinates`` /
    ``yCoordinates`` arrays under ``/science/LSAR/GSLC/grids/frequency<X>/``
    instead of CF-compliant CRS metadata, so GDAL's HDF5 driver opens the
    subdataset but reports an identity geotransform. Every downstream tool
    that expects a georeferenced raster (dolphin masking, sweets bounds
    intersection, rasterio) then trips over it.

    Writing a ~1-KB VRT that references the HDF5 subdataset and injects
    the correct ``<SRS>`` + ``<GeoTransform>`` fixes the problem without
    rewriting the ~20 MB of complex data. dolphin opens the VRT natively,
    sees the right grid, and the source HDF5 stays the single source of
    truth for the pixel values.

    Output filenames: ``<h5_stem>.<polarization>.vrt`` next to the source
    HDF5. Existing VRTs are left in place.
    """
    import h5py
    from osgeo import osr

    out_paths: list[Path] = []
    grid_path = f"/science/LSAR/GSLC/grids/frequency{frequency}"
    with h5py.File(h5_path, "r") as hf:
        if grid_path not in hf:
            msg = (
                f"{h5_path.name}: no `{grid_path}` group — likely a"
                " metadata-only stub from a non-intersecting bbox."
            )
            raise RuntimeError(msg)
        grid = hf[grid_path]
        x_coords = grid["xCoordinates"][:]
        y_coords = grid["yCoordinates"][:]
        epsg = int(grid["projection"][()])

        n_cols = len(x_coords)
        n_rows = len(y_coords)
        assert (
            n_cols >= 2 and n_rows >= 2
        ), f"{h5_path.name}: grid is degenerate ({n_rows}x{n_cols})"

        # NISAR coordinates are pixel centers; the VRT geotransform is
        # the top-left corner, so step back by half a pixel.
        dx = float(x_coords[1] - x_coords[0])
        dy = float(y_coords[1] - y_coords[0])  # negative when y decreases (typical)
        x_origin = float(x_coords[0]) - dx / 2
        y_origin = float(y_coords[0]) - dy / 2

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(epsg)
        wkt = srs.ExportToWkt()

        for pol in polarizations:
            if pol not in grid:
                logger.warning(
                    f"{h5_path.name}: polarization {pol} not in HDF5; skipping."
                )
                continue
            out_path = h5_path.with_suffix(f".{pol}.vrt")
            if out_path.exists():
                logger.debug(f"{out_path.name} already exists; skipping.")
                out_paths.append(out_path)
                continue
            vrt_xml = (
                f'<VRTDataset rasterXSize="{n_cols}" rasterYSize="{n_rows}">\n'
                f"  <SRS>{wkt}</SRS>\n"
                f"  <GeoTransform>{x_origin}, {dx}, 0.0,"
                f" {y_origin}, 0.0, {dy}</GeoTransform>\n"
                f'  <VRTRasterBand dataType="CFloat32" band="1">\n'
                "    <SimpleSource>\n"
                '      <SourceFilename relativeToVRT="0">'
                f'HDF5:"{h5_path}"://science/LSAR/GSLC/grids/'
                f"frequency{frequency}/{pol}</SourceFilename>\n"
                "      <SourceBand>1</SourceBand>\n"
                f'      <SrcRect xOff="0" yOff="0" xSize="{n_cols}" ySize="{n_rows}"/>\n'
                f'      <DstRect xOff="0" yOff="0" xSize="{n_cols}" ySize="{n_rows}"/>\n'
                "    </SimpleSource>\n"
                "  </VRTRasterBand>\n"
                "</VRTDataset>\n"
            )
            out_path.write_text(vrt_xml)
            out_paths.append(out_path)
            logger.info(f"Wrote {out_path.name}")
    return out_paths


def _peek_nisar_grid_from_handle(hf) -> tuple[str, list[str]]:  # noqa: ANN001
    """Inspect an open NISAR GSLC HDF5 file handle for grid layout."""
    grids_path = "/science/LSAR/GSLC/grids"
    if grids_path not in hf:
        msg = f"NISAR HDF5 has no `{grids_path}` group"
        raise RuntimeError(msg)
    freq_groups = [k for k in hf[grids_path].keys() if k.startswith("frequency")]
    if not freq_groups:
        msg = f"NISAR HDF5 `{grids_path}` has no frequency subgroup"
        raise RuntimeError(msg)
    # Pick the first available frequency (filename order: A before B).
    freq_groups.sort()
    freq_path = f"{grids_path}/{freq_groups[0]}"
    freq_letter = freq_groups[0].removeprefix("frequency")
    pols = [
        k
        for k in hf[freq_path].keys()
        if k in ("HH", "VV", "HV", "VH", "RH", "RV", "LH", "LV")
    ]
    return freq_letter, pols
