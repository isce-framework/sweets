"""Interferogram workflow: crossmul + optional unwrap from geocoded SLCs.

An alternative to the full dolphin displacement workflow that produces
multilooked wrapped (and optionally unwrapped) interferograms directly
from a stack of geocoded SLCs.

The search/download/geocode stages are identical to the main
:class:`~sweets.core.Workflow`.  Differences start at step 3: instead of
running dolphin's phase-linking pipeline, this workflow

1. Selects pairs via a nearest-N or single-reference network.
2. Computes a multilooked interferogram + coherence per pair (blockwise,
   pure NumPy / SciPy crossmul — no GPU required).
3. Optionally unwraps with dolphin's SNAPHU / SPURT / WHIRLWIND backends.

Configuration serialises to / loads from ``sweets_ifg_config.yaml``.
"""

from __future__ import annotations

import re
import shutil
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, Optional, Union

from dolphin.utils import set_num_threads
from dolphin.workflows.config import YamlModel
from opera_utils import group_by_burst
from pydantic import ConfigDict, Field, computed_field, field_validator, model_validator
from shapely import wkt as shp_wkt

from loguru import logger

from ._burst_db import get_burst_db
from ._crossmul import CrossmulOptions, run_crossmul
from ._geocode_slcs import create_config_files, run_geocode, run_static_layers
from ._geometry import stitch_geometry
from ._log import log_runtime
from ._netrc import setup_nasa_netrc
from ._orbit import download_orbits
from ._types import Filename
from .dem import create_dem, create_water_mask
from .download import BurstSearch, LocalSafeSearch, NisarGslcSearch, OperaCslcSearch

if TYPE_CHECKING:
    pass

Source = Annotated[
    Union[BurstSearch, LocalSafeSearch, OperaCslcSearch, NisarGslcSearch],
    Field(discriminator="kind"),
]

UnwrapMethod = Literal["snaphu", "spurt", "whirlwind"]


class NetworkOptions(YamlModel):
    """Interferogram network selection options.

    Parameters
    ----------
    max_bandwidth : int | None
        Maximum number of nearest neighbors (by acquisition index) to include
        in the interferogram network.  ``None`` means only nearest-1.
    reference_date : str | None
        Date (``YYYY-MM-DD``) of a common reference SLC.  When set, every
        other SLC is paired against this one in addition to the nearest-N
        network.  Use alone (``max_bandwidth=0``) for a purely single-
        reference (small-baseline-like) network.
    max_temporal_baseline : float | None
        Exclude pairs whose temporal separation exceeds this threshold
        (days).  Applied after the bandwidth and reference-date selection.
    """

    max_bandwidth: int = Field(
        default=1,
        ge=0,
        description=(
            "Nearest-neighbor depth.  1 = sequential pairs only;"
            " 3 = add pairs up to 3 acquisitions apart."
        ),
    )
    reference_date: Optional[str] = Field(
        default=None,
        description=(
            "Date of the common reference SLC (YYYY-MM-DD).  When set, every"
            " other date is also paired to this one regardless of bandwidth."
        ),
    )
    max_temporal_baseline: Optional[float] = Field(
        default=None,
        description=(
            "Drop pairs whose temporal baseline exceeds this value (days)."
            " None (default) applies no temporal-baseline cap."
        ),
    )


class IfgUnwrapOptions(YamlModel):
    """Unwrapping options.

    Parameters
    ----------
    run_unwrap : bool
        If False, only wrapped phase and coherence are produced.
    unwrap_method : UnwrapMethod
        Algorithm: ``"snaphu"`` (default), ``"spurt"``, or ``"whirlwind"``.
    n_parallel_jobs : int
        Interferograms to unwrap concurrently.
    nlooks : float
        Effective number of looks (used by SNAPHU for coherence-based
        cost estimation).  Set to ``az_looks * rg_looks`` when unset.
    snaphu_ntiles : tuple[int, int] | "auto"
        Tile grid for SNAPHU.  ``"auto"`` selects 2x2 for small areas.
    snaphu_tile_overlap : tuple[int, int]
        Row and column overlap between SNAPHU tiles.
    """

    run_unwrap: bool = Field(
        default=True,
        description=(
            "Run the unwrapping step.  Set False to produce only wrapped"
            " phase and coherence."
        ),
    )
    unwrap_method: UnwrapMethod = Field(
        default="snaphu",
        description="Unwrapping algorithm.",
    )
    n_parallel_jobs: int = Field(
        default=2,
        ge=1,
        description="Interferograms to unwrap concurrently.",
    )
    nlooks: Optional[float] = Field(
        default=None,
        description=(
            "Effective number of looks for cost estimation.  Defaults to"
            " ``az_looks * rg_looks`` from the crossmul options."
        ),
    )
    snaphu_ntiles: Union[tuple[int, int], Literal["auto"]] = Field(
        default="auto",
        description="SNAPHU tile grid (rows, cols). 'auto' picks a 2x2 grid.",
    )
    snaphu_tile_overlap: tuple[int, int] = Field(
        default=(128, 128),
        description="SNAPHU tile overlap (rows, cols).",
    )
    snaphu_cost: Literal["defo", "smooth"] = Field(
        default="smooth",
        description="SNAPHU statistical cost mode.",
    )


class StitchOptions(YamlModel):
    """Options for stitching per-burst interferograms into a single output.

    Parameters
    ----------
    run_stitch : bool
        Merge all per-burst wrapped-phase and coherence files for each
        date pair into a single stitched GeoTIFF.
    crop_to_bbox : bool
        Crop the stitched output to the workflow ``bbox``.  Has no effect
        when ``run_stitch`` is False.
    run_burst_align : bool
        Estimate and remove inter-burst phase offsets before stitching via
        a least-squares constant (or planar) correction in each overlap
        region.  Requires ``dolphin.burst_alignment`` (available on the
        ``feat/burst-alignment`` branch of dolphin; not yet on main).
    burst_align_degree : {0, 1}
        Polynomial degree for burst-alignment corrections: 0 = constant
        DC offset per burst, 1 = planar ramp (offset + x-slope + y-slope).
    """

    run_stitch: bool = Field(
        default=True,
        description=(
            "Merge per-burst interferograms into one stitched GeoTIFF per"
            " date pair.  Outputs land in <ifg_dir>/stitched/."
        ),
    )
    crop_to_bbox: bool = Field(
        default=True,
        description="Crop stitched output to the workflow bbox.",
    )
    run_burst_align: bool = Field(
        default=False,
        description=(
            "Estimate and remove inter-burst phase offsets before stitching."
            " Requires dolphin.burst_alignment (feat/burst-alignment branch)."
        ),
    )
    burst_align_degree: Literal[0, 1] = Field(
        default=0,
        description="Burst-alignment polynomial degree: 0=constant, 1=planar ramp.",
    )


class IfgWorkflow(YamlModel):
    """Interferogram-only workflow: geocode + crossmul + stitch (+ optional unwrap).

    Configuration is a strict subset of :class:`~sweets.core.Workflow` with
    the ``dolphin`` displacement block replaced by a lightweight crossmul +
    unwrap section.  The search / download / geocode stages are identical.
    """

    work_dir: Path = Field(
        default_factory=Path.cwd,
        description="Root of working directory for processing.",
        validate_default=True,
    )
    bbox: Optional[tuple[float, float, float, float]] = Field(
        default=None,
        description=(
            "AOI as (left, bottom, right, top) in decimal degrees."
            " Either `bbox` or `wkt` must be set."
        ),
    )
    wkt: Optional[str] = Field(
        default=None,
        description="AOI as a WKT polygon (or path to a `.wkt` file). Overrides bbox.",
    )
    search: Source = Field(
        ...,
        description=(
            "Source of input SLCs — same options as the main Workflow:"
            " BurstSearch, LocalSafeSearch, OperaCslcSearch, NisarGslcSearch."
        ),
    )
    dem_filename: Path = Field(
        default_factory=lambda data: data["work_dir"] / "dem.tif",
        description="DEM in EPSG:4326.  Downloaded via sardem if absent.",
    )
    dem_bbox: Optional[tuple[float, float, float, float]] = Field(
        default=None,
        description=(
            "Optional DEM download extent override (left, bottom, right, top)."
        ),
    )
    water_mask_filename: Path = Field(
        default_factory=lambda data: data["work_dir"] / "watermask.tif",
        description="Water mask (uint8 GTiff, 1=land, 0=water).",
    )
    orbit_dir: Path = Field(
        default=Path("orbits"),
        description="Directory for Sentinel-1 precise orbit files.",
        validate_default=True,
    )
    slc_posting: tuple[float, float] = Field(
        default=(10, 5),
        description="Geocoded SLC posting (y, x) in meters.",
    )
    pol_type: Literal["co-pol", "cross-pol"] = Field(
        default="co-pol",
        description="Polarization type (COMPASS knob).",
    )
    gpu_enabled: bool = Field(
        default=True,
        description="Run COMPASS geocoding on GPU when an isce3-cuda build is available.",
    )

    crossmul: CrossmulOptions = Field(
        default_factory=CrossmulOptions,
        description=(
            "Multilooked interferogram options: look factors, filter type,"
            " and block size for streaming I/O."
        ),
    )
    network: NetworkOptions = Field(
        default_factory=NetworkOptions,
        description="Interferogram network selection (nearest-N and/or single-reference).",
    )
    unwrap: IfgUnwrapOptions = Field(
        default_factory=IfgUnwrapOptions,
        description="Optional unwrapping step.  Set `unwrap.run_unwrap = false` to skip.",
    )
    stitch: StitchOptions = Field(
        default_factory=StitchOptions,
        description="Stitch per-burst interferograms and optionally crop to bbox.",
    )

    n_workers: int = Field(
        default=4,
        ge=1,
        description="Process pool size for COMPASS geocoding.",
    )
    threads_per_worker: int = Field(
        default=8,
        ge=1,
        description="OMP_NUM_THREADS for each geocoding worker.",
    )
    overwrite: bool = Field(
        default=False,
        description="Overwrite existing intermediate / output files.",
    )

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    # ------------------------------------------------------------------
    # Validators  (mirrored from core.Workflow)
    # ------------------------------------------------------------------

    @field_validator("wkt", mode="before")
    @classmethod
    def _check_file_and_parse_wkt(cls, v: Any) -> Any:
        if v is None:
            return v
        if Path(v).exists():
            v = Path(v).read_text().strip()
        try:
            shp_wkt.loads(v)
        except Exception as e:
            msg = f"Invalid WKT string: {e}"
            raise ValueError(msg) from e
        return v

    @field_validator("work_dir", "orbit_dir")
    @classmethod
    def _expand_dirs(cls, v: Any) -> Path:
        return Path(v).expanduser().resolve()

    @model_validator(mode="before")
    @classmethod
    def _sync_aoi(cls, values: Any) -> Any:
        """Push top-level bbox/wkt into the search source (same as Workflow)."""
        if not isinstance(values, dict):
            return values
        if "search" not in values:
            values["search"] = {}
        elif isinstance(
            values["search"],
            (BurstSearch, LocalSafeSearch, OperaCslcSearch, NisarGslcSearch),
        ):
            values["search"] = values["search"].model_dump(
                exclude_unset=True, by_alias=True
            )
        if isinstance(values["search"], dict) and "kind" not in values["search"]:
            values["search"]["kind"] = "safe"
        outer_bbox = values.get("bbox")
        outer_wkt = values.get("wkt")
        inner = values["search"]
        inner_bbox = inner.get("bbox")
        inner_wkt = inner.get("wkt")
        bbox = outer_bbox or inner_bbox
        wkt_value = outer_wkt or inner_wkt
        if not bbox and not wkt_value:
            msg = "Must specify `bbox` or `wkt` (on IfgWorkflow or `search`)"
            raise ValueError(msg)
        if bbox is not None:
            values["bbox"] = bbox
            if not inner_bbox:
                inner["bbox"] = bbox
        if outer_wkt is not None:
            values["wkt"] = outer_wkt
            if not inner_wkt:
                inner["wkt"] = outer_wkt
        return values

    @model_validator(mode="after")
    def _set_bbox_and_wkt(self) -> "IfgWorkflow":
        if self.bbox is None and self.wkt is not None:
            self.bbox = shp_wkt.loads(self.wkt).bounds
        assert self.bbox is not None
        if self.bbox[1] > self.bbox[3]:
            msg = f"Latitude min must be lower than max, got {self.bbox}"
            raise ValueError(msg)
        if self.bbox[0] > self.bbox[2]:
            msg = f"Longitude min must be lower than max, got {self.bbox}"
            raise ValueError(msg)
        return self

    # ------------------------------------------------------------------
    # Computed paths
    # ------------------------------------------------------------------

    @computed_field  # type: ignore[prop-decorator]
    @property
    def log_dir(self) -> Path:
        return self.work_dir / "logs"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def gslc_dir(self) -> Path:
        return self.work_dir / "gslcs"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def geom_dir(self) -> Path:
        return self.work_dir / "geometry"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def ifg_dir(self) -> Path:
        return self.work_dir / "interferograms"

    # DEM buffer constants (same as Workflow)
    _DEM_BUFFER_DEG_COMPASS = 1.0
    _DEM_BUFFER_DEG_DEFAULT = 0.25

    def _pad_bbox(
        self, bbox: tuple[float, float, float, float], buf_deg: float
    ) -> tuple[float, float, float, float]:
        return (
            bbox[0] - buf_deg,
            bbox[1] - buf_deg,
            bbox[2] + buf_deg,
            bbox[3] + buf_deg,
        )

    @property
    def _dem_bbox(self) -> tuple[float, float, float, float]:
        if self.dem_bbox is not None:
            return self.dem_bbox
        assert self.bbox is not None
        buf = (
            self._DEM_BUFFER_DEG_COMPASS
            if isinstance(self.search, (BurstSearch, LocalSafeSearch))
            else self._DEM_BUFFER_DEG_DEFAULT
        )
        return self._pad_bbox(self.bbox, buf)

    @property
    def _water_mask_bbox(self) -> tuple[float, float, float, float]:
        assert self.bbox is not None
        return self._pad_bbox(self.bbox, self._DEM_BUFFER_DEG_DEFAULT)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def save(self, config_file: Filename = "sweets_ifg_config.yaml") -> None:
        """Save this configuration to a YAML file."""
        logger.info(f"Saving IFG config to {config_file}")
        self.to_yaml(config_file)

    @classmethod
    def load(cls, config_file: Filename = "sweets_ifg_config.yaml") -> "IfgWorkflow":
        """Load a configuration from a YAML file."""
        logger.info(f"Loading IFG config from {config_file}")
        return cls.from_yaml(config_file)

    # ------------------------------------------------------------------
    # Step helpers (duplicated from core.Workflow)
    # ------------------------------------------------------------------

    _MIN_VALID_GSLC_BYTES = 1 * 1024 * 1024

    def _existing_safes(self) -> list[Path]:
        assert isinstance(self.search, (BurstSearch, LocalSafeSearch))
        return self.search.existing_safes()

    def _existing_gslcs(self) -> list[Path]:
        if isinstance(self.search, OperaCslcSearch):
            return [
                p
                for p in self.search.existing_cslcs()
                if p.stat().st_size >= self._MIN_VALID_GSLC_BYTES
            ]
        if isinstance(self.search, NisarGslcSearch):
            import rasterio

            valid: list[Path] = []
            for p in self.search.existing_files():
                try:
                    with rasterio.open(p) as src:
                        assert not (src.transform.a == 1.0 and src.transform.e == 1.0)
                        assert src.crs is not None
                except Exception as e:
                    logger.warning(f"Dropping {p.name}: not a valid raster ({e}).")
                    continue
                valid.append(p)
            return valid
        return [
            p
            for p in sorted(self.gslc_dir.glob("t*/*/t*.h5"))
            if not p.name.startswith("static_")
            and p.stat().st_size >= self._MIN_VALID_GSLC_BYTES
        ]

    def _existing_static_layers(self) -> list[Path]:
        if isinstance(self.search, OperaCslcSearch):
            return self.search.existing_static_layers()
        if isinstance(self.search, NisarGslcSearch):
            return []
        return sorted(self.gslc_dir.glob("t*/*/static_*.h5"))

    def _gslc_root(self) -> Path:
        if isinstance(self.search, (OperaCslcSearch, NisarGslcSearch)):
            return self.search.out_dir
        return self.gslc_dir

    def _apply_missing_data_filter(self, gslc_files: list[Path]) -> list[Path]:
        if isinstance(self.search, NisarGslcSearch):
            return gslc_files
        if len(gslc_files) < 2:
            return gslc_files
        from opera_utils.missing_data import (
            get_missing_data_options,
            print_with_rich,
        )

        try:
            options = get_missing_data_options(
                slc_files=[str(p) for p in gslc_files],
            )
        except Exception as e:
            logger.warning(
                f"get_missing_data_options failed ({e}); skipping missing-data filter."
            )
            return gslc_files
        if not options:
            logger.warning("No missing-data options returned; skipping filter.")
            return gslc_files

        top = options[0]
        if top.num_candidate_bursts == top.total_num_bursts:
            logger.info(
                f"Missing-data filter: all {top.total_num_bursts} CSLCs are complete;"
                " nothing to exclude."
            )
            return gslc_files

        logger.info(f"Missing-data filter: {len(options)} consistent subset option(s).")
        print_with_rich(options, use_stderr=False)
        logger.info(
            f"Keeping option #1: {top.num_burst_ids} burst(s) x"
            f" {top.num_dates} date(s) = {top.total_num_bursts} CSLCs."
        )

        kept_set = {Path(p) for p in top.inputs}
        to_exclude = [p for p in gslc_files if p not in kept_set]
        root = self._gslc_root().resolve()
        excluded_dir = (self.work_dir / "excluded_cslcs").resolve()
        for f in to_exclude:
            src = f.resolve()
            try:
                rel = src.relative_to(root)
            except ValueError:
                rel = Path(src.name)
            dst = excluded_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            logger.warning(f"Excluding {rel} -> excluded_cslcs/{rel}")
            shutil.move(str(src), str(dst))

        return sorted(kept_set)

    @log_runtime
    def _download(self) -> list[Path]:
        if isinstance(self.search, OperaCslcSearch):
            existing = self.search.existing_cslcs()
            if existing and not self.overwrite:
                logger.info(
                    f"Found {len(existing)} existing OPERA CSLCs; skipping download."
                )
            else:
                self.search.download()
            if not self.search.existing_static_layers() or self.overwrite:
                self.search.download_static_layers()
            return self.search.existing_cslcs()

        if isinstance(self.search, NisarGslcSearch):
            existing = self.search.existing_files()
            if existing and not self.overwrite:
                logger.info(
                    f"Found {len(existing)} existing NISAR GSLCs; skipping download."
                )
                return existing
            return self.search.download()

        if isinstance(self.search, LocalSafeSearch):
            existing = self.search.existing_safes()
            if not existing:
                msg = (
                    f"LocalSafeSearch.out_dir={self.search.out_dir} has no"
                    " S1 SAFE dirs or zip archives."
                )
                raise RuntimeError(msg)
            logger.info(
                f"LocalSafeSearch: using {len(existing)} inputs from {self.search.out_dir}."
            )
            return existing

        assert isinstance(self.search, BurstSearch)
        existing = self.search.existing_safes()
        if existing and not self.overwrite:
            logger.info(
                f"Found {len(existing)} existing SAFE dirs; skipping burst2safe."
            )
            return existing
        return self.search.download()

    @log_runtime
    def _geocode_slcs(
        self, safes: list[Path], dem_file: Path, burst_db_file: Path
    ) -> tuple[list[Path], list[Path]]:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        using_zipped = safes[0].suffix == ".zip"
        compass_cfg_files = create_config_files(
            slc_dir=safes[0].parent,
            burst_db_file=burst_db_file,
            dem_file=dem_file,
            orbit_dir=self.orbit_dir,
            bbox=self.bbox,
            y_posting=self.slc_posting[0],
            x_posting=self.slc_posting[1],
            pol_type=self.pol_type,
            out_dir=self.gslc_dir,
            overwrite=self.overwrite,
            using_zipped=using_zipped,
            gpu_enabled=self.gpu_enabled,
        )

        existing = {p.name: p for p in self._existing_gslcs()}
        logger.info(f"Found {len(existing)} existing GSLCs")
        gslc_files: list[Path] = []
        todo: list[Path] = []
        for cfg in compass_cfg_files:
            name = _cfg_to_filename(cfg)
            if name in existing:
                gslc_files.append(existing[name])
            else:
                todo.append(cfg)
        if todo:
            run = partial(run_geocode, log_dir=self.log_dir)
            with ProcessPoolExecutor(max_workers=self.n_workers) as pool:
                gslc_files.extend(pool.map(run, todo))

        static_existing = {p.name: p for p in self._existing_static_layers()}
        first_per_burst = [
            cfgs[0] for cfgs in group_by_burst(compass_cfg_files).values()
        ]
        static_files: list[Path] = []
        static_todo: list[Path] = []
        for cfg in first_per_burst:
            name = _cfg_to_static_filename(cfg)
            if name in static_existing:
                static_files.append(static_existing[name])
            else:
                static_todo.append(cfg)
        if static_todo:
            run_sl = partial(run_static_layers, log_dir=self.log_dir)
            with ProcessPoolExecutor(max_workers=self.n_workers) as pool:
                static_files.extend(pool.map(run_sl, static_todo))

        return sorted(gslc_files), sorted(static_files)

    @log_runtime
    def _stitch_geometry(self, static_files: list[Path]) -> list[Path]:
        from dolphin._types import Bbox

        bbox = Bbox(*self.bbox) if self.bbox is not None else None
        return stitch_geometry(
            geom_path_list=[Path(p) for p in static_files],
            geom_dir=self.geom_dir,
            dem_filename=self.dem_filename,
            looks=self.crossmul.looks,
            bbox=bbox,
            overwrite=self.overwrite,
        )

    def _subdataset(self) -> str:
        if isinstance(self.search, NisarGslcSearch):
            return "/unused-for-raster-inputs"
        return "/data/VV"

    # ------------------------------------------------------------------
    # IFG network + crossmul
    # ------------------------------------------------------------------

    @log_runtime
    def _run_ifg_network(self, gslc_files: list[Path]) -> list[tuple[Path, Path]]:
        """Form all interferograms in the network, grouped by burst ID.

        Geocoded SLCs from different bursts have different spatial extents and
        cannot be directly crossmul'd together.  We group by burst ID (the
        directory two levels up from each HDF5, e.g. ``t146_312777_iw1``) and
        run an independent interferogram network per burst.  Outputs go into
        ``<ifg_dir>/<burst_id>/<date1>_<date2>/``.

        Returns
        -------
        list of (phase_path, coherence_path) tuples for every produced pair.

        """
        from dolphin.interferogram import Network

        self.ifg_dir.mkdir(parents=True, exist_ok=True)

        subdataset = self._subdataset()

        # Group by burst ID (parent directory name, e.g. t146_312777_iw1).
        # For OPERA / NISAR sources that are not burst-organized, all SLCs fall
        # into the same "single burst" group.
        burst_groups: dict[str, list[Path]] = {}
        for slc in gslc_files:
            burst_id = _burst_id_from_gslc(slc)
            burst_groups.setdefault(burst_id, []).append(slc)

        products: list[tuple[Path, Path]] = []
        for burst_id, burst_slcs in sorted(burst_groups.items()):
            burst_slcs_sorted = sorted(burst_slcs)
            logger.info(f"Burst {burst_id}: {len(burst_slcs_sorted)} SLC(s)")
            if len(burst_slcs_sorted) < 2:
                logger.warning(f"Burst {burst_id}: only 1 SLC, skipping.")
                continue

            burst_ifg_dir = self.ifg_dir / burst_id
            burst_ifg_dir.mkdir(parents=True, exist_ok=True)

            # Build pair list via dolphin's Network
            net = Network(
                slc_list=burst_slcs_sorted,
                outdir=burst_ifg_dir,
                max_bandwidth=(
                    self.network.max_bandwidth
                    if self.network.max_bandwidth > 0
                    else None
                ),
                max_temporal_baseline=self.network.max_temporal_baseline,
                subdataset=subdataset,
                write=False,
            )
            pairs: list = list(net.slc_file_pairs)

            # Add single-reference pairs if requested
            if self.network.reference_date is not None:
                ref_date_str = self.network.reference_date.replace("-", "")
                ref_slc: Path | None = None
                for slc in burst_slcs_sorted:
                    if ref_date_str in slc.name:
                        ref_slc = slc
                        break
                if ref_slc is None:
                    logger.warning(
                        f"Burst {burst_id}: reference_date"
                        f" {self.network.reference_date!r} not found; skipping"
                        " single-reference pairs."
                    )
                else:
                    existing_keys = {(str(a), str(b)) for a, b in pairs}
                    for sec_slc in burst_slcs_sorted:
                        if sec_slc == ref_slc:
                            continue
                        key = (str(ref_slc), str(sec_slc))
                        if key not in existing_keys:
                            pairs.append((ref_slc, sec_slc))
                            existing_keys.add(key)

            if not pairs:
                logger.warning(f"Burst {burst_id}: no pairs formed; skipping.")
                continue

            logger.info(f"Burst {burst_id}: forming {len(pairs)} interferogram(s)...")
            for slc1, slc2 in pairs:
                date1 = _date_from_gslc(Path(slc1))
                date2 = _date_from_gslc(Path(slc2))
                pair_dir = burst_ifg_dir / f"{date1}_{date2}"
                phase_p, coh_p = run_crossmul(
                    ref_file=Path(slc1),
                    sec_file=Path(slc2),
                    pair_dir=pair_dir,
                    date1=date1,
                    date2=date2,
                    options=self.crossmul,
                    subdataset=subdataset,
                )
                products.append((phase_p, coh_p))

        return products

    # ------------------------------------------------------------------
    # Optional unwrapping
    # ------------------------------------------------------------------

    @log_runtime
    def _run_unwrap(
        self,
        ifg_products: list[tuple[Path, Path]],
    ) -> list[Path]:
        """Unwrap all wrapped-phase files.

        Returns
        -------
        list[Path]
            Paths to unwrapped-phase GeoTIFFs.

        """
        from dolphin.unwrap import run as dolphin_unwrap
        from dolphin.workflows import UnwrapOptions

        # Derive wrapped-phase files from complex IFGs if needed (the stitched
        # path already produces _wrapped_phase.tif; the per-burst path does not).
        phase_files = [_ensure_wrapped_phase(p) for p, _ in ifg_products]
        coh_files = [c for _, c in ifg_products]

        unw_dir = self.work_dir / "unwrapped"
        unw_dir.mkdir(parents=True, exist_ok=True)

        opts = self.unwrap
        az_looks, rg_looks = self.crossmul.looks
        effective_nlooks = (
            opts.nlooks if opts.nlooks is not None else float(az_looks * rg_looks)
        )

        ntiles = (2, 2) if opts.snaphu_ntiles == "auto" else opts.snaphu_ntiles  # type: ignore[assignment]

        unwrap_options = UnwrapOptions.model_validate(
            {
                "unwrap_method": opts.unwrap_method,
                "n_parallel_jobs": opts.n_parallel_jobs,
                "snaphu_options": {
                    "ntiles": list(ntiles),
                    "tile_overlap": list(opts.snaphu_tile_overlap),
                    "cost": opts.snaphu_cost,
                },
            }
        )

        mask = self.water_mask_filename if self.water_mask_filename.exists() else None

        unw_files, _ = dolphin_unwrap(
            ifg_filenames=phase_files,
            cor_filenames=coh_files,
            output_path=unw_dir,
            unwrap_options=unwrap_options,
            nlooks=effective_nlooks,
            mask_filename=mask,
            overwrite=self.overwrite,
        )
        logger.info(f"Unwrapped {len(unw_files)} interferogram(s) in {unw_dir}")
        return unw_files

    def _stitch_ifgs(
        self,
        ifg_products: list[tuple[Path, Path]],
    ) -> list[tuple[Path, Path]]:
        """Stitch per-burst interferograms into one file per date pair.

        Parameters
        ----------
        ifg_products
            Per-burst ``(wrapped_phase, coherence)`` path pairs produced by
            :meth:`_run_ifg_network`.

        Returns
        -------
        list[tuple[Path, Path]]
            Stitched ``(wrapped_phase, coherence)`` path pairs in
            ``<ifg_dir>/stitched/``.

        """
        from dolphin.stitching import merge_by_date

        stitch_dir = self.ifg_dir / "stitched"
        stitch_dir.mkdir(parents=True, exist_ok=True)

        ifg_files = [p for p, _ in ifg_products]
        coh_files = [c for _, c in ifg_products]

        from dolphin._types import Bbox

        out_bounds: Bbox | None = (
            Bbox(*self.bbox) if self.stitch.crop_to_bbox and self.bbox else None
        )

        if self.stitch.run_burst_align:
            try:
                from dolphin.burst_alignment import align_bursts
            except ImportError as e:
                msg = (
                    "stitch.run_burst_align requires dolphin.burst_alignment,"
                    " which is on the feat/burst-alignment branch of dolphin"
                    " and not yet released. Install that branch or set"
                    " run_burst_align: false."
                )
                raise ImportError(msg) from e
            align_dir = stitch_dir / "burst_aligned"
            align_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Running burst alignment on complex IFG files...")
            ifg_files, _ = align_bursts(
                ifg_files,
                align_dir,
                degree=self.stitch.burst_align_degree,
            )

        logger.info(
            f"Stitching {len(ifg_files)} per-burst interferogram pairs"
            + (f" cropped to bbox {self.bbox}" if out_bounds else "")
        )
        # Stitch complex IFGs first — GDAL interpolates real+imag independently,
        # which is correct.  Wrapped phase is derived after so interpolation
        # never crosses a ±π discontinuity.
        ifg_map = merge_by_date(
            ifg_files,
            output_dir=stitch_dir,
            output_suffix="_ifg.tif",
            out_bounds=out_bounds,
            out_bounds_epsg=4326 if out_bounds else None,
            overwrite=self.overwrite,
        )
        coh_map = merge_by_date(
            coh_files,
            output_dir=stitch_dir,
            output_suffix="_coherence.tif",
            out_bounds=out_bounds,
            out_bounds_epsg=4326 if out_bounds else None,
            overwrite=self.overwrite,
        )

        # Derive wrapped-phase GeoTIFFs from the stitched complex IFGs.
        stitched: list[tuple[Path, Path]] = []
        for date_key, ifg_path in sorted(ifg_map.items()):
            coh_path = coh_map.get(date_key)
            if coh_path is None:
                continue
            phase_path = _derive_wrapped_phase(ifg_path)
            stitched.append((phase_path, coh_path))
        logger.info(f"Stitched {len(stitched)} interferogram pair(s) to {stitch_dir}")
        return stitched

    # ------------------------------------------------------------------
    # Top-level run
    # ------------------------------------------------------------------

    @log_runtime
    def run(self, starting_step: int = 1) -> list[tuple[Path, Path]]:
        """Run the interferogram workflow.

        Parameters
        ----------
        starting_step : int
            Skip earlier stages: 1 = download, 2 = geocode, 3 = crossmul.

        Returns
        -------
        list[tuple[Path, Path]]
            ``(wrapped_phase_path, coherence_path)`` pairs for each interferogram.

        """
        setup_nasa_netrc()
        set_num_threads(self.threads_per_worker)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        is_safe = isinstance(self.search, (BurstSearch, LocalSafeSearch))
        is_nisar = isinstance(self.search, NisarGslcSearch)
        needs_compass = is_safe

        if starting_step <= 1:
            with ThreadPoolExecutor(max_workers=4) as pool:
                dem_fut = pool.submit(create_dem, self.dem_filename, self._dem_bbox)
                mask_fut = pool.submit(
                    create_water_mask,
                    self.water_mask_filename,
                    self._water_mask_bbox,
                )
                burst_db_fut = pool.submit(get_burst_db) if needs_compass else None
                futures: list = [dem_fut, mask_fut]
                if burst_db_fut is not None:
                    futures.append(burst_db_fut)
                wait(futures)
                dem_fut.result()
                mask_fut.result()
                burst_db_file = burst_db_fut.result() if burst_db_fut else None
            self._download()
        else:
            burst_db_file = get_burst_db() if needs_compass else None

        if starting_step <= 2:
            if isinstance(self.search, NisarGslcSearch):
                logger.info("NISAR source: skipping COMPASS and geometry stitching.")
            elif isinstance(self.search, OperaCslcSearch):
                static_files = self._existing_static_layers()
                if not static_files:
                    msg = (
                        f"No CSLC-STATIC layers found in"
                        f" {self.search.static_layers_dir!s}."
                    )
                    raise RuntimeError(msg)
                self._stitch_geometry(static_files)
            else:
                safes = self._existing_safes()
                if not safes:
                    msg = f"No SAFE dirs found in {self.search.out_dir}."
                    raise RuntimeError(msg)
                download_orbits(self.search.out_dir, self.orbit_dir)
                assert burst_db_file is not None
                _, static_files = self._geocode_slcs(
                    safes, self.dem_filename, burst_db_file
                )
                self._stitch_geometry(static_files)

        # Collect GSLCs
        gslc_files = self._existing_gslcs()
        logger.info(f"Found {len(gslc_files)} GSLC files for interferogram formation")
        if not gslc_files:
            where = (
                self.search.out_dir
                if (is_nisar or isinstance(self.search, OperaCslcSearch))
                else self.gslc_dir
            )
            msg = f"No GSLCs found in {where}."
            raise RuntimeError(msg)

        gslc_files = self._apply_missing_data_filter(gslc_files)
        if len(gslc_files) < 2:
            msg = (
                f"Only 1 usable GSLC found ({gslc_files[0].name});"
                " need at least 2 to form an interferogram."
            )
            raise RuntimeError(msg)

        # Crossmul
        if starting_step <= 3:
            ifg_products = self._run_ifg_network(gslc_files)
        else:
            ifg_products = _collect_existing_ifg_products(self.ifg_dir)

        # Stitch per-burst interferograms + optional bbox crop
        if self.stitch.run_stitch:
            if starting_step <= 4:
                stitched = self._stitch_ifgs(ifg_products)
            else:
                stitched = _collect_existing_ifg_products(self.ifg_dir / "stitched")
            # Unwrap the stitched products (single file per pair, smaller)
            if self.unwrap.run_unwrap:
                self._run_unwrap(stitched)
            return stitched

        # Unwrap per-burst products (no stitching requested)
        if self.unwrap.run_unwrap:
            self._run_unwrap(ifg_products)

        return ifg_products


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


# Matches OPERA burst token like _T048-101101-IW3_ inside a filename stem.
# \b word-boundary doesn't work here because underscore is a word char.
_OPERA_BURST_RE = re.compile(r"(?:^|_)(T\d+-\d+-IW\d)(?:_|$)", re.IGNORECASE)


def _burst_id_from_gslc(path: Path) -> str:
    """Extract burst ID from a GSLC path.

    Tries three strategies in order:

    1. Grandparent directory name for COMPASS burst-organized output
       (``<burst_id>/<date>/<file>.h5``, e.g. ``t146_312777_iw1``).
    2. OPERA CSLC filename pattern ``T048-101101-IW3`` → ``t048_101101_iw3``.
    3. Falls back to ``"single"`` so non-burst sources still work.
    """
    parts = path.parts
    if len(parts) >= 3:
        grandparent = parts[-3]
        if (
            "_iw" in grandparent
            or grandparent.startswith("t0")
            or grandparent.startswith("t1")
        ):
            return grandparent

    m = _OPERA_BURST_RE.search(path.stem)
    if m:
        return m.group(1).replace("-", "_").lower()

    return "single"


def _date_from_gslc(path: Path) -> str:
    """Extract YYYYMMDD date from a GSLC filename.

    For COMPASS files (``t078_165578_iw3_20221029.h5``) the date is the
    last ``_``-delimited 8-digit token.  For OPERA CSLCs the acquisition
    date is embedded as ``20240101T232835Z``; ``opera_utils.get_dates``
    handles both formats and we take the first (earliest) date found.
    """
    from opera_utils import get_dates

    dates = get_dates(path)
    if dates:
        return dates[0].strftime("%Y%m%d")
    msg = f"Cannot extract YYYYMMDD date from GSLC filename: {path.name}"
    raise ValueError(msg)


def _derive_wrapped_phase(ifg_path: Path) -> Path:
    """Write ``_wrapped_phase.tif`` alongside a complex ``_ifg.tif``.

    Idempotent: returns the existing file if already present.
    """
    import numpy as np
    from dolphin.io import load_gdal, write_arr
    from osgeo import gdal

    phase_path = Path(str(ifg_path).replace("_ifg.tif", "_wrapped_phase.tif"))
    if phase_path.exists():
        return phase_path

    ds = gdal.Open(str(ifg_path))
    gt = ds.GetGeoTransform()
    crs = ds.GetProjection()
    ds = None

    ifg = load_gdal(ifg_path)
    phase = np.angle(ifg).astype(np.float32)
    phase[ifg == 0] = np.nan  # propagate nodata

    write_arr(
        arr=phase,
        output_name=phase_path,
        like_filename=ifg_path,
        dtype=np.float32,
        driver="GTiff",
        geotransform=list(gt),
        projection=crs,
        nodata=float("nan"),
    )
    return phase_path


def _ensure_wrapped_phase(ifg_path: Path) -> Path:
    """Return the wrapped-phase path for an IFG, deriving it if needed."""
    if "_ifg.tif" in ifg_path.name:
        return _derive_wrapped_phase(ifg_path)
    return ifg_path


def _collect_existing_ifg_products(ifg_dir: Path) -> list[tuple[Path, Path]]:
    """Collect already-produced (ifg, coherence) pairs from ``ifg_dir``.

    Handles both flat ``<ifg_dir>/`` and burst-grouped
    ``<ifg_dir>/<burst_id>/`` layouts.
    """
    products: list[tuple[Path, Path]] = []
    for ifg_f in sorted(ifg_dir.rglob("*_ifg.tif")):
        coh_f = ifg_f.parent / ifg_f.name.replace("_ifg.tif", "_coherence.tif")
        if coh_f.exists():
            products.append((ifg_f, coh_f))
    return products


def _cfg_to_filename(cfg_path: Path) -> str:
    """COMPASS runconfig path -> expected GSLC HDF5 filename."""
    date = cfg_path.name.split("_")[2]
    burst = "_".join(cfg_path.stem.split("_")[3:])
    return f"{burst}_{date}.h5"


def _cfg_to_static_filename(cfg_path: Path) -> str:
    """COMPASS runconfig path -> expected static-layers HDF5 filename."""
    burst = "_".join(cfg_path.stem.split("_")[3:])
    return f"static_layers_{burst}.h5"
