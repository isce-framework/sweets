"""End-to-end Sentinel-1 InSAR workflow.

Two source paths share the same downstream pipeline:

- ``BurstSearch`` (default): download burst-trimmed S1 SAFEs via burst2safe,
  geocode them with COMPASS, then run dolphin.
- ``OperaCslcSearch``: download pre-made OPERA CSLCs from ASF (skips COMPASS,
  locked to OPERA's posting), then run dolphin.

Optional post-step: tropospheric correction using OPERA L4 TROPO-ZENITH
products via opera_utils.tropo.

The workflow is defined as a single :class:`Workflow` Pydantic model that
can be serialized to / loaded from a ``sweets_config.yaml``.
"""

from __future__ import annotations

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
from ._dolphin import DolphinOptions, run_displacement
from ._geocode_slcs import create_config_files, run_geocode, run_static_layers
from ._geometry import stitch_geometry
from ._log import log_runtime
from ._netrc import setup_nasa_netrc
from ._orbit import download_orbits
from ._tropo import TropoOptions, run_tropo_correction
from ._types import Filename
from .dem import create_dem, create_water_mask
from .download import BurstSearch, LocalSafeSearch, NisarGslcSearch, OperaCslcSearch

if TYPE_CHECKING:
    from dolphin.workflows.displacement import OutputPaths

# Discriminated union on the `kind` field. Pydantic dispatches directly to
# the matching variant — much cleaner errors than a plain Union, which
# tries each variant in order and reports failures from all of them.
Source = Annotated[
    Union[BurstSearch, LocalSafeSearch, OperaCslcSearch, NisarGslcSearch],
    Field(discriminator="kind"),
]


class Workflow(YamlModel):
    """End-to-end Sentinel-1 InSAR workflow configuration."""

    work_dir: Path = Field(
        default_factory=Path.cwd,
        description="Root of working directory for processing.",
        validate_default=True,
    )

    bbox: Optional[tuple[float, float, float, float]] = Field(
        default=None,
        description=(
            "AOI as (left, bottom, right, top) in decimal degrees. Either"
            " `bbox` or `wkt` must be set."
        ),
    )
    wkt: Optional[str] = Field(
        default=None,
        description="AOI as a WKT polygon (or path to a `.wkt` file). Overrides bbox.",
    )

    search: Source = Field(
        ...,
        description=(
            "Source of input SLCs. One of `BurstSearch` (raw S1 bursts via"
            " burst2safe + COMPASS), `LocalSafeSearch` (pre-downloaded full-"
            "frame SAFE dirs / zips + COMPASS), `OperaCslcSearch` (pre-made"
            " OPERA CSLCs), or `NisarGslcSearch` (pre-made NISAR GSLCs);"
            " discriminated by the `kind` field."
        ),
    )

    dem_filename: Path = Field(
        default_factory=lambda data: data["work_dir"] / "dem.tif",
        description=(
            "DEM in EPSG:4326. If left as the default, sweets downloads a"
            " Copernicus DEM via sardem."
        ),
    )
    dem_bbox: Optional[tuple[float, float, float, float]] = Field(
        default=None,
        description=(
            "Optional AOI override for DEM download (left, bottom, right,"
            " top) in decimal degrees. Use this when the default buffer"
            " around the study-area `bbox` isn't enough — most commonly,"
            " when running COMPASS on a `BurstSearch` whose full IW burst"
            " footprint extends beyond the study area. If unset, sweets"
            " picks a default: ~1 deg around `bbox` for BurstSearch (large"
            " enough to cover a full IW burst) and ~0.25 deg around `bbox`"
            " for NISAR / OPERA-CSLC sources (which are already geocoded"
            " and only need the DEM for tropo + water-mask context)."
        ),
    )
    water_mask_filename: Path = Field(
        default_factory=lambda data: data["work_dir"] / "watermask.tif",
        description=(
            "Water mask in EPSG:4326 (uint8 GTiff, 1=land, 0=water). If left"
            " as the default, sweets derives one from a Copernicus DEM."
        ),
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
        description="Polarization type to geocode (COMPASS knob).",
    )
    gpu_enabled: bool = Field(
        default=True,
        description=(
            "Run COMPASS geocoding on the GPU when an isce3-cuda build is"
            " available (the `gpu` pixi environment). Harmless on CPU-only"
            " installs — sweets probes `isce3.cuda` before patching the"
            " COMPASS runconfigs and downgrades to CPU when it is missing,"
            " since COMPASS itself would otherwise raise. Independent of"
            " `dolphin.gpu_enabled`, which controls phase linking."
        ),
    )

    dolphin: DolphinOptions = Field(
        default_factory=DolphinOptions,
        description="Configuration for the dolphin displacement workflow.",
    )

    tropo: TropoOptions = Field(
        default_factory=TropoOptions,
        description=(
            "Configuration for the optional tropospheric correction step that"
            " runs after dolphin. Off by default; set `tropo.enabled = true`"
            " (or pass `--do-tropo` on the CLI) to turn it on."
        ),
    )

    n_workers: int = Field(
        default=4,
        description="Process pool size for COMPASS geocoding.",
        ge=1,
    )
    threads_per_worker: int = Field(
        default=8,
        description="OMP_NUM_THREADS for each geocoding worker.",
        ge=1,
    )
    overwrite: bool = Field(
        default=False,
        description="Overwrite existing intermediate / output files.",
    )

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("wkt", mode="before")
    @classmethod
    def _check_file_and_parse_wkt(cls, v):
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
    def _expand_dirs(cls, v):
        return Path(v).expanduser().resolve()

    @model_validator(mode="before")
    @classmethod
    def _sync_aoi(cls, values: Any) -> Any:
        """Push the top-level bbox/wkt down into the search source.

        The outer ``Workflow.bbox`` / ``Workflow.wkt`` are the canonical AOI;
        the nested ``search`` field gets the same bbox so the downloader (or
        local-source consumer) knows what to fetch / crop to. Only ``bbox``
        is forced — ``search.wkt`` is left alone so non-rectangular search
        polygons can be specified at the source level if a user wants.

        If ``search`` is provided as a dict without ``kind``, default to
        ``"safe"`` for backwards compatibility with existing configs.
        """
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
            msg = "Must specify `bbox` or `wkt` (on Workflow or `search`)"
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
    def _set_bbox_and_wkt(self) -> "Workflow":
        # Derive bbox from wkt if only wkt was supplied; downstream code
        # (DEM, dolphin bounds, etc.) all reads bbox, not wkt. We do NOT
        # auto-fill wkt from bbox: nothing in the workflow reads outer wkt
        # past this validator, and round-tripping a computed wkt was
        # contaminating reload equality (the inner search would gain a
        # wkt it never had on the first pass).
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
    def dolphin_dir(self) -> Path:
        return self.work_dir / "dolphin"

    # Buffer (deg) padded around `bbox` when deriving an implicit DEM / water-
    # mask extent. The COMPASS value is large enough to cover the full
    # footprint of any IW burst (~20 x 85 km) regardless of where within the
    # burst the study area sits; COMPASS geocoding needs DEM coverage for the
    # whole burst, not just the crop area. Applies to both BurstSearch and
    # LocalSafeSearch (full-frame SAFEs are larger still, so the same buffer
    # is still safe). All other sources consume the DEM only for water
    # masking + tropo context, so the study area + a small buffer is plenty.
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
        """Bbox passed to ``sardem`` when creating the DEM.

        Priority: user-set ``dem_bbox`` > COMPASS-path default (~1 deg
        buffer for BurstSearch / LocalSafeSearch) > everything-else default
        (~0.25 deg buffer). COMPASS needs DEM coverage for the full IW
        burst extent, not just the study area, so the buffer on the raw-
        SAFE path has to be big enough to absorb a whole burst.
        """
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
        """Bbox passed to the ASF water-mask tile mosaic.

        Always study-area-scoped: the water mask is only evaluated inside
        the dolphin bounds, and oversizing it would just waste tile
        downloads on the BurstSearch path (where ``_dem_bbox`` is large).
        """
        assert self.bbox is not None
        return self._pad_bbox(self.bbox, self._DEM_BUFFER_DEG_DEFAULT)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def save(self, config_file: Filename = "sweets_config.yaml") -> None:
        """Save this configuration to a YAML file."""
        logger.info(f"Saving config to {config_file}")
        self.to_yaml(config_file)

    @classmethod
    def load(cls, config_file: Filename = "sweets_config.yaml") -> "Workflow":
        """Load a configuration from a YAML file."""
        logger.info(f"Loading config from {config_file}")
        return cls.from_yaml(config_file)

    # ------------------------------------------------------------------
    # Step helpers
    # ------------------------------------------------------------------

    def _existing_safes(self) -> list[Path]:
        # Only meaningful for the COMPASS path (BurstSearch or LocalSafeSearch).
        # OperaCslcSearch / NisarGslcSearch deliver pre-geocoded HDF5s.
        assert isinstance(self.search, (BurstSearch, LocalSafeSearch))
        return self.search.existing_safes()

    # COMPASS-written CSLC HDF5s are tens to hundreds of MB. A 6-KB shell is
    # a leftover from a crashed run that wrote the attribute scaffolding but
    # never the data — accepting those silently breaks dolphin downstream
    # (issue #107). Treat anything below this size as not-yet-produced.
    _MIN_VALID_GSLC_BYTES = 1 * 1024 * 1024

    def _existing_gslcs(self) -> list[Path]:
        """Return on-disk CSLCs for whichever source variant is selected.

        - BurstSearch: COMPASS-written burst-organized HDF5s under gslc_dir.
        - OperaCslcSearch: pre-made OPERA CSLCs under search.out_dir.
        - NisarGslcSearch: pre-made NISAR GSLC HDF5s under search.out_dir.
          Each candidate is opened to confirm it actually carries the
          expected `/science/LSAR/GSLC/grids/frequency<X>` group — opera-
          utils' subsetter writes a metadata-only stub when the requested
          bbox doesn't intersect a product, and dolphin would crash on
          those.
        """
        if isinstance(self.search, OperaCslcSearch):
            return [
                p
                for p in self.search.existing_cslcs()
                if p.stat().st_size >= self._MIN_VALID_GSLC_BYTES
            ]
        if isinstance(self.search, NisarGslcSearch):
            # NisarGslcSearch.existing_files() returns the per-polarization
            # VRT wrappers that sweets writes alongside each downloaded
            # subset HDF5. The VRTs are tiny (~1 KB) so the MIN_VALID
            # byte-count guard from the COMPASS path doesn't apply; instead,
            # open each one through rasterio to confirm GDAL can see a real
            # geotransform and CRS through the VRT -> HDF5 subdataset.
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
        # NisarGslcSearch has no separate static layers product — the caller
        # is expected to skip the geometry-stitch step entirely.
        if isinstance(self.search, NisarGslcSearch):
            return []
        return sorted(self.gslc_dir.glob("t*/*/static_*.h5"))

    def _gslc_root(self) -> Path:
        """Return the directory we consider the 'root' of downloaded GSLCs.

        Used by `_apply_missing_data_filter` to compute relative paths
        for files that get moved into the excluded-CSLCs debug folder.
        """
        if isinstance(self.search, (OperaCslcSearch, NisarGslcSearch)):
            return self.search.out_dir
        return self.gslc_dir

    def _apply_missing_data_filter(self, gslc_files: list[Path]) -> list[Path]:
        """Keep only the largest spatially-consistent (burst_id, date) subset.

        Runs `opera_utils.missing_data.get_missing_data_options` to
        enumerate every (set of burst IDs, set of dates) subset where
        *every* chosen burst has *every* chosen date. The top option
        maximizes total bursts under that constraint — i.e. we keep the
        widest stack where dolphin can form a coherent network without
        spatial discontinuities from bursts-with-missing-dates.

        Files that aren't in the top option get moved (not deleted) to
        `<work_dir>/excluded_cslcs/...` preserving their relative path
        from the gslc root. That way a debugging user can pull them
        back if they want the raw download.

        NISAR GSLCs aren't burst-organized so this is a no-op for the
        NisarGslcSearch source — its `_download_group` already handles
        coverage selection via `_rank_signatures`.
        """
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
                f"get_missing_data_options failed ({e}); skipping"
                " missing-data filter."
            )
            return gslc_files
        if not options:
            logger.warning("No missing-data options returned; skipping filter.")
            return gslc_files

        top = options[0]
        if top.num_candidate_bursts == top.total_num_bursts:
            logger.info(
                f"Missing-data filter: all {top.total_num_bursts} CSLCs form a"
                f" complete {top.num_burst_ids}-burst x {top.num_dates}-date"
                " stack; nothing to exclude."
            )
            return gslc_files

        logger.info(f"Missing-data filter: {len(options)} consistent subset option(s).")
        print_with_rich(options, use_stderr=False)
        logger.info(
            f"Keeping option #1: {top.num_burst_ids} burst(s) x"
            f" {top.num_dates} date(s) = {top.total_num_bursts} CSLCs"
            f" (excluding {top.num_candidate_bursts - top.total_num_bursts}"
            " partial-coverage CSLCs)."
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
                    f"Found {len(existing)} existing OPERA CSLCs in"
                    f" {self.search.out_dir}; skipping ASF download."
                )
            else:
                self.search.download()
            # Always make sure static layers are present too.
            if not self.search.existing_static_layers() or self.overwrite:
                self.search.download_static_layers()
            return self.search.existing_cslcs()

        if isinstance(self.search, NisarGslcSearch):
            existing = self.search.existing_files()
            if existing and not self.overwrite:
                logger.info(
                    f"Found {len(existing)} existing NISAR GSLCs in"
                    f" {self.search.out_dir}; skipping CMR download."
                )
                return existing
            return self.search.download()

        if isinstance(self.search, LocalSafeSearch):
            existing = self.search.existing_safes()
            if not existing:
                msg = (
                    f"LocalSafeSearch.out_dir={self.search.out_dir} has no"
                    " S1 SAFE directories or zip archives; nothing to"
                    " geocode."
                )
                raise RuntimeError(msg)
            logger.info(self.search.summary())
            logger.info(
                f"LocalSafeSearch: using {len(existing)} pre-downloaded"
                f" input(s) from {self.search.out_dir}."
            )
            return existing

        assert isinstance(self.search, BurstSearch)
        existing = self.search.existing_safes()
        if existing and not self.overwrite:
            logger.info(
                f"Found {len(existing)} existing SAFE dirs in"
                f" {self.search.out_dir}; skipping burst2safe download."
            )
            return existing
        return self.search.download()

    @log_runtime
    def _geocode_slcs(
        self, safes: list[Path], dem_file: Path, burst_db_file: Path
    ) -> tuple[list[Path], list[Path]]:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        # COMPASS globs for either `*.zip` or `*.SAFE` based on this flag,
        # so infer it from what's actually on disk rather than making the
        # user declare it. BurstSearch always produces `.SAFE` directories.
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

        # Static layers (one per burst, not per date)
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
            looks=self.dolphin.strides,
            bbox=bbox,
            overwrite=self.overwrite,
        )

    def _dolphin_subdataset(self) -> str:
        """Pick the right HDF5 subdataset path for the current source.

        - COMPASS / OPERA CSLCs are HDF5 files; dolphin reads them via
          ``input_options.subdataset = /data/VV``.
        - NISAR GSLCs are wrapped by sweets in per-polarization VRTs
          that inject georeferencing on top of the raw HDF5 subdataset;
          dolphin opens the VRTs as plain rasters and ignores
          ``subdataset``, so any placeholder string works.
        """
        if isinstance(self.search, NisarGslcSearch):
            return "/unused-for-raster-inputs"
        return "/data/VV"

    def _dolphin_wavelength(self) -> Optional[float]:
        """Wavelength override for dolphin, or None to let dolphin auto-detect.

        For the NISAR source, sweets reads the precise carrier from the
        HDF5's ``centerFrequency`` dataset when available; this is
        richer than dolphin's own filename-based fallback (which can
        only tell L-band from S-band). dolphin's filename parser can't
        see through sweets' VRT wrappers anyway, so sweets must be the
        one peeking the HDF5. For every other source, return None and
        let dolphin's `model_post_init` handle it.
        """
        if isinstance(self.search, NisarGslcSearch):
            return self.search.wavelength()
        return None

    @log_runtime
    def _run_dolphin(self, gslc_files: list[Path]) -> "OutputPaths":
        mask = self.water_mask_filename if self.water_mask_filename.exists() else None
        return run_displacement(
            cslc_files=gslc_files,
            work_directory=self.dolphin_dir,
            options=self.dolphin,
            mask_file=mask,
            bounds=self.bbox,
            config_yaml=self.work_dir / "dolphin_config.yaml",
            subdataset=self._dolphin_subdataset(),
            wavelength=self._dolphin_wavelength(),
        )

    # ------------------------------------------------------------------
    # Top-level run
    # ------------------------------------------------------------------

    @log_runtime
    def run(self, starting_step: int = 1) -> "OutputPaths":
        """Run the full workflow.

        Parameters
        ----------
        starting_step : int
            Skip earlier stages if intermediate outputs are already on disk.
            ``1`` = download, ``2`` = geocode (BurstSearch only, OPERA path
            stitches geometry directly), ``3`` = dolphin.

        Returns
        -------
        dolphin.workflows.displacement.OutputPaths
            Output paths produced by dolphin.

        """
        setup_nasa_netrc()
        set_num_threads(self.threads_per_worker)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        is_safe = isinstance(self.search, (BurstSearch, LocalSafeSearch))
        is_nisar = isinstance(self.search, NisarGslcSearch)
        # COMPASS is only needed for the raw-SAFE path (BurstSearch or
        # LocalSafeSearch). OPERA and NISAR both deliver pre-geocoded HDF5s.
        needs_compass = is_safe

        # ---- Step 1: download (DEM, water mask, burst DB, source SLCs) ----
        if starting_step <= 1:
            with ThreadPoolExecutor(max_workers=4) as pool:
                dem_fut = pool.submit(create_dem, self.dem_filename, self._dem_bbox)
                mask_fut = pool.submit(
                    create_water_mask,
                    self.water_mask_filename,
                    self._water_mask_bbox,
                )
                # burst-db is only needed by COMPASS.
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

        # ---- Step 2: produce CSLCs + stitch geometry ----
        if starting_step <= 2:
            if isinstance(self.search, NisarGslcSearch):
                # NISAR GSLCs are already geocoded and carry their own
                # per-frame metadata; there's no static-layers product to
                # stitch, and dolphin reads the grid from the HDF5 itself.
                logger.info(
                    "NISAR source: skipping COMPASS and geometry stitching;"
                    " dolphin will read the grid from the GSLC HDF5s."
                )
            elif isinstance(self.search, OperaCslcSearch):
                # OPERA path: pre-made CSLCs, just stitch the static layers.
                static_files = self._existing_static_layers()
                if not static_files:
                    msg = (
                        f"No CSLC-STATIC layers found in"
                        f" {self.search.static_layers_dir!s}; cannot stitch geometry."
                    )
                    raise RuntimeError(msg)
                self._stitch_geometry(static_files)
            else:
                safes = self._existing_safes()
                if not safes:
                    msg = (
                        f"No SAFE directories found in {self.search.out_dir};"
                        " cannot geocode."
                    )
                    raise RuntimeError(msg)
                download_orbits(self.search.out_dir, self.orbit_dir)
                assert burst_db_file is not None
                _, static_files = self._geocode_slcs(
                    safes, self.dem_filename, burst_db_file
                )
                self._stitch_geometry(static_files)

        # ---- Step 3: dolphin ----
        # Always re-collect GSLCs from disk before dolphin so a starting_step=3
        # run still finds them.
        gslc_files = self._existing_gslcs()
        logger.info(f"Found {len(gslc_files)} GSLC files for dolphin")
        if not gslc_files:
            where = (
                self.search.out_dir
                if (is_nisar or isinstance(self.search, OperaCslcSearch))
                else self.gslc_dir
            )
            msg = f"No GSLCs found in {where}; cannot run dolphin."
            raise RuntimeError(msg)

        # Trim the stack to the largest spatially-consistent (burst,
        # date) subset — any burst missing one or more of the chosen
        # dates gets moved out to `excluded_cslcs/`. Keeps dolphin from
        # forming a network across partially-covered bursts (which
        # produces spatial discontinuities in the displacement field)
        # and from crashing on 1-SLC per-burst stacks like the one
        # that bit the BurstSearch path with a burst-boundary AOI.
        gslc_files = self._apply_missing_data_filter(gslc_files)

        if len(gslc_files) < 2:
            msg = (
                f"Only 1 usable GSLC survived for dolphin ({gslc_files[0].name});"
                " need at least 2 to form an interferogram. Widen the date"
                " range, pick a different track/frame, or drop the"
                " `frequency` / `polarizations` pins so sweets can auto-"
                "select whichever signature actually has a coherent stack."
            )
            raise RuntimeError(msg)
        out_paths = self._run_dolphin(gslc_files)

        # ---- Optional post-step: tropospheric correction ----
        if self.tropo.enabled:
            if isinstance(self.search, NisarGslcSearch):
                logger.warning(
                    "Tropo correction is not supported with the NISAR GSLC"
                    " source yet (no stitched incidence angle raster is"
                    " produced for NISAR). Skipping."
                )
            else:
                self._run_tropo(gslc_files, out_paths)

        return out_paths

    @log_runtime
    def _run_tropo(
        self, gslc_files: list[Path], out_paths: "OutputPaths"
    ) -> list[Path]:
        """Apply OPERA L4 TROPO-ZENITH corrections to dolphin's outputs.

        The new `run_tropo_correction` reaches into `dolphin_dir/unwrapped/`
        and `dolphin_dir/timeseries/` itself, correcting both the unwrapped
        interferograms and the per-pair timeseries rasters in one shot.
        """
        del out_paths  # glob from disk — survives starting_step=3 reruns
        incidence_path = self.geom_dir / "local_incidence_angle.tif"
        if not incidence_path.exists():
            msg = (
                f"Tropo correction needs the stitched local_incidence_angle"
                f" raster at {incidence_path}; rerun starting_step=2 first."
            )
            raise RuntimeError(msg)
        return run_tropo_correction(
            slc_files=gslc_files,
            dem_path=self.dem_filename,
            incidence_angle_path=incidence_path,
            dolphin_work_dir=self.dolphin_dir,
            options=self.tropo,
        )


def _cfg_to_filename(cfg_path: Path) -> str:
    """COMPASS runconfig path -> expected GSLC HDF5 filename.

    e.g. ``geo_runconfig_20221029_t078_165578_iw3.yaml``
        -> ``t078_165578_iw3_20221029.h5``
    """
    date = cfg_path.name.split("_")[2]
    burst = "_".join(cfg_path.stem.split("_")[3:])
    return f"{burst}_{date}.h5"


def _cfg_to_static_filename(cfg_path: Path) -> str:
    """COMPASS runconfig path -> expected static-layers HDF5 filename."""
    burst = "_".join(cfg_path.stem.split("_")[3:])
    return f"static_layers_{burst}.h5"
