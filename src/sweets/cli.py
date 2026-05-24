"""Sweets command-line interface (tyro-driven).

Four subcommands:

- ``sweets config``  — write a ``sweets_config.yaml`` from CLI flags.
- ``sweets run``     — execute a workflow from a config file.
- ``sweets schema``  — dump the workflow JSON schema.
- ``sweets report``  — render an HTML report for a finished run.

``sweets config`` is a pydantic subclass of :class:`sweets.core.Workflow`,
so every Workflow field — scalars, nested ``dolphin`` and ``tropo`` models,
everything — is surfaced on the CLI automatically with no hand-maintained
shadow fields. Tyro walks the pydantic schema and builds the flag layout
from it.

The one twist is ``Workflow.search``, which is a discriminated union over
``BurstSearch | OperaCslcSearch | NisarGslcSearch``. Tyro would otherwise
spread that across subcommand-style choices, forcing users to invoke
``sweets config burst-search ...``. Instead, :class:`ConfigCli` hides
``search`` from tyro with ``tyro.conf.Suppress`` and rebuilds it from a
handful of flat, ergonomic flags (``--source``, ``--start``, ``--end``,
``--track``, ...) in a wrap-mode validator that runs before Workflow's
own ``_sync_aoi`` pre-validator.

Every subcommand class exposes ``.execute()`` as the CLI dispatch entry
point (rather than ``.run()``), so :class:`ConfigCli`'s entry point
doesn't collide with :meth:`Workflow.run` inherited from the parent.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, Optional

import tyro
from pydantic import Field, model_validator

from sweets.core import Source, Workflow

SourceKind = Literal["safe", "local", "opera-cslc", "nisar-gslc"]


class ConfigCli(Workflow):
    """Create a ``sweets_config.yaml`` from CLI arguments.

    Subclass of :class:`Workflow` so every Workflow field — including
    nested ``dolphin`` and ``tropo`` — is automatically surfaced on the
    CLI with no redeclaration. The only additions here are:

    - Flat source flags (``start``, ``end``, ``source``, ``track``,
      ``frame``, ``frequency``, ``polarizations``, ``swaths``,
      ``out_dir``) used to build ``search`` in ``_assemble_search``
      since the discriminated-union ``search`` field is hidden from the
      CLI.
    - Legacy ``do_tropo`` alias for ``--tropo.enabled``.
    - Output-file knobs ``output`` / ``with_schema`` that only control
      where this subcommand writes its YAML (and are not part of the
      serialized config itself).

    All of these flat fields are ``exclude=True``, so dumping a
    ``ConfigCli`` via ``to_yaml`` produces a pure Workflow YAML —
    byte-for-byte identical to what ``Workflow.to_yaml`` would write.
    """

    # Hide the discriminated-union `search` from tyro; we rebuild it
    # from the flat source flags in `_assemble_search` below. The
    # `type: ignore` is because we're narrowing a required field to an
    # Optional one in the subclass. A non-empty description is required
    # so dolphin's YAML-comment writer (which walks the full schema,
    # including excluded fields) doesn't trip over a missing key.
    search: Annotated[Optional[Source], tyro.conf.Suppress] = Field(  # type: ignore[assignment]
        default=None,
        description=(
            "Source of input SLCs. Built by `_assemble_search` from the"
            " flat `--source`/`--start`/`--track`/... flags; hidden from"
            " the CLI because it's a discriminated union."
        ),
    )

    # Workflow declares these with `default_factory=lambda data: ...`
    # that reads already-validated data (for `<work_dir>/dem.tif` etc.).
    # Tyro materializes defaults at parse time before any data exists,
    # so it can't call those factories — we override with plain
    # `Optional[Path] = None` and the wrap validator strips them when
    # unset so Workflow's own factory takes over downstream.
    dem_filename: Optional[Path] = Field(  # type: ignore[assignment]
        default=None,
        description=(
            "DEM raster in EPSG:4326. Defaults to `<work_dir>/dem.tif`"
            " (downloaded via sardem)."
        ),
    )
    water_mask_filename: Optional[Path] = Field(  # type: ignore[assignment]
        default=None,
        description=(
            "Water mask in EPSG:4326 (uint8, 1=land, 0=water). Defaults"
            " to `<work_dir>/watermask.tif` (derived from the DEM)."
        ),
    )

    # --- Source-flat flags (folded into `search`, not dumped) ---

    start: Optional[str] = Field(
        default=None,
        description=(
            "Start date for the search (YYYY-MM-DD). Required for `safe`,"
            " `opera-cslc`, `nisar-gslc`; ignored for `local` (which uses"
            " whatever files already exist in --out-dir)."
        ),
        exclude=True,
    )
    end: Optional[str] = Field(
        default=None,
        description=(
            "End date for the search (YYYY-MM-DD). Required for `safe`,"
            " `opera-cslc`, `nisar-gslc`; ignored for `local`."
        ),
        exclude=True,
    )
    source: SourceKind = Field(
        default="safe",
        description=(
            "Where the input SLCs come from. `safe` (default): raw S1"
            " bursts via burst2safe + COMPASS. `local`: pre-downloaded"
            " full-frame S1 SAFE dirs / zips in --out-dir + COMPASS."
            " `opera-cslc`: pre-made OPERA CSLC HDF5s from ASF."
            " `nisar-gslc`: pre-made NISAR GSLC HDF5s via CMR (L-band,"
            " UTM, already geocoded)."
        ),
        exclude=True,
    )
    track: Optional[int] = Field(
        default=None,
        description=(
            "Relative orbit / track number. Required for --source safe;"
            " optional but recommended for opera-cslc and nisar-gslc."
            " For NISAR this is the `Track` field on ASF Vertex (the"
            " RRR digits in the granule filename)."
        ),
        exclude=True,
    )
    frame: Optional[int] = Field(
        default=None,
        description=(
            "NISAR track-frame number — the `Frame` field on ASF Vertex"
            " (the TTT digits in the granule filename, e.g. `71`). Only"
            " honored by --source nisar-gslc."
        ),
        exclude=True,
    )
    frequency: Literal["A", "B"] = Field(
        default="A",
        description=(
            "NISAR frequency band (`A` = L-band, `B` reserved). Only"
            " honored by --source nisar-gslc."
        ),
        exclude=True,
    )
    polarizations: list[str] = Field(
        default_factory=lambda: ["VV"],
        description=(
            "Polarizations to keep. Defaults to ['VV'] for S1/OPERA;"
            " pass --polarizations HH for NISAR."
        ),
        exclude=True,
    )
    swaths: Optional[list[str]] = Field(
        default=None,
        description=(
            "Restrict to specific subswaths (e.g. ['IW2']). Only"
            " honored by --source safe."
        ),
        exclude=True,
    )
    out_dir: Path = Field(
        default_factory=lambda: Path("data"),
        description="Where downloaded SLC inputs will live.",
        exclude=True,
    )

    do_tropo: bool = Field(
        default=False,
        description=(
            "Alias for `--tropo.enabled`. Kept for backwards"
            " compatibility with pre-refactor README/docs."
        ),
        exclude=True,
    )

    # --- CLI-only output knobs (not serialized) ---

    output: Path = Field(
        default=Path("sweets_config.yaml"),
        description="Where to write the config file.",
        exclude=True,
    )
    with_schema: bool = Field(
        default=True,
        description=(
            "Also write a sibling `<output>.schema.json` and prepend a"
            " `# yaml-language-server: $schema=...` modeline. Editors"
            " with the YAML Language Server (VS Code, Neovim-yamlls,"
            " JetBrains, etc.) use that for inline hover docs,"
            " autocomplete, and validation. Pass --no-with-schema to"
            " skip."
        ),
        exclude=True,
    )

    @model_validator(mode="wrap")
    @classmethod
    def _assemble_search(cls, data: Any, handler: Any) -> Any:
        """Build ``search`` from the flat source flags before the rest of validation.

        ``mode="wrap"`` runs around the entire validation pipeline, so
        the input dict is mutated before Workflow's mode="before"
        ``_sync_aoi`` validator sees it. That matters because
        ``_sync_aoi`` would crash on a missing / None ``search`` value;
        we need it to receive a fully-formed dict that encodes the flat
        CLI flags.

        The flat-flag path only triggers when the caller provided one
        of ``start`` / ``source`` (i.e. it's a CLI instantiation).
        Loading a serialized YAML via ``from_yaml`` passes ``search``
        directly and skips this branch entirely.
        """
        if isinstance(data, dict) and ("start" in data or "source" in data):
            data = dict(data)
            src = data.get("source", "safe")
            search: dict[str, Any] = {
                "kind": src,
                "out_dir": data.get("out_dir", Path("data")),
            }
            if src == "local":
                # LocalSafeSearch has no dates; files already on disk.
                pass
            else:
                if data.get("start") is None or data.get("end") is None:
                    msg = f"--start and --end are required for --source {src}"
                    raise ValueError(msg)
                search["start"] = data["start"]
                search["end"] = data["end"]
            if src == "safe":
                if data.get("track") is None:
                    msg = "--track is required for --source safe"
                    raise ValueError(msg)
                search["track"] = data["track"]
                search["polarizations"] = data.get("polarizations", ["VV"])
                if data.get("swaths") is not None:
                    search["swaths"] = data["swaths"]
            elif src == "opera-cslc":
                if data.get("track") is not None:
                    search["track"] = data["track"]
            elif src == "nisar-gslc":
                if data.get("track") is not None:
                    search["track"] = data["track"]
                if data.get("frame") is not None:
                    search["frame"] = data["frame"]
                search["frequency"] = data.get("frequency", "A")
                search["polarizations"] = data.get("polarizations", ["VV"])
            data["search"] = search

            if data.get("do_tropo"):
                tropo_obj: Any = data.get("tropo")
                dump = getattr(tropo_obj, "model_dump", None)
                if callable(dump):
                    tropo_dict: dict[str, Any] = dump()
                elif tropo_obj is None:
                    tropo_dict = {}
                else:
                    tropo_dict = dict(tropo_obj)
                tropo_dict["enabled"] = True
                data["tropo"] = tropo_dict

            # Strip CLI-only None overrides for fields whose Workflow
            # default_factory builds a `<work_dir>/...` path — leaving
            # them as None would override Workflow's factory with None.
            for key in ("dem_filename", "water_mask_filename"):
                if data.get(key) is None:
                    data.pop(key, None)

        return handler(data)

    def execute(self) -> None:
        """Write the config YAML (and, optionally, its schema sidecar).

        Serializes via a fresh :class:`Workflow` instance rather than
        dumping ``self`` directly. That matters because ConfigCli
        overrides ``dem_filename`` / ``water_mask_filename`` as
        ``Optional[Path] = None`` (tyro can't materialize Workflow's
        own ``default_factory=lambda data: ...`` at parse time), so
        dumping ``self`` would serialize ``null`` for those fields and
        a later ``Workflow.from_yaml`` would reject them. Stripping the
        None values and going through ``Workflow.model_validate`` lets
        Workflow's own defaults produce concrete ``<work_dir>/...``
        paths in the YAML.
        """
        if self.bbox is None and self.wkt is None:
            print("error: one of --bbox or --wkt is required", file=sys.stderr)
            raise SystemExit(2)
        payload = self.model_dump(by_alias=True)
        for key in ("dem_filename", "water_mask_filename"):
            if payload.get(key) is None:
                payload.pop(key, None)
        workflow = Workflow.model_validate(payload)
        workflow.to_yaml(self.output)
        if self.with_schema:
            _emit_schema_sidecar(self.output)
        print(f"wrote {self.output}", file=sys.stderr)


def _emit_schema_sidecar(yaml_path: Path) -> None:
    """Write a JSON schema next to the YAML and add a modeline comment.

    The schema is ``Workflow.model_json_schema()`` emitted verbatim;
    pydantic produces JSON Schema Draft 2020-12 with a ``oneOf +
    discriminator`` for the ``Workflow.search`` field, which the YAML
    Language Server handles natively. The modeline is read by the
    redhat.vscode-yaml extension (and every editor that speaks yamlls)
    to attach the schema at load time.
    """
    import json

    schema_path = yaml_path.with_suffix(yaml_path.suffix + ".schema.json")
    schema_path.write_text(json.dumps(Workflow.model_json_schema(), indent=2) + "\n")

    existing = yaml_path.read_text()
    modeline = f"# yaml-language-server: $schema={schema_path.name}\n"
    if modeline.strip() not in existing:
        yaml_path.write_text(modeline + existing)
    print(f"wrote {schema_path}", file=sys.stderr)


@dataclass
class SchemaCmd:
    """Dump the JSON schema for the sweets workflow config to stdout."""

    def execute(self) -> None:
        import json

        print(json.dumps(Workflow.model_json_schema(), indent=2))


@dataclass
class ReportCmd:
    """Render a single-file HTML report for a finished sweets run."""

    config_file: Annotated[Path, tyro.conf.Positional]
    """Path to the sweets_config.yaml used for the run. A work directory
    containing a sweets_config.yaml is also accepted and resolved to the
    yaml inside it."""

    output: Optional[Path] = None
    """Where to write the report. Defaults to `<work_dir>/sweets_report.html`."""

    def execute(self) -> None:
        from sweets._report import build_report

        path = build_report(
            config_file=self.config_file,
            output=self.output,
        )
        print(f"wrote {path}", file=sys.stderr)


@dataclass
class RunCmd:
    """Execute a sweets workflow from a config file."""

    config_file: Annotated[Path, tyro.conf.Positional]
    """Path to a sweets_config.yaml."""

    starting_step: int = 1
    """Skip earlier stages (1=download, 2=geocode, 3=dolphin)."""

    def execute(self) -> None:
        """Load the workflow and run it."""
        if not self.config_file.exists():
            msg = f"config file {self.config_file} does not exist"
            raise SystemExit(msg)
        workflow = Workflow.from_yaml(self.config_file)
        workflow.run(starting_step=self.starting_step)


@dataclass
class ServerCmd:
    """Launch the sweets web UI (FastAPI backend + bundled React frontend).

    Install the optional web extras first::

        pip install -e ".[web]"   # or: pixi install -e <env-with-web-extras>

    Then run ``sweets server`` to start uvicorn on http://localhost:8000.
    During frontend development, run ``npm run dev`` inside
    ``src/sweets/web/frontend/`` for hot-reloaded React on :5173 (Vite
    proxies ``/api`` to :8000).
    """

    host: str = "127.0.0.1"
    """Bind address. Use 0.0.0.0 to expose on the network."""

    port: int = 8000
    """TCP port."""

    reload: bool = False
    """Enable uvicorn auto-reload (development only)."""

    def execute(self) -> None:
        try:
            import uvicorn
        except ImportError as e:
            msg = (
                "sweets server requires the `web` extras. Install via:\n"
                '    pip install -e ".[web]"\n'
                f"(original error: {e})"
            )
            raise SystemExit(msg) from e

        uvicorn.run(
            "sweets.web.app:app",
            host=self.host,
            port=self.port,
            reload=self.reload,
        )


def main() -> None:
    """Top-level CLI entry point."""
    cmd = tyro.extras.subcommand_cli_from_dict(
        {
            "config": ConfigCli,
            "run": RunCmd,
            "schema": SchemaCmd,
            "report": ReportCmd,
            "server": ServerCmd,
        },
        prog="sweets",
        description="Sentinel-1 InSAR workflow runner.",
    )
    cmd.execute()


if __name__ == "__main__":
    main()
