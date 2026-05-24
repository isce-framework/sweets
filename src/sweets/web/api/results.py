"""Job results: on-disk manifest + bowser handoff.

The manifest endpoint walks the job's working directory and reports the
"interesting" output files (GSLCs, dolphin outputs, DEM, water mask, etc.)
so the UI can show a results panel without re-deriving paths from the
config schema.

The bowser endpoint shells out to ``bowser setup-dolphin <dolphin_dir>``
to convert dolphin outputs into bowser-readable form and (optionally) starts
a `bowser` dev/server process so the user can click straight into the viewer.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session

from sweets.web.models import Job
from sweets.web.models.database import get_session

router = APIRouter()

SessionDep = Annotated[Session, Depends(get_session)]


# Globs of "interesting" files relative to the work_dir. Keep this list short
# and focused — the frontend renders them inline, so dumping every file in
# the work_dir would be noisy.
MANIFEST_GLOBS: list[tuple[str, str]] = [
    ("dem", "dem.tif"),
    ("watermask", "watermask.tif"),
    ("config", "sweets_config.yaml"),
    ("report", "sweets_report.html"),
    ("gslc", "gslcs/**/t*.h5"),
    ("opera-cslc", "data/OPERA_L2_CSLC-S1_*.h5"),
    ("nisar-gslc", "data/NISAR_L2_GSLC_*.h5"),
    ("geometry", "geometry/*.tif"),
    ("dolphin-unwrapped", "dolphin/unwrapped/*.tif"),
    ("dolphin-timeseries", "dolphin/timeseries/*.tif"),
    ("dolphin-velocity", "dolphin/timeseries/velocity.tif"),
]


def _resolve_work_dir(job: Job) -> Path | None:
    if job.work_dir:
        return Path(job.work_dir)
    cfg = job.config or {}
    wd = cfg.get("work_dir")
    if wd:
        return Path(wd)
    return None


@router.get("/{job_id}/manifest")
def get_manifest(job_id: int, session: SessionDep) -> dict:
    """List interesting output files under the job's work_dir."""
    job = session.get(Job, job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    work_dir = _resolve_work_dir(job)
    if work_dir is None or not work_dir.exists():
        return {
            "job_id": job_id,
            "work_dir": str(work_dir) if work_dir else None,
            "exists": False,
            "entries": [],
        }

    seen: set[Path] = set()
    entries: list[dict] = []
    for kind, pattern in MANIFEST_GLOBS:
        for p in sorted(work_dir.glob(pattern)):
            if p in seen or not p.is_file():
                continue
            seen.add(p)
            entries.append(
                {
                    "path": str(p.relative_to(work_dir)),
                    "size": p.stat().st_size,
                    "kind": kind,
                }
            )

    return {
        "job_id": job_id,
        "work_dir": str(work_dir),
        "exists": True,
        "entries": entries,
    }


@router.post("/{job_id}/view")
def bowser_view(
    job_id: int,
    session: SessionDep,
    autostart: bool = False,
    port: int = 4280,
) -> dict:
    """Run ``bowser setup-dolphin <dolphin_dir>`` and (optionally) start bowser.

    Always returns the command we would have run plus the resolved
    ``dolphin_dir``, so the UI can show the exact CLI invocation to the user
    even when ``bowser`` isn't on PATH. When ``autostart`` is true and
    ``bowser`` is available, kicks off a detached ``bowser serve`` and returns
    the local URL.
    """
    job = session.get(Job, job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    work_dir = _resolve_work_dir(job)
    if work_dir is None:
        raise HTTPException(400, "Job has no resolved work_dir")
    dolphin_dir = work_dir / "dolphin"
    cmd = f"bowser setup-dolphin {dolphin_dir}"

    bowser_bin = shutil.which("bowser")
    if bowser_bin is None or not dolphin_dir.exists():
        return {
            "job_id": job_id,
            "dolphin_dir": str(dolphin_dir),
            "command": cmd,
            "url": None,
            "ran": False,
            "stdout": "",
            "stderr": (
                "bowser CLI not found on PATH"
                if bowser_bin is None
                else f"{dolphin_dir} does not exist yet"
            ),
        }

    proc = subprocess.run(
        [bowser_bin, "setup-dolphin", str(dolphin_dir)],
        capture_output=True,
        text=True,
        check=False,
    )

    url = None
    if autostart and proc.returncode == 0:
        # Best-effort: kick off `bowser serve` detached and surface the URL.
        # Failures here aren't fatal — the manual command still works.
        try:
            subprocess.Popen(  # noqa: S603 - controlled args
                [bowser_bin, "serve", "--port", str(port)],
                cwd=str(dolphin_dir),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                env={**os.environ},
            )
            url = f"http://localhost:{port}/"
        except FileNotFoundError:
            pass

    return {
        "job_id": job_id,
        "dolphin_dir": str(dolphin_dir),
        "command": cmd,
        "url": url,
        "ran": proc.returncode == 0,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
