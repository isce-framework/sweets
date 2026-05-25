"""FastAPI application for sweets web UI."""

from __future__ import annotations

import logging
import os
import signal
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlmodel import Session, select

from sweets.web.api import jobs, results, schema, search, websocket
from sweets.web.models import Job, JobStatus
from sweets.web.models.database import create_db_and_tables, engine

logger = logging.getLogger(__name__)


def _sweep_stale_jobs() -> None:
    """Mark RUNNING jobs as FAILED on startup.

    If the server crashed or was restarted mid-job the row stays "running"
    forever — the subprocess is orphaned and we've lost its stdout pipe,
    so we can't reattach or stream logs anyway. Best-effort SIGTERM the
    pid if it's still alive (so we don't leak the orphan), then flip the
    status to FAILED with a note so the user knows to re-create.
    """
    with Session(engine) as session:
        stale = session.exec(select(Job).where(Job.status == JobStatus.RUNNING)).all()
        for job in stale:
            if job.pid:
                try:
                    os.kill(job.pid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    pass
            job.status = JobStatus.FAILED
            job.pid = None
            note = "[server restart: orphaned subprocess, status unknown]"
            job.error_message = (
                f"{job.error_message}\n{note}" if job.error_message else note
            )
            session.add(job)
        session.commit()
        if stale:
            logger.warning(
                "Marked %d stale RUNNING job(s) as FAILED on startup", len(stale)
            )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database and reconcile orphaned jobs on startup."""
    create_db_and_tables()
    _sweep_stale_jobs()
    yield


app = FastAPI(
    title="Sweets",
    description="Web UI for InSAR processing workflows",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS for local development (Vite dev server on different port)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


# API routes
app.include_router(jobs.router, prefix="/api/jobs", tags=["jobs"])
app.include_router(results.router, prefix="/api/jobs", tags=["results"])
app.include_router(schema.router, prefix="/api/schema", tags=["schema"])
app.include_router(search.router, prefix="/api/search", tags=["search"])
app.include_router(websocket.router, prefix="/api/ws", tags=["websocket"])


# Serve static frontend (production) — keep this LAST so the catch-all
# `/` mount doesn't shadow `/api/*` routes registered above.
STATIC_DIR = Path(__file__).parent / "frontend" / "dist"
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
