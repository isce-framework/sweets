"""Job CRUD API endpoints."""

from __future__ import annotations

import os
import signal
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from sweets.web.models import Job, JobCreate, JobRead, JobStatus, JobUpdate
from sweets.web.models.database import get_session
from sweets.web.services.executor import start_job_background
from sweets.web.services.log_manager import log_manager

router = APIRouter()

SessionDep = Annotated[Session, Depends(get_session)]


@router.get("/", response_model=list[JobRead])
def list_jobs(
    session: SessionDep,
    skip: int = 0,
    limit: int = 100,
    status: JobStatus | None = None,
):
    """List all jobs, optionally filtered by status."""
    query = (
        select(Job)
        .offset(skip)
        .limit(limit)
        .order_by(Job.created_at.desc())  # type: ignore[attr-defined]
    )
    if status:
        query = query.where(Job.status == status)
    return session.exec(query).all()


@router.post("/", response_model=JobRead)
def create_job(job: JobCreate, session: SessionDep):
    """Create a new job (does not start it)."""
    db_job = Job.model_validate(job)
    session.add(db_job)
    session.commit()
    session.refresh(db_job)
    return db_job


@router.get("/{job_id}", response_model=JobRead)
def get_job(job_id: int, session: SessionDep):
    """Get a job by ID."""
    job = session.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.patch("/{job_id}", response_model=JobRead)
def update_job(job_id: int, job_update: JobUpdate, session: SessionDep):
    """Update a job."""
    job = session.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    update_data = job_update.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(job, key, value)

    session.add(job)
    session.commit()
    session.refresh(job)
    return job


@router.delete("/{job_id}")
def delete_job(job_id: int, session: SessionDep):
    """Delete a job."""
    job = session.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Don't allow deleting running jobs
    if job.status == JobStatus.RUNNING:
        raise HTTPException(status_code=400, detail="Cannot delete a running job")

    # Clear log buffer
    log_manager.clear_buffer(job_id)

    session.delete(job)
    session.commit()
    return {"ok": True}


@router.post("/{job_id}/start", response_model=JobRead)
def start_job(job_id: int, session: SessionDep):
    """Start a pending job."""
    job = session.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status != JobStatus.PENDING:
        raise HTTPException(
            status_code=400,
            detail=f"Job is {job.status.value}, can only start pending jobs",
        )

    # Start in background thread with log streaming
    start_job_background(job_id, job.config)

    # Refresh to get updated status
    session.refresh(job)
    return job


@router.post("/{job_id}/cancel", response_model=JobRead)
def cancel_job(job_id: int, session: SessionDep):
    """Cancel a running job."""
    job = session.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status != JobStatus.RUNNING:
        raise HTTPException(status_code=400, detail="Job is not running")

    if job.pid:
        try:
            os.kill(job.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass  # Already dead

    job.status = JobStatus.CANCELLED
    job.completed_at = datetime.utcnow()
    job.pid = None

    log_manager.append_log(job_id, "Job cancelled by user")

    session.add(job)
    session.commit()
    session.refresh(job)
    return job


@router.get("/{job_id}/logs")
def get_job_logs(job_id: int, session: SessionDep):
    """Get buffered logs for a job (for non-WebSocket clients)."""
    job = session.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    buffer = log_manager.get_buffer(job_id)
    return {
        "job_id": job_id,
        "lines": buffer.get_history(),
        "current_step": buffer.current_step,
    }
