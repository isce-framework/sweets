"""Job model for tracking workflow executions."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import field_validator
from sqlmodel import Column, Field, SQLModel
from sqlalchemy import JSON


class JobStatus(str, Enum):
    """Status of a workflow job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobBase(SQLModel):
    """Base job fields shared between create/update/read."""

    name: str = Field(index=True)
    config: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    work_dir: str | None = None


class Job(JobBase, table=True):  # type: ignore[call-arg]
    """Job database model."""

    id: int | None = Field(default=None, primary_key=True)
    status: JobStatus = Field(default=JobStatus.PENDING)
    current_step: int = Field(default=0)
    error_message: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    pid: int | None = None  # Process ID for running jobs


class JobCreate(JobBase):
    """Schema for creating a new job."""

    @field_validator("config", mode="before")
    @classmethod
    def validate_config(cls, v: dict[str, Any]) -> dict[str, Any]:
        # Basic validation - more thorough validation happens in Workflow
        assert "bbox" in v or "wkt" in v, "Must specify bbox or wkt"
        return v


class JobUpdate(SQLModel):
    """Schema for updating a job.

    User-facing fields only. ``pid`` and ``status`` are internal — they're
    set by the executor / cancel endpoint and shouldn't be writable via
    PATCH (a client setting ``pid=12345`` could mis-target a SIGTERM).
    """

    name: str | None = None
    error_message: str | None = None


class JobRead(JobBase):
    """Schema for reading a job (response model)."""

    id: int
    status: JobStatus
    current_step: int
    error_message: str | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
