"""WebSocket endpoints for live log streaming."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlmodel import Session

from sweets.web.models import Job, JobStatus
from sweets.web.models.database import engine
from sweets.web.services.log_manager import log_manager

router = APIRouter()


@router.websocket("/jobs/{job_id}/logs")
async def job_logs_websocket(websocket: WebSocket, job_id: int):
    """WebSocket endpoint for streaming job logs.

    Sends:
    - {"type": "history", "lines": [...], "step": N} on connect
    - {"type": "log", "line": "...", "step": N} for each new line
    - {"type": "status", "status": "completed|failed|cancelled"} when job ends
    """
    await websocket.accept()

    # Verify job exists
    with Session(engine) as session:
        job = session.get(Job, job_id)
        if not job:
            await websocket.send_json({"type": "error", "message": "Job not found"})
            await websocket.close()
            return

        initial_status = job.status

    # Get log buffer and subscribe
    buffer = log_manager.get_buffer(job_id)
    queue = buffer.subscribe()

    try:
        # Send history first
        history = buffer.get_history()
        await websocket.send_json(
            {
                "type": "history",
                "lines": history,
                "step": buffer.current_step,
            }
        )

        # If job already finished, send final status and close
        if initial_status in (
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
        ):
            await websocket.send_json(
                {
                    "type": "status",
                    "status": initial_status.value,
                }
            )
            return

        # Stream new logs
        while True:
            try:
                # Wait for new log with timeout to check job status
                msg = await asyncio.wait_for(queue.get(), timeout=2.0)
                await websocket.send_json(msg)
            except asyncio.TimeoutError:
                # Check if job finished
                with Session(engine) as session:
                    job = session.get(Job, job_id)
                    if job and job.status in (
                        JobStatus.COMPLETED,
                        JobStatus.FAILED,
                        JobStatus.CANCELLED,
                    ):
                        await websocket.send_json(
                            {
                                "type": "status",
                                "status": job.status.value,
                            }
                        )
                        break

    except WebSocketDisconnect:
        pass
    finally:
        buffer.unsubscribe(queue)
