"""Job execution service with log streaming."""

from __future__ import annotations

import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

import yaml
from sqlmodel import Session

from sweets.web.models import Job, JobStatus
from sweets.web.models.database import engine
from sweets.web.services.log_manager import log_manager


def _stream_output(proc: subprocess.Popen, job_id: int):
    """Read subprocess output line by line and stream to log manager."""
    assert proc.stdout is not None
    for line in iter(proc.stdout.readline, ""):
        if not line:
            break
        line = line.rstrip("\n\r")
        log_manager.append_log(job_id, line)
    proc.stdout.close()


def run_workflow_sync(job_id: int, config: dict):
    """Run a workflow synchronously (call from background thread/task).

    This function:
    1. Writes config to a temp YAML file
    2. Spawns `sweets run` subprocess
    3. Streams stdout to log_manager line by line
    4. Updates job status in database when complete
    """
    # Write config to temp file
    with NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, prefix=f"sweets_job_{job_id}_"
    ) as f:
        yaml.safe_dump(config, f)
        config_path = Path(f.name)

    with Session(engine) as session:
        job = session.get(Job, job_id)
        assert job is not None

        job.status = JobStatus.RUNNING
        job.started_at = datetime.utcnow()
        # Cache the resolved work_dir on the Job row so the manifest /
        # bowser-handoff endpoints can find it without re-parsing the config.
        if not job.work_dir:
            cfg_wd = config.get("work_dir")
            if cfg_wd:
                job.work_dir = str(cfg_wd)
        session.add(job)
        session.commit()

    log_manager.append_log(job_id, f"Starting workflow for job {job_id}")
    log_manager.append_log(job_id, f"Config: {config_path}")

    try:
        # Run sweets as subprocess
        proc = subprocess.Popen(
            [sys.executable, "-m", "sweets", "run", str(config_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # Line buffered
        )

        # Update PID in database
        with Session(engine) as session:
            job = session.get(Job, job_id)
            assert job is not None
            job.pid = proc.pid
            session.add(job)
            session.commit()

        log_manager.append_log(job_id, f"Process started with PID {proc.pid}")

        # Stream output in a separate thread (so we can handle it async if needed)
        _stream_output(proc, job_id)

        # Wait for process to complete
        return_code = proc.wait()

        if return_code == 0:
            final_status = JobStatus.COMPLETED
            log_manager.append_log(job_id, "Workflow completed successfully")
        else:
            final_status = JobStatus.FAILED
            log_manager.append_log(
                job_id, f"Workflow failed with exit code {return_code}"
            )

    except Exception as e:
        final_status = JobStatus.FAILED
        log_manager.append_log(job_id, f"Error: {e}")

    finally:
        # Update job status
        with Session(engine) as session:
            job = session.get(Job, job_id)
            assert job is not None
            job.status = final_status
            job.completed_at = datetime.utcnow()
            job.current_step = log_manager.get_current_step(job_id)
            job.pid = None

            # Store last N lines as error message if failed
            if final_status == JobStatus.FAILED:
                history = log_manager.get_buffer(job_id).get_history()
                job.error_message = "\n".join(history[-50:])

            session.add(job)
            session.commit()

        # Clean up temp config file
        config_path.unlink(missing_ok=True)


def start_job_background(job_id: int, config: dict):
    """Start a job in a background thread."""
    thread = threading.Thread(
        target=run_workflow_sync,
        args=(job_id, config),
        daemon=True,
    )
    thread.start()
    return thread
