"""Smoke tests for the sweets web API.

Only exercises the always-available endpoints (no live ASF/CMR, no
subprocess spawn). Skipped wholesale when the ``web`` extras (fastapi,
sqlmodel, ...) aren't installed.
"""

from __future__ import annotations

import importlib.util

import pytest

if any(
    importlib.util.find_spec(mod) is None for mod in ("fastapi", "sqlmodel", "httpx")
):
    pytest.skip("web extras not installed", allow_module_level=True)

from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A TestClient bound to an isolated on-disk SQLite DB."""
    db_path = tmp_path / "sweets.db"
    monkeypatch.setattr(
        "sweets.web.models.database.DEFAULT_DB_PATH", db_path, raising=True
    )
    from sqlmodel import SQLModel, create_engine

    fresh_engine = create_engine(f"sqlite:///{db_path}", echo=False)
    monkeypatch.setattr("sweets.web.models.database.engine", fresh_engine, raising=True)
    # Repoint the modules that captured `engine` at import time.
    import sweets.web.api.jobs as jobs_mod
    import sweets.web.api.websocket as ws_mod
    import sweets.web.app as app_mod
    import sweets.web.services.executor as exec_mod

    for mod in (jobs_mod, ws_mod, exec_mod, app_mod):
        monkeypatch.setattr(mod, "engine", fresh_engine, raising=False)
    SQLModel.metadata.create_all(fresh_engine)

    return TestClient(app_mod.app)


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_schema_endpoint(client):
    r = client.get("/api/schema")
    assert r.status_code == 200
    schema = r.json()
    # The Workflow schema must expose the discriminated `search` union so
    # the frontend's RJSF form picks the right oneOf branch.
    assert "properties" in schema
    assert "search" in schema["properties"]


def test_create_job_rejects_missing_aoi(client):
    """JobCreate's validator must reject a config with no bbox/wkt."""
    r = client.post("/api/jobs/", json={"name": "j", "config": {}})
    assert r.status_code in (400, 422)


def test_stale_running_jobs_are_swept(client):
    """A RUNNING job left over from a previous server run should be reaped.

    Simulates the case where the server crashed mid-job: pid is dead (or
    never existed), but the DB row still says RUNNING. The lifespan sweep
    must flip it to FAILED so the user isn't stuck.
    """
    from sweets.web.app import _sweep_stale_jobs
    from sweets.web.models import Job, JobStatus
    from sweets.web.models.database import engine
    from sqlmodel import Session

    with Session(engine) as session:
        stuck = Job(
            name="zombie",
            config={"bbox": [0.0, 0.0, 1.0, 1.0]},
            status=JobStatus.RUNNING,
            pid=0,  # PID 0 raises OSError on os.kill; that path is exercised
        )
        session.add(stuck)
        session.commit()
        session.refresh(stuck)
        stuck_id = stuck.id

    _sweep_stale_jobs()

    r = client.get(f"/api/jobs/{stuck_id}")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "failed"
    assert "server restart" in (body.get("error_message") or "")


def test_job_lifecycle_without_spawn(client):
    """Create → list → fetch → delete a job without ever running it."""
    payload = {
        "name": "smoke",
        "config": {"bbox": [-118.5, 34.0, -118.0, 34.3]},
    }
    r = client.post("/api/jobs/", json=payload)
    assert r.status_code == 200, r.text
    job = r.json()
    job_id = job["id"]
    assert job["status"] == "pending"

    r = client.get("/api/jobs/")
    assert r.status_code == 200
    assert any(j["id"] == job_id for j in r.json())

    r = client.get(f"/api/jobs/{job_id}")
    assert r.status_code == 200
    assert r.json()["name"] == "smoke"

    # A pending job has no work_dir yet — the manifest should report
    # exists=False rather than 500.
    r = client.get(f"/api/jobs/{job_id}/manifest")
    assert r.status_code == 200
    assert r.json()["exists"] is False

    r = client.delete(f"/api/jobs/{job_id}")
    assert r.status_code == 200

    r = client.get(f"/api/jobs/{job_id}")
    assert r.status_code == 404
