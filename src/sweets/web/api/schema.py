"""JSON Schema endpoints for sweets workflow configurations.

Exposes schema for both the full displacement workflow and the lighter
interferogram-only workflow so the frontend can auto-generate config forms.
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("")
def workflow_schema() -> dict:
    """Return the JSON Schema for ``sweets.core.Workflow`` (displacement)."""
    from sweets.core import Workflow

    return Workflow.model_json_schema()


@router.get("/ifg")
def ifg_workflow_schema() -> dict:
    """Return the JSON Schema for ``sweets.ifg.IfgWorkflow`` (interferogram)."""
    from sweets.ifg import IfgWorkflow

    return IfgWorkflow.model_json_schema()
