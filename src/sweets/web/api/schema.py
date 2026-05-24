"""JSON Schema endpoint for the sweets workflow.

Surfaces ``sweets.core.Workflow.model_json_schema()`` so the frontend can
auto-generate a configuration form (currently consumed by react-jsonschema-form
on the React side).
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("")
def workflow_schema() -> dict:
    """Return the JSON Schema for ``sweets.core.Workflow``."""
    from sweets.core import Workflow

    return Workflow.model_json_schema()
