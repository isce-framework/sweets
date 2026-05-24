"""Web UI for sweets - FastAPI backend with Svelte frontend."""

from __future__ import annotations

__all__ = ["create_app"]


def create_app():
    """Create and configure the FastAPI application."""
    from sweets.web.app import app

    return app
