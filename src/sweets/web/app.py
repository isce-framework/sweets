"""FastAPI application for sweets web UI."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from sweets.web.api import jobs, results, schema, search, websocket
from sweets.web.models.database import create_db_and_tables


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database on startup."""
    create_db_and_tables()
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

# API routes
app.include_router(jobs.router, prefix="/api/jobs", tags=["jobs"])
app.include_router(results.router, prefix="/api/jobs", tags=["results"])
app.include_router(schema.router, prefix="/api/schema", tags=["schema"])
app.include_router(search.router, prefix="/api/search", tags=["search"])
app.include_router(websocket.router, prefix="/api/ws", tags=["websocket"])


# Serve static frontend (production)
STATIC_DIR = Path(__file__).parent / "frontend" / "dist"
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}
