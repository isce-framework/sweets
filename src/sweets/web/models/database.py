"""Database setup and session management."""

from __future__ import annotations

from pathlib import Path

from sqlmodel import Session, SQLModel, create_engine

# Default to ~/.sweets/sweets.db
DEFAULT_DB_PATH = Path.home() / ".sweets" / "sweets.db"


def _database_url() -> str:
    DEFAULT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{DEFAULT_DB_PATH}"


engine = create_engine(_database_url(), echo=False)


def create_db_and_tables():
    """Create all database tables."""
    SQLModel.metadata.create_all(engine)


def get_session():
    """Yield a database session."""
    with Session(engine) as session:
        yield session
