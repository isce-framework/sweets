"""Database setup and session management."""

from __future__ import annotations

from pathlib import Path

from sqlmodel import Session, SQLModel, create_engine

# Default to ~/.sweets/sweets.db
DEFAULT_DB_PATH = Path.home() / ".sweets" / "sweets.db"


def get_database_url(db_path: Path | None = None) -> str:
    """Get SQLite database URL."""
    path = db_path or DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path}"


engine = create_engine(get_database_url(), echo=False)


def create_db_and_tables():
    """Create all database tables."""
    SQLModel.metadata.create_all(engine)


def get_session():
    """Yield a database session."""
    with Session(engine) as session:
        yield session
