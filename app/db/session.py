"""
SQLAlchemy engine and session factory with SQLite/PostgreSQL support.
"""

from pathlib import Path
from typing import Optional

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.db.models import Base

logger = get_logger(__name__)


def create_db_engine(database_url: Optional[str] = None, settings: Optional[Settings] = None) -> Engine:
    """Create a SQLAlchemy engine for the configured database URL."""
    settings = settings or get_settings()
    url = database_url or settings.database_url

    connect_args = {}
    if url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
        db_path = url.replace("sqlite:///", "", 1)
        if db_path and db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    engine = create_engine(url, echo=settings.database_echo, connect_args=connect_args, future=True)

    if url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def _enable_sqlite_fk(dbapi_connection, _record):  # type: ignore[no-untyped-def]
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return engine


def init_database(engine: Engine) -> None:
    """Create all tables if they do not exist (idempotent initialization)."""
    Base.metadata.create_all(engine)
    logger.info("Database schema initialized")


def get_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Return a session factory bound to the engine."""
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)
