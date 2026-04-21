"""
db.py — SQLAlchemy Engine, Session, Base & get_db for SHIELD
=============================================================
Uses SQLite by default (zero config). Switch to PostgreSQL
by changing DATABASE_URL in your .env file.
"""

from __future__ import annotations

from typing import Generator

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from config import settings


# ── Engine ────────────────────────────────────────────────────────────────────

connect_args = {"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {}

engine = create_engine(
    settings.DATABASE_URL,
    connect_args  = connect_args,
    pool_pre_ping = True,
    echo          = False,
)


@event.listens_for(engine, "connect")
def _sqlite_pragmas(dbapi_conn, _):
    """Enable WAL + foreign keys for SQLite (ignored for other DBs)."""
    if "sqlite" in settings.DATABASE_URL:
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()


# ── Session Factory ───────────────────────────────────────────────────────────

SessionLocal = sessionmaker(
    bind             = engine,
    autocommit       = False,
    autoflush        = False,
    expire_on_commit = False,
)


# ── Declarative Base ──────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


# ── FastAPI / Streamlit Dependency ────────────────────────────────────────────

def get_db() -> Generator[Session, None, None]:
    """Yield a DB session; always closes on exit."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def get_session() -> Session:
    """
    Return a plain session (not a generator).
    Use this in Streamlit pages which aren't FastAPI routes.

    Usage:
        with get_session() as db:
            db.query(...)
    """
    return SessionLocal()


def init_db() -> None:
    """Create all tables. Called once at app startup."""
    from models import Base as ModelBase   # noqa: F401 — registers all models
    Base.metadata.create_all(bind=engine)


def check_connection() -> bool:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
