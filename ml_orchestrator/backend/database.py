"""
Shared DB setup used by all four verticals.
SQLite for local dev/demo; schema.sql shows the Postgres-grade version.
"""
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

import os

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_DB_PATH = os.path.join(_BASE_DIR, "scheduler.db")
SQLALCHEMY_DATABASE_URL = f"sqlite:///{_DB_PATH}"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False, "timeout": 30},
)


@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """
    Force DELETE journal mode so every commit is written directly into the
    .db file — no WAL sidecar.  This makes the database immediately visible
    to external viewers (SQLite Viewer, DB Browser, etc.) without needing
    to merge a -wal file.
    """
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=DELETE")
    cursor.execute("PRAGMA synchronous=FULL")
    cursor.close()


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
