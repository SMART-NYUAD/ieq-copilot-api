"""Shared PostgreSQL connection helpers for API services."""

from contextlib import contextmanager
from typing import Iterator

import psycopg2
import psycopg2.extras

try:
    from storage.db_config import DATABASE_URL
except ImportError:
    from .db_config import DATABASE_URL


@contextmanager
def get_connection() -> Iterator[psycopg2.extensions.connection]:
    conn = psycopg2.connect(DATABASE_URL)
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def get_cursor(real_dict: bool = True):
    with get_connection() as conn:
        cursor_factory = psycopg2.extras.RealDictCursor if real_dict else None
        with conn.cursor(cursor_factory=cursor_factory) as cur:
            yield cur

