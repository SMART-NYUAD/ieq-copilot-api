"""Shared PostgreSQL connection helpers for API services."""

from contextlib import contextmanager
import os
from threading import Lock
from typing import Iterator

import psycopg2
import psycopg2.extras
from psycopg2.pool import PoolError, ThreadedConnectionPool

try:
    from storage.db_config import DATABASE_URL
except ImportError:
    from .db_config import DATABASE_URL


_POOL_LOCK = Lock()
_POOL: ThreadedConnectionPool | None = None


def _pool_minconn() -> int:
    raw = str(os.getenv("POSTGRES_POOL_MINCONN", "1")).strip()
    try:
        value = int(raw)
    except ValueError:
        value = 1
    return max(1, value)


def _pool_maxconn() -> int:
    raw = str(os.getenv("POSTGRES_POOL_MAXCONN", "12")).strip()
    try:
        value = int(raw)
    except ValueError:
        value = 12
    return max(_pool_minconn(), value)


def _get_pool() -> ThreadedConnectionPool:
    global _POOL
    if _POOL is None:
        with _POOL_LOCK:
            if _POOL is None:
                _POOL = ThreadedConnectionPool(
                    minconn=_pool_minconn(),
                    maxconn=_pool_maxconn(),
                    dsn=DATABASE_URL,
                )
    return _POOL


def close_connection_pool() -> None:
    """Close all pooled connections.

    Useful for tests and controlled shutdown flows.
    """
    global _POOL
    with _POOL_LOCK:
        if _POOL is not None:
            _POOL.closeall()
            _POOL = None


@contextmanager
def get_connection() -> Iterator[psycopg2.extensions.connection]:
    pool = _get_pool()
    try:
        conn = pool.getconn()
    except PoolError:
        # Keep behavior resilient if pool is temporarily unavailable.
        conn = psycopg2.connect(DATABASE_URL)
        using_pool = False
    else:
        using_pool = True
    try:
        yield conn
        try:
            conn.commit()
        except Exception:
            pass
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    finally:
        if using_pool:
            close_conn = bool(getattr(conn, "closed", False))
            pool.putconn(conn, close=close_conn)
        else:
            conn.close()


@contextmanager
def get_cursor(real_dict: bool = True):
    with get_connection() as conn:
        cursor_factory = psycopg2.extras.RealDictCursor if real_dict else None
        with conn.cursor(cursor_factory=cursor_factory) as cur:
            yield cur

