"""Standalone database configuration helpers for API services."""

from __future__ import annotations

import os

try:
    from core_settings import ensure_env_loaded
except ImportError:
    from ..core_settings import ensure_env_loaded


def load_database_url() -> str:
    """Resolve DATABASE_URL from environment.

    Put ``DATABASE_URL=...`` in ``RAG_API_SERVER/.env``, or set individual
    ``DB_USER``, ``DB_PASSWORD``, ``DB_HOST``, ``DB_PORT``, ``DB_NAME`` there
    and omit ``DATABASE_URL`` to build a default URL.
    """
    ensure_env_loaded()

    from_env = str(os.getenv("DATABASE_URL", "")).strip()
    if from_env:
        return from_env

    db_user = str(os.getenv("DB_USER", "")).strip()
    db_password = str(os.getenv("DB_PASSWORD", "")).strip()
    db_host = str(os.getenv("DB_HOST", "")).strip()
    db_port = str(os.getenv("DB_PORT", "")).strip()
    db_name = str(os.getenv("DB_NAME", "")).strip()

    missing = [
        key
        for key, value in (
            ("DB_USER", db_user),
            ("DB_PASSWORD", db_password),
            ("DB_HOST", db_host),
            ("DB_PORT", db_port),
            ("DB_NAME", db_name),
        )
        if not value
    ]
    if missing:
        missing_keys = ", ".join(missing)
        raise RuntimeError(
            "Database credentials are not configured. Set DATABASE_URL or all DB_* keys in .env "
            f"(missing: {missing_keys})."
        )
    return f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"


DATABASE_URL = load_database_url()
