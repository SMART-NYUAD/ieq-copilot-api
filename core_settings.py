"""Centralized runtime settings for the RAG API server.

This module intentionally keeps a small, stable configuration surface for the
first refactor phase. Additional settings can be added incrementally without
changing call sites by extending ``AppSettings``.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import List

_ENV_LOADED = False


def _parse_bool(value: str, default: bool) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return default
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _parse_float(value: str, default: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def _parse_int(value: str, default: int, minimum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, parsed)


def _parse_csv_list(value: str, default: List[str]) -> List[str]:
    raw = str(value or "").strip()
    if not raw:
        return list(default)
    items = [item.strip() for item in raw.split(",")]
    cleaned = [item for item in items if item]
    return cleaned or list(default)


def _load_env_file_if_present() -> None:
    """Load simple KEY=VALUE pairs from local .env once per process.

    This avoids an extra dependency while still giving predictable local
    behavior when users create `.env` from `.env.example`.
    """
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return
    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            if key and key not in os.environ:
                os.environ[key] = value
    except Exception:
        # Keep startup resilient even if .env parsing fails.
        return


def ensure_env_loaded() -> None:
    """Public helper for modules that rely on `.env` keys."""
    _load_env_file_if_present()


@dataclass(frozen=True)
class AppSettings:
    """Top-level app settings shared by HTTP and orchestration layers."""

    server_host: str
    server_port: int
    cors_allow_origins: List[str]
    cors_allow_credentials: bool
    cors_allow_methods: List[str]
    cors_allow_headers: List[str]
    router_clarify_threshold: float


def load_settings() -> AppSettings:
    """Load settings from environment using safe defaults.

    Defaults intentionally preserve current behavior to avoid runtime regressions
    during the refactor.
    """
    ensure_env_loaded()
    return AppSettings(
        server_host=str(os.getenv("RAG_API_HOST", "0.0.0.0")).strip() or "0.0.0.0",
        server_port=_parse_int(os.getenv("RAG_API_PORT", "8001"), default=8001, minimum=1),
        cors_allow_origins=_parse_csv_list(os.getenv("RAG_API_CORS_ALLOW_ORIGINS", ""), default=["*"]),
        cors_allow_credentials=_parse_bool(os.getenv("RAG_API_CORS_ALLOW_CREDENTIALS", "true"), default=True),
        cors_allow_methods=_parse_csv_list(os.getenv("RAG_API_CORS_ALLOW_METHODS", ""), default=["*"]),
        cors_allow_headers=_parse_csv_list(os.getenv("RAG_API_CORS_ALLOW_HEADERS", ""), default=["*"]),
        router_clarify_threshold=_parse_float(
            os.getenv("ROUTER_CLARIFY_THRESHOLD", "0.5"), default=0.5, minimum=0.0, maximum=1.0
        ),
    )

