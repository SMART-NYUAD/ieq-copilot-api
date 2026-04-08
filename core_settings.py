"""Centralized runtime settings for the RAG API server.

This module intentionally keeps a small, stable configuration surface for the
first refactor phase. Additional settings can be added incrementally without
changing call sites by extending ``AppSettings``.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys
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
    is_unittest = "unittest" in sys.modules
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
            # Keep unit tests deterministic: strict-routing should be controlled
            # by test code/env patches, not by developer-local .env defaults.
            if is_unittest and key == "AGENT_ROUTING_STRICT":
                continue
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
    agentic_mode: bool
    agent_max_steps: int
    agent_max_consecutive_failures: int
    agent_stall_threshold: int
    agent_stream_step_events: bool
    agent_routing_strict: bool


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
        agentic_mode=_parse_bool(os.getenv("AGENTIC_MODE", "true"), default=True),
        agent_max_steps=_parse_int(os.getenv("AGENT_MAX_STEPS", "4"), default=4, minimum=1),
        agent_max_consecutive_failures=_parse_int(
            os.getenv("AGENT_MAX_CONSECUTIVE_FAILURES", "2"), default=2, minimum=1
        ),
        agent_stall_threshold=_parse_int(os.getenv("AGENT_STALL_THRESHOLD", "2"), default=2, minimum=1),
        agent_stream_step_events=_parse_bool(
            os.getenv("AGENT_STREAM_STEP_EVENTS", "true"),
            default=True,
        ),
        agent_routing_strict=_parse_bool(
            os.getenv("AGENT_ROUTING_STRICT", "false"),
            default=False,
        ),
    )


def router_base_url() -> str:
    ensure_env_loaded()
    return (
        os.getenv("OLLAMA_ROUTER_BASE_URL")
        or os.getenv("OLLAMA_BASE_URL")
        or "http://127.0.0.1:11434"
    ).rstrip("/")


def router_model() -> str:
    ensure_env_loaded()
    return (
        os.getenv("OLLAMA_ROUTER_MODEL", "qwen3:30b-a3b-instruct-2507-q4_K_M")
        or "qwen3:30b-a3b-instruct-2507-q4_K_M"
    ).strip()


def router_temperature() -> float:
    ensure_env_loaded()
    raw = (os.getenv("OLLAMA_ROUTER_TEMPERATURE", "0.0") or "0.0").strip()
    try:
        return float(raw)
    except ValueError:
        return 0.0


def router_timeout_seconds() -> float:
    ensure_env_loaded()
    raw = (os.getenv("OLLAMA_ROUTER_TIMEOUT_SECONDS", "20") or "20").strip()
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 20.0


def router_thinking_enabled() -> bool:
    ensure_env_loaded()
    return _parse_bool(os.getenv("OLLAMA_ROUTER_THINKING", "false"), default=False)


def router_max_retries() -> int:
    ensure_env_loaded()
    raw = (os.getenv("OLLAMA_ROUTER_MAX_RETRIES", "2") or "2").strip()
    try:
        return max(1, min(5, int(raw)))
    except ValueError:
        return 2


def router_retry_jitter_ms() -> int:
    ensure_env_loaded()
    raw = (os.getenv("OLLAMA_ROUTER_RETRY_JITTER_MS", "180") or "180").strip()
    try:
        return max(0, min(2000, int(raw)))
    except ValueError:
        return 180


def router_semantic_rewrite_enabled() -> bool:
    ensure_env_loaded()
    return _parse_bool(os.getenv("ROUTER_SEMANTIC_REWRITE_ENABLED", "false"), default=False)


def router_semantic_rewrite_timeout_seconds() -> float:
    ensure_env_loaded()
    raw = (os.getenv("ROUTER_SEMANTIC_REWRITE_TIMEOUT_SECONDS", "4") or "4").strip()
    try:
        return max(0.5, min(10.0, float(raw)))
    except ValueError:
        return 4.0


def agent_routing_strict_enabled() -> bool:
    ensure_env_loaded()
    return _parse_bool(os.getenv("AGENT_ROUTING_STRICT", "false"), default=False)

