"""Environment-backed router settings helpers."""

from __future__ import annotations

import os


def router_mode() -> str:
    return (os.getenv("ROUTER_MODE", "llm") or "llm").strip().lower()


def router_base_url() -> str:
    return (
        os.getenv("OLLAMA_ROUTER_BASE_URL")
        or os.getenv("OLLAMA_BASE_URL")
        or "http://127.0.0.1:11434"
    ).rstrip("/")


def router_model() -> str:
    return (
        os.getenv("OLLAMA_ROUTER_MODEL", "qwen3:30b-a3b-instruct-2507-q4_K_M")
        or "qwen3:30b-a3b-instruct-2507-q4_K_M"
    ).strip()


def router_temperature() -> float:
    raw = (os.getenv("OLLAMA_ROUTER_TEMPERATURE", "0.0") or "0.0").strip()
    try:
        return float(raw)
    except ValueError:
        return 0.0


def router_timeout_seconds() -> float:
    raw = (os.getenv("OLLAMA_ROUTER_TIMEOUT_SECONDS", "20") or "20").strip()
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 20.0


def router_thinking_enabled() -> bool:
    raw = (os.getenv("OLLAMA_ROUTER_THINKING", "false") or "false").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def router_max_retries() -> int:
    raw = (os.getenv("OLLAMA_ROUTER_MAX_RETRIES", "2") or "2").strip()
    try:
        return max(1, min(5, int(raw)))
    except ValueError:
        return 2


def router_retry_jitter_ms() -> int:
    raw = (os.getenv("OLLAMA_ROUTER_RETRY_JITTER_MS", "180") or "180").strip()
    try:
        return max(0, min(2000, int(raw)))
    except ValueError:
        return 180


def legacy_fallback_enabled() -> bool:
    raw = (os.getenv("OLLAMA_ROUTER_LEGACY_FALLBACK", "false") or "false").strip().lower()
    return raw in {"1", "true", "yes", "on"}
