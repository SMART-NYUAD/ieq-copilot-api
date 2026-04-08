"""Compatibility shim: router settings now live in `core_settings`."""

from __future__ import annotations

try:
    from core_settings import (
        router_base_url,
        router_max_retries,
        router_model,
        router_retry_jitter_ms,
        router_temperature,
        router_thinking_enabled,
        router_timeout_seconds,
    )
except ImportError:
    from ..core_settings import (
        router_base_url,
        router_max_retries,
        router_model,
        router_retry_jitter_ms,
        router_temperature,
        router_thinking_enabled,
        router_timeout_seconds,
    )

__all__ = [
    "router_base_url",
    "router_model",
    "router_temperature",
    "router_timeout_seconds",
    "router_thinking_enabled",
    "router_max_retries",
    "router_retry_jitter_ms",
]
