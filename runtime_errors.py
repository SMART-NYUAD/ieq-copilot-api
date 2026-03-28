"""Lightweight runtime error taxonomy and formatting helpers.

The taxonomy is intentionally small and pragmatic for incremental adoption.
It gives routes a consistent shape for logs and streamed error payloads.
"""

from __future__ import annotations

from enum import Enum
import logging
from typing import Any, Dict


LOGGER = logging.getLogger("rag_api_server")


class ErrorCode(str, Enum):
    """Stable error codes used in logs and streaming payloads."""

    INVALID_INPUT = "invalid_input"
    ROUTING_ERROR = "routing_error"
    EXECUTION_ERROR = "execution_error"
    STREAM_ERROR = "stream_error"
    INTERNAL_ERROR = "internal_error"


def classify_error(exc: Exception) -> ErrorCode:
    """Map arbitrary exceptions into a small set of stable codes."""
    text = str(exc or "").lower()
    name = type(exc).__name__.lower()
    if "validation" in text or "invalid" in text:
        return ErrorCode.INVALID_INPUT
    if "route" in text or "planner" in text:
        return ErrorCode.ROUTING_ERROR
    if "db" in text or "cursor" in text or "sql" in text or "executor" in text:
        return ErrorCode.EXECUTION_ERROR
    if "stream" in text:
        return ErrorCode.STREAM_ERROR
    if "http" in name or "connection" in text or "timeout" in text:
        return ErrorCode.EXECUTION_ERROR
    return ErrorCode.INTERNAL_ERROR


def log_exception(exc: Exception, *, scope: str, extra: Dict[str, Any] | None = None) -> ErrorCode:
    """Log exception with structured scope context and return classified code."""
    code = classify_error(exc)
    LOGGER.exception(
        "runtime_error scope=%s code=%s extra=%s",
        scope,
        code.value,
        extra or {},
    )
    return code


def stream_error_payload(exc: Exception, *, scope: str, extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Build a consistent payload for SSE/OpenAI stream errors."""
    code = log_exception(exc, scope=scope, extra=extra)
    return {
        "detail": str(exc),
        "code": code.value,
        "scope": scope,
    }

