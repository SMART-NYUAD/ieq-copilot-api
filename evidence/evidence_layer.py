"""Evidence Layer: normalize, validate, and repair executor provenance.

The goal of this module is to centralize evidence shaping so executors can
focus on data retrieval/generation and mappers can rely on a single envelope.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

try:
    from http_schemas import validate_tool_evidence
except ImportError:
    from ..http_schemas import validate_tool_evidence


def build_repaired_evidence(executor: str, lab_name: Optional[str], reason: str = "repaired_evidence") -> Dict[str, Any]:
    """Create a deterministic fallback evidence envelope for error cases."""
    return {
        "evidence_version": "v1",
        "evidence_kind": executor,
        "intent": None,
        "strategy": "direct",
        "metric_aliases": [],
        "resolved_scope": lab_name,
        "resolved_time_window": None,
        "provenance_sources": [],
        "confidence_notes": ["clarification_required" if executor == "clarify_gate" else reason],
        "recommendation_allowed": executor != "clarify_gate",
        "repair_reason": reason,
    }


def normalize_evidence(
    raw: Optional[Dict[str, Any]],
    *,
    executor: str,
    lab_name: Optional[str],
    fallback_reason: str = "repaired_evidence",
) -> Dict[str, Any]:
    """Return validated evidence; repair automatically if payload is invalid."""
    if isinstance(raw, dict):
        try:
            return validate_tool_evidence(raw)
        except Exception:
            # Keep behavior resilient and deterministic even on bad upstream shape.
            pass
    return build_repaired_evidence(executor=executor, lab_name=lab_name, reason=fallback_reason)

