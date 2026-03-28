"""Evidence Layer: normalize, validate, and repair executor provenance.

The goal of this module is to centralize evidence shaping so executors can
focus on data retrieval/generation and mappers can rely on a single envelope.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

try:
    from contracts.progressive_contracts import build_progressive_evidence_envelope
    from http_schemas import validate_tool_evidence
except ImportError:
    from ..contracts.progressive_contracts import build_progressive_evidence_envelope
    from ..http_schemas import validate_tool_evidence


def build_repaired_evidence(executor: str, lab_name: Optional[str], reason: str = "repaired_evidence") -> Dict[str, Any]:
    """Create a deterministic fallback evidence envelope for error cases."""
    recommendation_allowed = executor != "clarify_gate"
    confidence_note = "clarification_required" if executor == "clarify_gate" else reason
    progressive = build_progressive_evidence_envelope(
        evidence_kind=executor,
        intent=None,
        strategy="direct",
        metric_aliases=[],
        resolved_scope=lab_name,
        resolved_time_window=None,
        provenance_sources=[],
        confidence_notes=[confidence_note],
        recommendation_allowed=recommendation_allowed,
        extras={"repair_reason": reason},
    )
    return validate_tool_evidence(dict(progressive))


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

