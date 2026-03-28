"""Progressive contracts for cross-layer data exchange.

These contracts intentionally define a stable core plus optional extension
fields. The pattern keeps development flexible while reducing accidental
shape drift between routing, execution, evidence, and response mapping.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


CONTRACT_VERSION_V1 = "v1"


class EvidenceSourceCore(TypedDict, total=False):
    """Stable source metadata expected by evidence consumers."""

    source_kind: str
    table: str
    operation: str
    metric: str
    source_label: str
    topic: str
    title: str
    details: Dict[str, Any]


class EvidenceEnvelopeCore(TypedDict, total=False):
    """Stable evidence envelope passed from evidence layer to mappers."""

    contract_version: str
    evidence_kind: str
    intent: Optional[str]
    strategy: Optional[str]
    metric_aliases: List[str]
    resolved_scope: Optional[str]
    resolved_time_window: Optional[Dict[str, Any]]
    provenance_sources: List[EvidenceSourceCore]
    confidence_notes: List[str]
    recommendation_allowed: bool
    extras: Dict[str, Any]


def build_progressive_evidence_envelope(
    *,
    evidence_kind: str,
    intent: Optional[str],
    strategy: str,
    metric_aliases: Optional[List[str]] = None,
    resolved_scope: Optional[str] = None,
    resolved_time_window: Optional[Dict[str, Any]] = None,
    provenance_sources: Optional[List[EvidenceSourceCore]] = None,
    confidence_notes: Optional[List[str]] = None,
    recommendation_allowed: bool = True,
    extras: Optional[Dict[str, Any]] = None,
) -> EvidenceEnvelopeCore:
    """Construct a stable evidence envelope with optional extension fields."""
    return EvidenceEnvelopeCore(
        contract_version=CONTRACT_VERSION_V1,
        evidence_kind=str(evidence_kind or "").strip(),
        intent=intent,
        strategy=str(strategy or "direct").strip() or "direct",
        metric_aliases=list(metric_aliases or []),
        resolved_scope=resolved_scope,
        resolved_time_window=resolved_time_window,
        provenance_sources=list(provenance_sources or []),
        confidence_notes=list(confidence_notes or []),
        recommendation_allowed=bool(recommendation_allowed),
        extras=dict(extras or {}),
    )

