"""Pydantic schemas for RAG API request/response contracts."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str
    k: Optional[int] = 5
    lab_name: Optional[str] = None
    allow_clarify: Optional[bool] = True
    conversation_id: Optional[str] = None
    turn_index: Optional[int] = None


class QueryResponse(BaseModel):
    answer: str
    timescale: str
    cards_retrieved: int
    recent_card: bool
    conversation_id: Optional[str] = None
    turn_index: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    visualization_type: Optional[str] = "none"
    chart: Optional[Dict[str, Any]] = None


class EvidenceSource(BaseModel):
    source_kind: str
    table: Optional[str] = None
    operation: Optional[str] = None
    metric: Optional[str] = None
    source_label: Optional[str] = None
    topic: Optional[str] = None
    title: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)


class ToolEvidence(BaseModel):
    evidence_version: str = "v1"
    evidence_kind: str
    intent: Optional[str] = None
    strategy: Optional[str] = None
    metric_aliases: List[str] = Field(default_factory=list)
    resolved_scope: Optional[str] = None
    resolved_time_window: Optional[Dict[str, Any]] = None
    provenance_sources: List[EvidenceSource] = Field(default_factory=list)
    confidence_notes: List[str] = Field(default_factory=list)
    recommendation_allowed: bool = True


def validate_tool_evidence(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize evidence payload for cross-layer contracts."""
    if hasattr(ToolEvidence, "model_validate"):
        return ToolEvidence.model_validate(raw).model_dump()
    return ToolEvidence.parse_obj(raw).dict()

