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
    footnotes: List["GuidelineFootnote"] = Field(default_factory=list)
    citation_sources: List["CitationSource"] = Field(default_factory=list)


class _CitationBase(BaseModel):
    index: int
    source_key: Optional[str] = None
    source_label: str
    section_ref: Optional[str] = None
    citation_tier: str
    source_url: Optional[str] = None
    threshold_value: Optional[float] = None
    threshold_unit: Optional[str] = None
    caveat_text: Optional[str] = None


class GuidelineFootnote(_CitationBase):
    pass


class CitationSource(_CitationBase):
    pass
