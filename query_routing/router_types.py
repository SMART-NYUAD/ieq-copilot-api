"""Shared router datatypes."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

try:
    from query_routing.intent_classifier import IntentType
except ImportError:
    from .intent_classifier import IntentType


class RouteExecutor(str, Enum):
    KNOWLEDGE_QA = "knowledge_qa"
    DB_QUERY = "db_query"


@dataclass(frozen=True)
class RoutePlan:
    intent: IntentType
    confidence: float
    lab_name: Optional[str]
    second_lab_name: Optional[str]
    metrics: List[str]
    time_phrase: Optional[str]
    model: str = ""
    fallback_used: bool = False
