"""Shared router datatypes."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple

try:
    from query_routing.intent_classifier import IntentType, RouteDecision
except ImportError:
    from .intent_classifier import IntentType, RouteDecision


class IntentCategory(str, Enum):
    SEMANTIC_EXPLANATORY = "semantic_explanatory"
    STRUCTURED_FACTUAL_DB = "structured_factual_db"
    ANALYTICAL_VISUALIZATION = "analytical_visualization"
    PREDICTION = "prediction"


class AnswerStrategy(str, Enum):
    DIRECT = "direct"
    DECOMPOSE = "decompose"
    CLARIFY = "clarify"


class DecompositionTemplate(str, Enum):
    STATE_RECOMMENDATION = "state_recommendation"
    TREND_INTERPRETATION = "trend_interpretation"
    ANOMALY_EXPLANATION = "anomaly_explanation"


class QueryScopeClass(str, Enum):
    DOMAIN = "domain"
    NON_DOMAIN = "non_domain"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class RoutePlan:
    decision: RouteDecision
    intent_category: IntentCategory
    route_source: str
    planner_model: str
    planner_fallback_used: bool
    planner_fallback_reason: Optional[str] = None
    planner_raw: Optional[Dict[str, Any]] = None
    planner_parameters: Dict[str, Any] = field(default_factory=dict)
    answer_strategy: AnswerStrategy = AnswerStrategy.DIRECT
    secondary_intents: Tuple[IntentType, ...] = field(default_factory=tuple)
    decomposition_template: Optional[DecompositionTemplate] = None
