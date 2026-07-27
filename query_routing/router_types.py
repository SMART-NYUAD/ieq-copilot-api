"""Shared router datatypes."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from query_routing.intent_classifier import IntentType


class RouteExecutor(str, Enum):
    KNOWLEDGE_QA = "knowledge_qa"
    DB_QUERY = "db_query"
    VIEWER_CONTROL = "viewer_control"
    HEATMAP_CONTROL = "heatmap_control"
    DOWNLOAD_DATA = "download_data"
    IFC_QA = "ifc_qa"
    SENSOR_INSPECTION = "sensor_inspection"


@dataclass(frozen=True)
class RoutePlan:
    intent: IntentType
    confidence: float
    lab_name: Optional[str]
    time_phrase: Optional[str]
    model: str = ""
    fallback_used: bool = False
    second_lab_name: Optional[str] = None
    metrics: List[str] = field(default_factory=list)
    viewer_type: Optional[str] = None
    heatmap_action: Optional[str] = None   # "on" | "off" when intent is heatmap_control
    heatmap_metric: Optional[str] = None   # temperature | humidity | voc | pm25, else None
    download_format: Optional[str] = None  # "csv" | "json" when intent is download_data
    download_metric: Optional[str] = None  # canonical metric (temperature|humidity|co2|voc|pm25) — REQUIRED to fulfil a download
    download_interval: Optional[str] = None  # aggregation interval (e.g. "1m", "1h", "1d") when intent is download_data
    analysis_mode: Optional[str] = None    # "diagnostic" when the user asks WHY an index/metric is bad / what is driving it, else None
    resolved_question: Optional[str] = None  # current question rewritten self-contained from prior turns (LLM context resolution); None => use the raw question
    # The router decides when a question cannot be answered without guessing, and supplies the
    # question to ask back. Honored only when the request allows clarification.
    needs_clarification: bool = False
    clarification_question: Optional[str] = None
    # Which family of metrics the question needs (metric_planning.VALID_METRIC_SCOPES).
    # None => the DB path infers it from question text (emergency/keyword path).
    metric_scope: Optional[str] = None
