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
