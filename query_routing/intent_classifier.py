"""Rule-based intent classifier for selecting card RAG vs DB query paths."""

from dataclasses import dataclass
from enum import Enum
import re


class IntentType(str, Enum):
    DEFINITION_EXPLANATION = "definition_explanation"
    CURRENT_STATUS_DB = "current_status_db"
    POINT_LOOKUP_DB = "point_lookup_db"
    AGGREGATION_DB = "aggregation_db"
    COMPARISON_DB = "comparison_db"
    ANOMALY_ANALYSIS_DB = "anomaly_analysis_db"
    FORECAST_DB = "forecast_db"
    UNKNOWN_FALLBACK = "unknown_fallback"
    # Legacy aliases retained for backward-compatible references.
    SEMANTIC_CARD = DEFINITION_EXPLANATION
    TREND_CARD = AGGREGATION_DB
    ANOMALY_CARD = ANOMALY_ANALYSIS_DB


@dataclass(frozen=True)
class RouteDecision:
    intent: IntentType
    confidence: float
    reason: str


_TREND_HINTS = (
    "trend",
    "over time",
    "evolution",
    "history",
    "historical",
    "pattern",
    "last days",
    "last hours",
    "over the last",
)
_GRAPH_HINTS = (
    "graph",
    "chart",
    "plot",
    "visualize",
    "visualise",
    "line chart",
    "bar chart",
)
_AIR_QUALITY_HINTS = (
    "air quality",
    "ieq",
    "indoor air quality",
)

_ANOMALY_HINTS = (
    "anomaly",
    "anomalies",
    "anormality",
    "anormalities",
    "anormal",
    "abnormal",
    "abnormality",
    "abnormalities",
    "outlier",
    "outliers",
    "spike",
    "spikes",
    "unusual",
    "unexpected",
    "deviation",
    "deviations",
)

_COMPARISON_HINTS = (
    "compare",
    "comparison",
    "versus",
    " vs ",
    "higher than",
    "lower than",
    "better than",
    "worse than",
)

_AGGREGATION_HINTS = (
    "average",
    "avg",
    "mean",
    "median",
    "sum",
    "count",
    "maximum",
    "minimum",
    "max",
    "min",
)

_POINT_LOOKUP_HINTS = (
    "what is",
    "current",
    "now",
    "latest",
    "right now",
    "at this moment",
)
_COMFORT_HINTS = (
    "comfortable",
    "comfort",
    "too hot",
    "too cold",
    "stuffy",
    "fresh air",
    "dry",
    "humid",
)
_VALUE_LIST_HINTS = (
    "values",
    "value",
    "readings",
    "data points",
    "per hour",
    "hourly",
)
_FORECAST_HINTS = (
    "forecast",
    "predict",
    "prediction",
    "project",
    "projection",
    "next hour",
    "next hours",
    "next day",
    "next days",
    "next week",
    "next month",
    "tomorrow",
)

_METRIC_HINT_RE = re.compile(
    r"\b(pm\s*2\.?\s*5|pm2\.?5|pm25|co2|tvoc|voc|temperature|temp|humidity|light|lux|sound|noise|ieq)\b"
)
_ANOMALY_RE = re.compile(r"\b(anomal(y|ies)|anormalit(y|ies)|anormal)\b")
_TIME_WINDOW_RE = re.compile(
    r"\b("
    r"this week|last week|this month|last month|today|yesterday|"
    r"last\s+\d+\s+(hour|hours|day|days|week|weeks|month|months)|"
    r"past\s+\d+\s+(hour|hours|day|days|week|weeks|month|months)|"
    r"january|february|march|april|may|june|july|august|september|october|november|december|"
    r"jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec|"
    r"\d{4}-\d{1,2}-\d{1,2}|"
    r"\d{1,2}(st|nd|rd|th)?(\s+of)?\s+"
    r"(january|february|march|april|may|june|july|august|september|october|november|december)|"
    r"(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(st|nd|rd|th)?"
    r")\b"
)
_DUAL_SPACE_TOKEN_RE = re.compile(
    r"\b([a-z0-9]+(?:_[a-z0-9]+)+)\b.*\b(?:and|vs|versus|with|against)\b.*\b([a-z0-9]+(?:_[a-z0-9]+)+)\b"
)
_LAB_REFERENCE_RE = re.compile(r"\b([a-z0-9]+_lab|[a-z0-9]+\s+lab)\b")


def classify_intent(question: str) -> RouteDecision:
    text = f" {question.strip().lower()} "
    if not text.strip():
        return RouteDecision(IntentType.UNKNOWN_FALLBACK, 0.0, "empty_question")

    has_anomaly = bool(_ANOMALY_RE.search(text)) or any(hint in text for hint in _ANOMALY_HINTS)
    has_trend = any(hint in text for hint in _TREND_HINTS)
    has_graph = any(hint in text for hint in _GRAPH_HINTS)
    has_air_quality_phrase = any(hint in text for hint in _AIR_QUALITY_HINTS)
    has_comparison = any(hint in text for hint in _COMPARISON_HINTS)
    has_aggregation = any(hint in text for hint in _AGGREGATION_HINTS)
    has_forecast = any(hint in text for hint in _FORECAST_HINTS)
    has_metric = bool(_METRIC_HINT_RE.search(text))
    has_time_window = bool(_TIME_WINDOW_RE.search(text))
    has_lab_reference = bool(_LAB_REFERENCE_RE.search(text))
    asks_for_values = any(hint in text for hint in _VALUE_LIST_HINTS)
    has_dual_space_tokens = bool(_DUAL_SPACE_TOKEN_RE.search(text))
    has_comfort = any(hint in text for hint in _COMFORT_HINTS)
    has_definition_phrase = any(
        hint in text for hint in ("what does", "meaning of", "definition", "define", "explain")
    )

    if has_anomaly:
        return RouteDecision(IntentType.ANOMALY_ANALYSIS_DB, 0.95, "anomaly_keyword")

    if has_comparison:
        return RouteDecision(IntentType.COMPARISON_DB, 0.9, "comparison_keyword")

    if has_dual_space_tokens and (has_air_quality_phrase or has_metric or has_time_window):
        return RouteDecision(IntentType.COMPARISON_DB, 0.88, "dual_space_tokens")

    if has_forecast:
        return RouteDecision(IntentType.FORECAST_DB, 0.94, "forecast_keyword")

    if has_air_quality_phrase and has_time_window:
        return RouteDecision(IntentType.AGGREGATION_DB, 0.84, "air_quality_time_window_db")

    if has_air_quality_phrase and any(hint in text for hint in _POINT_LOOKUP_HINTS):
        return RouteDecision(IntentType.CURRENT_STATUS_DB, 0.85, "air_quality_point_lookup_db")

    # Metric + explicit time window should prefer DB over semantic card retrieval.
    if has_metric and (has_time_window or asks_for_values or has_graph):
        return RouteDecision(IntentType.AGGREGATION_DB, 0.88, "metric_time_window_db")
    if has_definition_phrase and not (has_time_window or has_comparison or has_forecast):
        return RouteDecision(
            IntentType.DEFINITION_EXPLANATION, 0.86, "definition_explanation_phrase"
        )

    # If a query mixes trend words with aggregation words (e.g. "average ... over
    # last 24 hours"), prefer DB aggregation to avoid losing numeric precision.
    if has_aggregation:
        if has_trend:
            return RouteDecision(IntentType.AGGREGATION_DB, 0.86, "aggregation_over_trend")
        return RouteDecision(IntentType.AGGREGATION_DB, 0.82, "aggregation_keyword")

    if has_trend:
        return RouteDecision(IntentType.AGGREGATION_DB, 0.9, "trend_keyword")

    if any(hint in text for hint in _POINT_LOOKUP_HINTS) and (has_metric or has_air_quality_phrase):
        return RouteDecision(IntentType.CURRENT_STATUS_DB, 0.8, "point_lookup_phrase_with_metric")

    if has_metric and has_lab_reference and re.search(r"\bhow\s+(?:is|was)\b", text):
        return RouteDecision(IntentType.CURRENT_STATUS_DB, 0.82, "how_is_metric_in_lab")

    if has_metric and re.search(r"\b(now|latest|current)\b", text):
        return RouteDecision(IntentType.CURRENT_STATUS_DB, 0.72, "metric_with_recentness_word")
    if has_comfort and any(hint in text for hint in _POINT_LOOKUP_HINTS):
        return RouteDecision(IntentType.CURRENT_STATUS_DB, 0.79, "comfort_current_status")
    if has_comfort and has_time_window:
        return RouteDecision(IntentType.AGGREGATION_DB, 0.78, "comfort_time_window")
    if any(hint in text for hint in ("what is", "what does", "meaning of", "explain", "definition")):
        return RouteDecision(
            IntentType.DEFINITION_EXPLANATION, 0.8, "definition_explanation_keyword"
        )
    return RouteDecision(IntentType.DEFINITION_EXPLANATION, 0.65, "fallback_definition")

