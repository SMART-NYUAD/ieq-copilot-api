"""Rule-based intent classifier built on shared routing signals."""

from dataclasses import dataclass
from enum import Enum
import re
from typing import Dict, Tuple


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
    ranked_intents: Tuple[Tuple[IntentType, float], ...] = tuple()


_TREND_HINTS = ("trend", "over time", "history", "historical", "pattern", "evolution")
_GRAPH_HINTS = ("graph", "chart", "plot", "visualize", "visualise", "line chart", "bar chart")
_ANOMALY_HINTS = (
    "anomaly",
    "anomalies",
    "abnormal",
    "abnormality",
    "abnormalities",
    "outlier",
    "outliers",
    "spike",
    "spikes",
    "deviation",
    "deviations",
)
_COMPARISON_HINTS = ("compare", "comparison", "versus", " vs ", "higher than", "lower than")
_AGGREGATION_HINTS = ("average", "avg", "mean", "median", "sum", "count", "maximum", "minimum", "max", "min")
_POINT_LOOKUP_HINTS = ("what is", "current", "now", "latest", "right now", "at this moment")
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

_ANOMALY_RE = re.compile(r"\b(anomal(y|ies)|anormalit(y|ies)|anormal)\b")
_DUAL_SPACE_TOKEN_RE = re.compile(
    r"\b([a-z0-9]+(?:_[a-z0-9]+)+)\b.*\b(?:and|vs|versus|with|against)\b.*\b([a-z0-9]+(?:_[a-z0-9]+)+)\b"
)


def _rank_intent_candidates(
    *,
    has_anomaly: bool,
    has_trend: bool,
    has_graph: bool,
    has_air_quality_phrase: bool,
    has_comparison: bool,
    has_aggregation: bool,
    has_forecast: bool,
    has_metric: bool,
    has_time_window: bool,
    has_lab_reference: bool,
    asks_for_values: bool,
    has_dual_space_tokens: bool,
    has_comfort: bool,
    has_definition_phrase: bool,
    asks_for_db_facts: bool,
    metric_strength: float,
    scope_strength: float,
    diagnostic_strength: float,
) -> Tuple[Tuple[IntentType, float], ...]:
    scores: Dict[IntentType, float] = {intent: 0.0 for intent in IntentType}

    if has_definition_phrase:
        scores[IntentType.DEFINITION_EXPLANATION] += 0.6
    if has_air_quality_phrase and not has_time_window and not has_lab_reference:
        scores[IntentType.DEFINITION_EXPLANATION] += 0.2
    if has_forecast:
        scores[IntentType.FORECAST_DB] += 0.9
    if has_comparison or has_dual_space_tokens:
        scores[IntentType.COMPARISON_DB] += 0.85
    if has_anomaly:
        scores[IntentType.ANOMALY_ANALYSIS_DB] += 0.7
    if has_aggregation or has_trend or has_graph or asks_for_values:
        scores[IntentType.AGGREGATION_DB] += 0.65
    if has_metric and (has_time_window or asks_for_values):
        scores[IntentType.AGGREGATION_DB] += 0.25
    if has_metric and (has_lab_reference or has_time_window):
        scores[IntentType.CURRENT_STATUS_DB] += 0.45
    if has_air_quality_phrase and not has_time_window:
        scores[IntentType.CURRENT_STATUS_DB] += 0.2
    if has_comfort and not has_time_window:
        scores[IntentType.CURRENT_STATUS_DB] += 0.3

    db_boost = 0.2 * scope_strength + 0.15 * metric_strength
    for intent in (
        IntentType.CURRENT_STATUS_DB,
        IntentType.POINT_LOOKUP_DB,
        IntentType.AGGREGATION_DB,
        IntentType.COMPARISON_DB,
        IntentType.ANOMALY_ANALYSIS_DB,
        IntentType.FORECAST_DB,
    ):
        scores[intent] += db_boost
    scores[IntentType.ANOMALY_ANALYSIS_DB] += 0.2 * diagnostic_strength
    if asks_for_db_facts:
        scores[IntentType.AGGREGATION_DB] += 0.2
        scores[IntentType.CURRENT_STATUS_DB] += 0.2
    if not asks_for_db_facts and scope_strength < 0.2:
        scores[IntentType.DEFINITION_EXPLANATION] += 0.25
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_ranked = [(intent, round(score, 4)) for intent, score in ranked[:4] if score > 0.0]
    return tuple(top_ranked)


def _decision(
    intent: IntentType,
    confidence: float,
    reason: str,
    ranked: Tuple[Tuple[IntentType, float], ...],
) -> RouteDecision:
    ordered: list[Tuple[IntentType, float]] = []
    ordered.append((intent, round(max(confidence, 0.0), 4)))
    for candidate_intent, candidate_score in ranked:
        if candidate_intent == intent:
            continue
        ordered.append((candidate_intent, candidate_score))
        if len(ordered) >= 4:
            break
    return RouteDecision(
        intent=intent,
        confidence=confidence,
        reason=reason,
        ranked_intents=tuple(ordered),
    )


def classify_intent(question: str) -> RouteDecision:
    text = f" {question.strip().lower()} "
    if not text.strip():
        return RouteDecision(
            IntentType.UNKNOWN_FALLBACK,
            0.0,
            "empty_question",
            ranked_intents=((IntentType.UNKNOWN_FALLBACK, 0.0),),
        )
    # Lazy import avoids circular dependency with router_types.
    try:
        from query_routing.router_signals import extract_query_signals
    except ImportError:
        from .router_signals import extract_query_signals

    signals = extract_query_signals(question=question, lab_name=None)

    has_anomaly = bool(_ANOMALY_RE.search(text)) or any(hint in text for hint in _ANOMALY_HINTS)
    has_trend = any(hint in text for hint in _TREND_HINTS)
    has_graph = any(hint in text for hint in _GRAPH_HINTS)
    has_air_quality_phrase = bool(signals.get("is_air_assessment_phrase")) or "ieq" in text
    has_comparison = any(hint in text for hint in _COMPARISON_HINTS)
    has_aggregation = any(hint in text for hint in _AGGREGATION_HINTS)
    has_forecast = any(hint in text for hint in _FORECAST_HINTS)
    has_metric = bool(signals.get("has_metric_reference"))
    has_time_window = bool(signals.get("has_time_window_hint"))
    has_lab_reference = bool(signals.get("has_lab_reference"))
    asks_for_values = any(hint in text for hint in _VALUE_LIST_HINTS)
    has_dual_space_tokens = bool(_DUAL_SPACE_TOKEN_RE.search(text))
    has_comfort = bool(signals.get("is_comfort_assessment_phrase")) or any(hint in text for hint in _COMFORT_HINTS)
    has_definition_phrase = any(
        hint in text for hint in ("what does", "meaning of", "definition", "define", "explain")
    )
    asks_for_db_facts = bool(signals.get("asks_for_db_facts"))
    metric_strength = float(signals.get("metric_signal_strength") or 0.0)
    scope_strength = float(signals.get("scope_signal_strength") or 0.0)
    diagnostic_strength = float(signals.get("diagnostic_signal_strength") or 0.0)
    ranked = _rank_intent_candidates(
        has_anomaly=has_anomaly,
        has_trend=has_trend,
        has_graph=has_graph,
        has_air_quality_phrase=has_air_quality_phrase,
        has_comparison=has_comparison,
        has_aggregation=has_aggregation,
        has_forecast=has_forecast,
        has_metric=has_metric,
        has_time_window=has_time_window,
        has_lab_reference=has_lab_reference,
        asks_for_values=asks_for_values,
        has_dual_space_tokens=has_dual_space_tokens,
        has_comfort=has_comfort,
        has_definition_phrase=has_definition_phrase,
        asks_for_db_facts=asks_for_db_facts,
        metric_strength=metric_strength,
        scope_strength=scope_strength,
        diagnostic_strength=diagnostic_strength,
    )
    anomaly_scope_evidence = (
        has_time_window
        or has_metric
        or has_lab_reference
        or has_comparison
        or has_aggregation
        or asks_for_db_facts
    )

    if has_anomaly:
        if anomaly_scope_evidence:
            return _decision(IntentType.ANOMALY_ANALYSIS_DB, 0.84, "anomaly_keyword_scoped", ranked)
        if has_definition_phrase:
            return _decision(IntentType.DEFINITION_EXPLANATION, 0.74, "anomaly_conceptual_phrase", ranked)
        return _decision(IntentType.AGGREGATION_DB, 0.68, "anomaly_keyword_unscoped", ranked)

    if has_comparison:
        return _decision(IntentType.COMPARISON_DB, 0.9, "comparison_keyword", ranked)

    if has_dual_space_tokens and (has_air_quality_phrase or has_metric or has_time_window):
        return _decision(IntentType.COMPARISON_DB, 0.88, "dual_space_tokens", ranked)

    if has_forecast:
        return _decision(IntentType.FORECAST_DB, 0.94, "forecast_keyword", ranked)

    if has_air_quality_phrase and has_time_window:
        return _decision(IntentType.AGGREGATION_DB, 0.84, "air_quality_time_window_db", ranked)

    if has_air_quality_phrase and any(hint in text for hint in _POINT_LOOKUP_HINTS):
        return _decision(IntentType.CURRENT_STATUS_DB, 0.85, "air_quality_point_lookup_db", ranked)

    # Metric + explicit time window should prefer DB over semantic card retrieval.
    if has_metric and (has_time_window or asks_for_values or has_graph):
        return _decision(IntentType.AGGREGATION_DB, 0.88, "metric_time_window_db", ranked)
    if has_comfort and asks_for_db_facts and not has_time_window:
        return _decision(IntentType.CURRENT_STATUS_DB, 0.82, "comfort_scoped_current_status", ranked)
    if has_comfort and has_time_window:
        return _decision(IntentType.AGGREGATION_DB, 0.78, "comfort_time_window", ranked)
    if has_definition_phrase and not (has_time_window or has_comparison or has_forecast):
        return _decision(
            IntentType.DEFINITION_EXPLANATION,
            0.86,
            "definition_explanation_phrase",
            ranked,
        )

    # If a query mixes trend words with aggregation words (e.g. "average ... over
    # last 24 hours"), prefer DB aggregation to avoid losing numeric precision.
    if has_aggregation:
        if has_trend:
            return _decision(IntentType.AGGREGATION_DB, 0.86, "aggregation_over_trend", ranked)
        return _decision(IntentType.AGGREGATION_DB, 0.82, "aggregation_keyword", ranked)

    if has_trend:
        return _decision(IntentType.AGGREGATION_DB, 0.9, "trend_keyword", ranked)

    if any(hint in text for hint in _POINT_LOOKUP_HINTS) and (has_metric or has_air_quality_phrase):
        return _decision(IntentType.CURRENT_STATUS_DB, 0.8, "point_lookup_phrase_with_metric", ranked)

    if has_metric and has_lab_reference and re.search(r"\bhow\s+(?:is|was)\b", text):
        return _decision(IntentType.CURRENT_STATUS_DB, 0.82, "how_is_metric_in_lab", ranked)

    if has_metric and re.search(r"\b(now|latest|current)\b", text):
        return _decision(IntentType.CURRENT_STATUS_DB, 0.72, "metric_with_recentness_word", ranked)
    if asks_for_db_facts and not has_definition_phrase:
        # Generic measured-scope fallback defaults to aggregation.
        return _decision(IntentType.AGGREGATION_DB, 0.7, "db_scope_signal", ranked)
    if any(hint in text for hint in ("what is", "what does", "meaning of", "explain", "definition")):
        return _decision(
            IntentType.DEFINITION_EXPLANATION,
            0.8,
            "definition_explanation_keyword",
            ranked,
        )
    return _decision(IntentType.DEFINITION_EXPLANATION, 0.65, "fallback_definition", ranked)

