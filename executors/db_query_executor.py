"""Executor for point lookup, aggregation, and comparison DB queries."""

from datetime import datetime, timedelta, timezone
import calendar
import json
import re
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple
from langchain_core.output_parsers import StrOutputParser
import pandas as pd

try:
    from prophet import Prophet
except ImportError:  # pragma: no cover - dependency presence varies by deployment.
    Prophet = None

try:
    from query_routing.intent_classifier import IntentType
    from storage.postgres_client import get_cursor
except ImportError:
    from ..query_routing.intent_classifier import IntentType
    from ..storage.postgres_client import get_cursor

try:
    from executors.env_query_langchain import get_llm_client, search_knowledge_cards
except ImportError:
    from ..executors.env_query_langchain import get_llm_client, search_knowledge_cards
try:
    from executors.db_support import charts as db_charts
    from executors.db_support import query_handlers as db_handlers
    from executors.db_support import query_parsing as db_parsing
    from executors.db_support import response_helpers as db_helpers
except ImportError:
    from .db_support import charts as db_charts
    from .db_support import query_handlers as db_handlers
    from .db_support import query_parsing as db_parsing
    from .db_support import response_helpers as db_helpers
try:
    from prompting.shared_prompts import build_grounded_context_sections, get_shared_prompt_template
except ImportError:
    from ..prompting.shared_prompts import build_grounded_context_sections, get_shared_prompt_template

try:
    from http_schemas import validate_tool_evidence
except ImportError:
    from ..http_schemas import validate_tool_evidence


METRIC_UNIT_MAP = {
    "air_contribution": "%",
    "pm25": "ug/m3",
    "pm2.5": "ug/m3",
    "co2": "ppm",
    "tvoc": "ppm",
    "voc": "ppm",
    "temperature": "degC",
    "temp": "degC",
    "humidity": "%",
    "light": "lux",
    "lux": "lux",
    "sound": "dB",
    "noise": "dB",
    "ieq": "index",
    "index": "index",
}
MAX_CHART_LOOKBACK_POINTS = 0  # 0 disables chart-side truncation; preserve requested windows
LLM_PAYLOAD_MAX_RECENT_POINTS = 48
LLM_PAYLOAD_MAX_NON_TIMESERIES_ROWS = 12
_TARGET_TZ = timezone(timedelta(hours=4))


def _to_target_timezone(dt: datetime) -> datetime:
    normalized = dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    return normalized.astimezone(_TARGET_TZ)


def _serialize_datetime_iso(dt: datetime) -> str:
    return _to_target_timezone(dt).isoformat()


def _serialize_timestamp_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return _serialize_datetime_iso(value)
    if isinstance(value, list):
        return [_serialize_timestamp_value(item) for item in value]
    if isinstance(value, dict):
        return {k: _serialize_timestamp_value(v) for k, v in value.items()}
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return value
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return value
        return _serialize_datetime_iso(parsed)
    return value
DB_TOOL_RESPONSE_DIRECTIVE = """
You are answering from a structured DB query result.
- First, answer the exact user question directly before additional detail.
- For air-quality assessment/summary queries, include:
  1) overall status,
  2) metric-by-metric interpretation,
  3) explicit analysis window using the provided time bounds ("from ... to ..."),
  4) stability/trend summary and notable peaks/dips when those stats are available,
  5) missing-metric coverage note (especially TVOC, PM2.5, CO2, humidity),
  6) confidence qualifier tied to metric coverage,
  7) 2-4 practical recommendations.
- For risk-focused questions, lead with the main risk level and concrete risk drivers first.
- When `display_start` and `display_end` are present in measured room facts, copy those values verbatim
    when mentioning the analysis window. Do not rewrite or infer date/month values.
- If a metric was requested but not available, explicitly state it as "not available in this window".
- Keep recommendations actionable and grounded in provided measurements/guidelines.
""".strip()
DB_TOOL_RESPONSE_DIRECTIVE_POINT_LOOKUP = """
You are answering a point lookup from a structured DB query result.
- Lead with the current/latest value requested.
- Give a short plain-language interpretation with unit and citation-style classification if available.
- Keep it concise (one short paragraph + up to 2 bullets).
- If value is missing, say it clearly and suggest the nearest useful fallback window.
""".strip()
DB_TOOL_RESPONSE_DIRECTIVE_AIR_QUALITY_POINT_LOOKUP = """
You are answering a current air-quality point lookup from a structured DB query result.
- First, directly answer the exact question asked.
- Provide an overall current air-quality status in plain language.
- Include concise metric-by-metric interpretation for available core metrics (CO2, PM2.5, TVOC, humidity, and IEQ when present).
- Explain what occupants would likely notice/feel.
- Add 2-4 practical recommendations (maintenance actions are allowed when conditions are good).
- If any core metric is missing, call it out clearly and lower confidence in the overall assessment.
- If the question is risk-focused, start with the risk level and the top risk drivers (or say no major risk is evident).
- Do not collapse the answer to a single sentence.
""".strip()
DB_TOOL_RESPONSE_DIRECTIVE_COMPARISON = """
You are answering a comparison from a structured DB query result.
- Highlight which space is better/worse for each available metric and by how much.
- Call out missing metrics explicitly (especially TVOC for air-quality comparisons).
- End with 2-4 practical actions to improve the weaker metric(s).
""".strip()
DB_TOOL_RESPONSE_DIRECTIVE_FORECAST = """
You are answering a forecast from a structured DB query result.
- Report forecast horizon, trend direction, and confidence in plain language.
- Mention assumptions/limits and avoid deterministic claims beyond provided forecast output.
- Provide 2-3 operational recommendations tied to confidence and trend.
""".strip()
DB_TOOL_RESPONSE_DIRECTIVE_ANOMALY = """
You are answering an anomaly analysis from a structured DB query result.
- State whether anomalies were detected, when they occurred, and likely occupant impact.
- If no anomalies are detected, say so explicitly.
- Provide 2-4 practical troubleshooting/monitoring actions grounded in observed data.
""".strip()


def _is_air_quality_query_text(question: str) -> bool:
    return db_helpers.is_air_quality_query_text(question)


def _is_comfort_assessment_query_text(question: str) -> bool:
    return db_helpers.is_comfort_assessment_query_text(question)


def _db_response_directive(intent: IntentType, question: str = "") -> str:
    return db_helpers.db_response_directive(intent, question=question)


def _build_lab_alias_map() -> Dict[str, str]:
    """Compatibility shim: delegate lab alias map building to db_support parser."""
    return db_parsing.build_lab_alias_map()


def _resolve_lab_alias(raw_lab: Optional[str]) -> Optional[str]:
    """Compatibility shim retained for existing tests/imports."""
    raw = str(raw_lab or "").strip().lower()
    if not raw:
        return None
    token = re.sub(r"[^a-z0-9_\s]", "", raw)
    token = re.sub(r"\s+", " ", token).strip()
    if not token:
        return None

    alias_map = _build_lab_alias_map()
    if not alias_map:
        return token.replace(" ", "_")

    candidates: List[str] = []

    def _push(value: str) -> None:
        normalized = str(value or "").strip().lower()
        if normalized and normalized not in candidates:
            candidates.append(normalized)

    _push(token)
    _push(token.replace(" ", "_"))
    _push(token.replace("_", " "))
    if token.endswith(" lab"):
        base = token[: -len(" lab")].strip()
        _push(base)
        _push(f"{base}_lab")
    if token.endswith("_lab"):
        base = token[: -len("_lab")].strip("_")
        _push(base)
        _push(base.replace("_", " "))
    if " " not in token and "_" not in token:
        _push(f"{token}_lab")

    for candidate in candidates:
        canonical = alias_map.get(candidate)
        if canonical:
            return canonical
    return token.replace(" ", "_")


def _planner_card_controls(planner_hints: Optional[Dict[str, Any]]) -> Tuple[bool, List[str], int]:
    """Compatibility shim: canonical logic lives in db_support.query_parsing."""
    return db_parsing.planner_card_controls(planner_hints)


def _default_window_hours_for_intent(intent: IntentType) -> int:
    """Compatibility shim: canonical logic lives in db_support.query_parsing."""
    return db_parsing.default_window_hours_for_intent(intent)


def _build_point_lookup_answer(metric_alias: str, row: Dict, window_label: str) -> str:
    return db_helpers.build_point_lookup_answer(metric_alias, row, window_label)


def _build_aggregation_answer(metric_alias: str, rows: List[Dict], window_label: str) -> str:
    return db_helpers.build_aggregation_answer(metric_alias, rows, window_label)


def _build_timeseries_answer(metric_alias: str, rows: List[Dict], window_label: str) -> str:
    return db_helpers.build_timeseries_answer(metric_alias, rows, window_label)


def _detect_anomaly_points(
    rows: List[Dict[str, Any]],
    z_threshold: float = 2.5,
) -> List[Dict[str, Any]]:
    return db_helpers.detect_anomaly_points(rows, z_threshold=z_threshold)


def _build_anomaly_answer(
    metric_alias: str,
    rows: List[Dict[str, Any]],
    anomalies: List[Dict[str, Any]],
    window_label: str,
    lab_name: Optional[str],
) -> str:
    return db_helpers.build_anomaly_answer(
        metric_alias=metric_alias,
        rows=rows,
        anomalies=anomalies,
        window_label=window_label,
        lab_name=lab_name,
    )


def _build_comparison_answer(metric_alias: str, rows: List[Dict], window_label: str) -> str:
    return db_helpers.build_comparison_answer(metric_alias, rows, window_label)


def _build_correlation_answer(
    metric_x: str,
    metric_y: str,
    correlation: Optional[float],
    row_count: int,
    window_label: str,
    lab_name: Optional[str],
) -> str:
    return db_helpers.build_correlation_answer(
        metric_x=metric_x,
        metric_y=metric_y,
        correlation=correlation,
        row_count=row_count,
        window_label=window_label,
        lab_name=lab_name,
    )


def _build_forecast_answer(
    metric_alias: str,
    forecast: Optional[Dict[str, Any]],
    window_label: str,
    horizon_label: str,
) -> str:
    return db_helpers.build_forecast_answer(
        metric_alias=metric_alias,
        forecast=forecast,
        window_label=window_label,
        horizon_label=horizon_label,
    )


def _ensure_think_prefix(text: str) -> str:
    return db_helpers.ensure_think_prefix(text)


def _build_db_payload(
    intent: IntentType,
    metric_alias: str,
    window_label: str,
    rows: List[Dict],
    window_start: Optional[str] = None,
    window_end: Optional[str] = None,
    display_start: Optional[str] = None,
    display_end: Optional[str] = None,
    forecast: Optional[Dict[str, Any]] = None,
    knowledge_cards: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    return db_helpers.build_db_payload(
        intent=intent,
        metric_alias=metric_alias,
        window_label=window_label,
        rows=rows,
        window_start=window_start,
        window_end=window_end,
        display_start=display_start,
        display_end=display_end,
        forecast=forecast,
        knowledge_cards=knowledge_cards,
    )


def _metric_unit(metric_alias: str) -> str:
    return db_helpers.metric_unit(metric_alias)


def _wants_time_series(question: str) -> bool:
    return db_helpers.wants_time_series(question)


def _wants_forecast(question: str) -> bool:
    return db_helpers.wants_forecast(question)


def _topic_matches_card(card: Dict[str, Any], requested_topics: List[str]) -> bool:
    return db_helpers.topic_matches_card(card, requested_topics)


def _fetch_knowledge_cards(
    question: str, limit: int = 4, card_topics: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    return db_helpers.fetch_knowledge_cards(
        question=question,
        search_fn=search_knowledge_cards,
        limit=limit,
        card_topics=card_topics,
    )


def _split_knowledge_cards(cards: Optional[List[Dict[str, Any]]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    return db_helpers.split_knowledge_cards(cards)


def _extract_forecast_horizon_hours(question: str) -> Tuple[int, str]:
    return db_helpers.extract_forecast_horizon_hours(question)


def _forecast_history_window(
    question: str,
    horizon_hours: int,
    default_start: datetime,
    default_end: datetime,
    default_label: str,
) -> Tuple[datetime, datetime, str]:
    return db_helpers.forecast_history_window(
        question=question,
        horizon_hours=horizon_hours,
        default_start=default_start,
        default_end=default_end,
        default_label=default_label,
    )


def _build_forecast_from_rows(
    series_rows: List[Dict[str, Any]],
    horizon_hours: int,
) -> Optional[Dict[str, Any]]:
    return db_helpers.build_forecast_from_rows(series_rows, horizon_hours=horizon_hours)


def _build_multi_metric_comparison_answer(
    metric_aliases: List[str], rows: List[Dict[str, Any]], window_label: str
) -> str:
    return db_helpers.build_multi_metric_comparison_answer(metric_aliases, rows, window_label)


def _build_multi_metric_bar_chart(
    metric_aliases: List[str], unit_by_metric: Dict[str, str], window_label: str, rows: List[Dict[str, Any]]
) -> Dict[str, Any]:
    return db_helpers.build_multi_metric_bar_chart(metric_aliases, unit_by_metric, window_label, rows)


def _build_multi_metric_aggregation_answer(
    metric_aliases: List[str], row: Dict[str, Any], window_label: str
) -> str:
    return db_helpers.build_multi_metric_aggregation_answer(metric_aliases, row, window_label)


def _build_multi_metric_snapshot_chart(
    metric_aliases: List[str], unit_by_metric: Dict[str, str], window_label: str, row: Dict[str, Any]
) -> Dict[str, Any]:
    return db_helpers.build_multi_metric_snapshot_chart(metric_aliases, unit_by_metric, window_label, row)


def _build_db_sources(
    *,
    operation_type: str,
    metric_alias: str,
    window_label: str,
    window_start: datetime,
    window_end: datetime,
    resolved_lab_name: Optional[str],
    compared_spaces: List[str],
    rows: List[Dict[str, Any]],
    forecast: Optional[Dict[str, Any]],
    metric_pair: Optional[List[str]] = None,
    correlation: Optional[float] = None,
    metrics_used: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    return db_helpers.build_db_sources(
        operation_type=operation_type,
        metric_alias=metric_alias,
        window_label=window_label,
        window_start=window_start,
        window_end=window_end,
        resolved_lab_name=resolved_lab_name,
        compared_spaces=compared_spaces,
        rows=rows,
        forecast=forecast,
        metric_pair=metric_pair,
        correlation=correlation,
        metrics_used=metrics_used,
    )


def _to_float_or_none(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _pick_primary_value_key(rows: List[Dict[str, Any]]) -> Optional[str]:
    preferred = ("value", "avg_value", "mean_value", "min_value", "max_value")
    for key in preferred:
        for row in rows:
            if _to_float_or_none(row.get(key)) is not None:
                return key
    return None


def _compact_rows_for_llm(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not rows:
        return []
    has_bucket = any(row.get("bucket") is not None for row in rows)
    if has_bucket:
        return rows[-LLM_PAYLOAD_MAX_RECENT_POINTS:]
    return rows[:LLM_PAYLOAD_MAX_NON_TIMESERIES_ROWS]


def _build_rows_summary(rows: List[Dict[str, Any]], compact_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "total_rows": len(rows),
        "rows_in_prompt": len(compact_rows),
        "rows_trimmed": max(0, len(rows) - len(compact_rows)),
    }
    if not rows:
        return summary

    value_key = _pick_primary_value_key(rows)
    if value_key:
        values = [_to_float_or_none(row.get(value_key)) for row in rows]
        numeric_values = [value for value in values if value is not None]
        if numeric_values:
            summary["primary_value_key"] = value_key
            summary["value_stats"] = {
                "min": round(min(numeric_values), 4),
                "max": round(max(numeric_values), 4),
                "mean": round(sum(numeric_values) / len(numeric_values), 4),
            }

    buckets = [row.get("bucket") for row in rows if row.get("bucket") is not None]
    if buckets:
        summary["bucket_range"] = {
            "start": _serialize_timestamp_value(buckets[0]),
            "end": _serialize_timestamp_value(buckets[-1]),
        }

    labs: List[str] = []
    for row in rows:
        token = str(row.get("lab_space") or "").strip()
        if token and token not in labs:
            labs.append(token)
    if labs:
        summary["lab_spaces"] = labs[:5]
    return summary


def _build_compact_llm_rows_and_summary(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    compact = _compact_rows_for_llm(rows)
    summary = _build_rows_summary(rows=rows, compact_rows=compact)
    return compact, summary


def _clarify_text_for_invariant_violation(invariant: Dict[str, Any]) -> str:
    violations = list(invariant.get("violations") or [])
    if "lab_scope_not_justified" in violations:
        return (
            "I can answer this with measured data, but I need the lab first. "
            "Which lab should I use (for example: smart_lab, concrete_lab, or eco_lab)?"
        )
    if "comparison_second_space_not_justified" in violations:
        return (
            "I can run cross-space comparison once both spaces are explicit. "
            "Please name two labs (for example: smart_lab vs concrete_lab)."
        )
    if "metric_not_justified" in violations and "time_window_not_justified" in violations:
        return (
            "I can run this once scope is explicit. "
            "Please provide metric and time window (for example: 'average CO2 in smart_lab last 24 hours')."
        )
    if "metric_not_justified" in violations:
        return (
            "I can run this once the metric is explicit. "
            "Which metric should I use (for example: CO2, PM2.5, TVOC, humidity, temperature, light, or IEQ)?"
        )
    if "time_window_not_justified" in violations:
        return (
            "I can run this once the time window is explicit. "
            "Please specify a window (for example: last hour, last 24 hours, this week, or last week)."
        )
    return (
        "I can run this once scope is clear. Please specify at least one of: "
        "metric, time window, or lab (for example: "
        "'average CO2 in smart_lab last 24 hours')."
    )


def prepare_db_query(
    question: str,
    intent: IntentType,
    lab_name: Optional[str],
    planner_hints: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    query_text = str(question or "").strip()
    metric_alias, metric_column = db_parsing.pick_metric(query_text)
    explicit_metrics = db_parsing.extract_metric_aliases(query_text)
    hinted_metrics = db_parsing.planner_metrics(planner_hints)
    # Guardrail: explicit user metrics should win over planner hints.
    if explicit_metrics:
        explicit_metric = explicit_metrics[0]
        explicit_column = db_parsing.CANONICAL_METRIC_COLUMN_MAP.get(explicit_metric)
        if explicit_column:
            metric_alias, metric_column = explicit_metric, explicit_column
    elif hinted_metrics:
        top_metric = hinted_metrics[0]
        hinted_column = db_parsing.CANONICAL_METRIC_COLUMN_MAP.get(top_metric)
        if hinted_column:
            metric_alias, metric_column = top_metric, hinted_column
    unit = _metric_unit(metric_alias)
    window_start, window_end, window_label = db_parsing.extract_time_window(
        query_text,
        default_hours=db_parsing.default_window_hours_for_intent(intent),
    )
    display_start, display_end = db_parsing.format_display_window_bounds(window_start, window_end)
    resolved_lab_name = db_parsing.resolve_lab_alias(lab_name) or db_parsing.extract_space_from_question(query_text)
    compared_spaces = db_parsing.extract_compared_spaces(query_text)
    invariant = db_parsing.validate_db_execution_invariants(
        question=query_text,
        intent=intent,
        selected_metric=metric_alias,
        resolved_lab_name=resolved_lab_name,
        request_lab_name=lab_name,
        explicit_metrics=explicit_metrics,
        hinted_metrics=hinted_metrics,
        planner_hints=planner_hints,
    )
    if not bool(invariant.get("allowed")):
        return {
            "intent": intent,
            "metric_alias": metric_alias,
            "window_label": window_label,
            "rows": [],
            "payload": [],
            "fallback_answer": _clarify_text_for_invariant_violation(invariant),
            "timescale": "clarify",
            "time_window": {
                "label": window_label,
                "start": window_start.isoformat(),
                "end": window_end.isoformat(),
                "display_start": display_start,
                "display_end": display_end,
            },
            "resolved_lab_name": resolved_lab_name,
            "forecast": None,
            "knowledge_cards": [],
            "cards_retrieved": 0,
            "correlation": None,
            "sources": [],
            "visualization_type": "none",
            "chart": None,
            "invariant_violation": invariant,
        }
    with get_cursor(real_dict=True) as cur:
        branch_result = db_handlers.execute_intent_query(
            cur=cur,
            question=query_text,
            intent=intent,
            metric_alias=metric_alias,
            metric_column=metric_column,
            unit=unit,
            window_start=window_start,
            window_end=window_end,
            window_label=window_label,
            resolved_lab_name=resolved_lab_name,
            compared_spaces=compared_spaces,
            explicit_metrics=explicit_metrics,
            hinted_metrics=hinted_metrics,
            max_chart_lookback_points=MAX_CHART_LOOKBACK_POINTS,
        )

    operation_type = str(branch_result["operation_type"])
    rows = list(branch_result["rows"])
    fallback_answer = str(branch_result["fallback_answer"])
    chart_payload = dict(branch_result["chart_payload"])
    forecast_data = branch_result.get("forecast_data")
    correlation_data = branch_result.get("correlation_data")
    metric_alias = str(branch_result.get("metric_alias") or metric_alias)
    metrics_used = list(branch_result.get("metrics_used") or [metric_alias])
    requested_window_start = window_start
    requested_window_end = window_end
    requested_window_label = window_label
    window_start = branch_result.get("window_start") or window_start
    window_end = branch_result.get("window_end") or window_end
    window_label = str(branch_result.get("window_label") or window_label)
    if operation_type == "prediction":
        # Forecast handler may use a broader internal model window. Keep external
        # metadata/source window aligned to the user-requested scope.
        window_start = requested_window_start
        window_end = requested_window_end
        window_label = requested_window_label
    compared_spaces = list(branch_result.get("compared_spaces") or compared_spaces)

    knowledge_cards: List[Dict[str, Any]] = []

    payload_rows = rows
    row_summary: Dict[str, Any] = {}
    if operation_type == "prediction" and forecast_data:
        # Keep LLM forecast context focused on future points only.
        payload_rows = []
    else:
        payload_rows, row_summary = _build_compact_llm_rows_and_summary(rows)

    needs_cards, card_topics, max_cards = db_parsing.planner_card_controls(planner_hints)
    knowledge_cards = (
        _fetch_knowledge_cards(question=question, limit=max_cards, card_topics=card_topics)
        if needs_cards
        else []
    )

    payload = _build_db_payload(
        intent=intent,
        metric_alias=metric_alias,
        window_label=window_label,
        rows=payload_rows,
        window_start=window_start.isoformat(),
        window_end=window_end.isoformat(),
        display_start=display_start,
        display_end=display_end,
        forecast=forecast_data,
        knowledge_cards=knowledge_cards,
    )
    if isinstance(correlation_data, dict) and "correlations" in correlation_data:
        payload["correlation_analysis"] = _serialize_timestamp_value(correlation_data)
    if operation_type == "prediction" and forecast_data:
        payload["forecast_only_context"] = True
    if row_summary:
        payload["row_summary"] = _serialize_timestamp_value(row_summary)
    metric_pair = None
    correlation_value = None
    if isinstance(correlation_data, dict):
        metric_x = correlation_data.get("metric_x")
        metric_y = correlation_data.get("metric_y")
        if metric_x and metric_y:
            metric_pair = [str(metric_x), str(metric_y)]
            correlation_value = correlation_data.get("correlation")
        elif "top_culprits" in correlation_data:
            top_culprits = list(correlation_data.get("top_culprits") or [])
            top_scores = dict(correlation_data.get("top_culprit_scores") or {})
            if top_culprits:
                top_metric = str(top_culprits[0])
                metric_pair = ["ieq", top_metric]
                correlation_value = top_scores.get(top_metric)
    return {
        "intent": intent,
        "metric_alias": metric_alias,
        "window_label": window_label,
        "rows": rows,
        "payload": payload,
        "fallback_answer": fallback_answer,
        "timescale": "1hour",
        "time_window": {
            "label": window_label,
            "start": window_start.isoformat(),
            "end": window_end.isoformat(),
            "display_start": display_start,
            "display_end": display_end,
        },
        "resolved_lab_name": resolved_lab_name,
        "forecast": forecast_data,
        "knowledge_cards": knowledge_cards,
        "cards_retrieved": len(knowledge_cards),
        "correlation": correlation_data,
        "sources": _build_db_sources(
            operation_type=operation_type,
            metric_alias=metric_alias,
            window_label=window_label,
            window_start=window_start,
            window_end=window_end,
            resolved_lab_name=resolved_lab_name,
            compared_spaces=compared_spaces,
            rows=rows,
            forecast=forecast_data,
            metric_pair=metric_pair,
            correlation=correlation_value,
            metrics_used=metrics_used,
        )
        + [
            {
                "source_kind": "knowledge_card",
                "table": "env_knowledge_cards",
                "card_type": card.get("card_type"),
                "topic": card.get("topic"),
                "title": card.get("title"),
                "source_label": card.get("source_label"),
            }
            for card in knowledge_cards
        ],
        "visualization_type": chart_payload.get("visualization_type", "none"),
        "chart": chart_payload.get("chart"),
    }


def _render_db_answer_with_llm(
    question: str,
    intent: IntentType,
    metric_alias: str,
    window_label: str,
    rows: List[Dict],
    fallback_answer: str,
    time_window: Optional[Dict[str, Any]] = None,
    forecast: Optional[Dict[str, Any]] = None,
    correlation: Optional[Dict[str, Any]] = None,
    knowledge_cards: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[str, bool]:
    llm_rows = rows
    row_summary: Dict[str, Any] = {}
    if intent == IntentType.FORECAST_DB and forecast:
        llm_rows = []
    else:
        llm_rows, row_summary = _build_compact_llm_rows_and_summary(rows)
    payload = _build_db_payload(
        intent,
        metric_alias,
        window_label,
        llm_rows,
        window_start=str((time_window or {}).get("start") or ""),
        window_end=str((time_window or {}).get("end") or ""),
        display_start=str((time_window or {}).get("display_start") or ""),
        display_end=str((time_window or {}).get("display_end") or ""),
        forecast=forecast,
        knowledge_cards=knowledge_cards,
    )
    if intent == IntentType.FORECAST_DB and forecast:
        payload["forecast_only_context"] = True
    if row_summary:
        payload["row_summary"] = _serialize_timestamp_value(row_summary)
    if correlation:
        if isinstance(correlation, dict) and "correlations" in correlation:
            payload["correlation_analysis"] = _serialize_timestamp_value(correlation)
        else:
            payload["correlation"] = {
                "metric_x": correlation.get("metric_x"),
                "metric_y": correlation.get("metric_y"),
                "correlation": correlation.get("correlation"),
                "row_count": correlation.get("row_count"),
            }
    interpretation_cards, guardrails = _split_knowledge_cards(knowledge_cards)
    context_data = build_grounded_context_sections(
        measured_room_facts=payload,
        backend_semantic_state=None,
        knowledge_cards=interpretation_cards,
        communication_guardrails=guardrails,
    )
    prompt_template = get_shared_prompt_template(
        response_directive=_db_response_directive(intent, question=question)
    )
    try:
        qa_chain = prompt_template | get_llm_client() | StrOutputParser()
        text = qa_chain.invoke(
            {
                "question": question,
                "context_label": "Structured DB Query Result with knowledge grounding",
                "context_data": context_data,
            }
        )
        text = _ensure_think_prefix(text)
        if text:
            return text, True
    except Exception:
        pass
    return _ensure_think_prefix(fallback_answer), False


def run_db_query(
    question: str,
    intent: IntentType,
    lab_name: Optional[str],
    planner_hints: Optional[Dict[str, Any]] = None,
) -> Dict:
    query_text = str(question or "").strip()
    context = prepare_db_query(
        question=question,
        intent=intent,
        lab_name=lab_name,
        planner_hints=planner_hints,
    )
    invariant_violation = context.get("invariant_violation")
    if invariant_violation:
        return {
            "answer": str(context.get("fallback_answer") or ""),
            "data": [],
            "cards_retrieved": 0,
            "forecast": None,
            "correlation": None,
            "timescale": "clarify",
            "llm_used": False,
            "time_window": context.get("time_window"),
            "resolved_lab_name": context.get("resolved_lab_name"),
            "sources": [],
            "visualization_type": "none",
            "chart": None,
            "invariant_violation": invariant_violation,
            "evidence": validate_tool_evidence(
                {
                    "evidence_kind": "clarify_gate",
                    "intent": intent.value,
                    "strategy": "clarify",
                    "metric_aliases": [str(context.get("metric_alias") or "")],
                    "resolved_scope": context.get("resolved_lab_name"),
                    "resolved_time_window": context.get("time_window"),
                    "provenance_sources": [],
                    "confidence_notes": ["db_invariant_violation"],
                    "recommendation_allowed": False,
                }
            ),
        }
    answer, llm_used = _render_db_answer_with_llm(
        question=query_text,
        intent=intent,
        metric_alias=context["metric_alias"],
        window_label=context["window_label"],
        rows=context["rows"],
        fallback_answer=context["fallback_answer"],
        time_window=context.get("time_window"),
        forecast=context.get("forecast"),
        correlation=context.get("correlation"),
        knowledge_cards=context.get("knowledge_cards"),
    )
    confidence_notes: List[str] = []
    if not llm_used:
        confidence_notes.append("llm_render_fallback_used")
    if not context.get("rows") and not context.get("forecast"):
        confidence_notes.append("low_data_coverage")

    evidence_sources: List[Dict[str, Any]] = []
    for src in context.get("sources") or []:
        evidence_sources.append(
            {
                "source_kind": str(src.get("source_kind") or "unknown"),
                "table": src.get("table"),
                "operation": src.get("operation"),
                "metric": src.get("metric"),
                "source_label": src.get("source_label"),
                "topic": src.get("topic"),
                "title": src.get("title"),
                "details": {
                    "window_label": src.get("window_label"),
                    "row_count": src.get("row_count"),
                    "lab_scope": src.get("lab_scope"),
                    "metrics": src.get("metrics"),
                },
            }
        )

    evidence = validate_tool_evidence(
        {
            "evidence_kind": "db_query",
            "intent": intent.value,
            "strategy": "direct",
            "metric_aliases": [str(context.get("metric_alias") or "")],
            "resolved_scope": context.get("resolved_lab_name"),
            "resolved_time_window": context.get("time_window"),
            "provenance_sources": evidence_sources,
            "confidence_notes": confidence_notes,
            "recommendation_allowed": True,
        }
    )

    return {
        "answer": answer,
        "data": context["rows"],
        "cards_retrieved": int(context.get("cards_retrieved") or 0),
        "forecast": context.get("forecast"),
        "correlation": context.get("correlation"),
        "timescale": context["timescale"],
        "llm_used": llm_used,
        "time_window": context["time_window"],
        "resolved_lab_name": context["resolved_lab_name"],
        "sources": context.get("sources", []),
        "visualization_type": context.get("visualization_type", "none"),
        "chart": context.get("chart"),
        "evidence": evidence,
    }


async def stream_db_query(
    question: str,
    intent: IntentType,
    lab_name: Optional[str],
    planner_hints: Optional[Dict[str, Any]] = None,
    query_context: Optional[Dict[str, Any]] = None,
    think: Optional[bool] = None,
) -> AsyncIterator[str]:
    query_text = str(question or "").strip()
    context = query_context or prepare_db_query(
        question=question,
        intent=intent,
        lab_name=lab_name,
        planner_hints=planner_hints,
    )
    if context.get("invariant_violation"):
        yield str(context.get("fallback_answer") or "")
        return
    payload = context["payload"]
    fallback_answer = context["fallback_answer"]

    prompt_template = get_shared_prompt_template(
        response_directive=_db_response_directive(intent, question=query_text)
    )
    interpretation_cards, guardrails = _split_knowledge_cards(context.get("knowledge_cards"))
    context_data = build_grounded_context_sections(
        measured_room_facts=payload,
        backend_semantic_state=None,
        knowledge_cards=interpretation_cards,
        communication_guardrails=guardrails,
    )
    messages = prompt_template.format_messages(
        question=query_text,
        context_label="Structured DB Query Result with knowledge grounding",
        context_data=context_data,
    )

    prompt_parts = []
    for m in messages:
        role = getattr(m, "type", "user").upper()
        content = str(getattr(m, "content", "") or "")
        prompt_parts.append(f"{role}:\n{content}")
    prompt_text = "\n\n".join(prompt_parts)

    import os
    import httpx

    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    model = os.getenv("OLLAMA_MODEL", "qwen3:30b-a3b-instruct-2507-q4_K_M")
    api_url = f"{base_url}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt_text,
        "stream": True,
        "temperature": 0.1,
    }
    # Some model/runtime combos emit more raw reasoning when think=False is passed
    # explicitly. We only pass think=True and use server-side filtering for think=False.
    if think is True:
        payload["think"] = True

    in_thinking_block = False
    emitted_anything = False
    include_thinking = think is not False
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("POST", api_url, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    thinking_text = str(event.get("thinking") or "")
                    response_text = str(event.get("response") or "")

                    if include_thinking and thinking_text:
                        if not in_thinking_block:
                            in_thinking_block = True
                            emitted_anything = True
                            yield "<think>"
                        yield thinking_text

                    if response_text:
                        if in_thinking_block:
                            in_thinking_block = False
                            yield "</think>"
                        emitted_anything = True
                        yield response_text
    except Exception:
        yield _ensure_think_prefix(fallback_answer) if include_thinking else str(fallback_answer or "")
        return

    if include_thinking and in_thinking_block:
        yield "</think>"
    elif not emitted_anything:
        # Streaming may complete without tokens in error/edge cases.
        yield _ensure_think_prefix(fallback_answer) if include_thinking else str(fallback_answer or "")
