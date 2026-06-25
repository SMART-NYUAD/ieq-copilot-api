"""Intent-specific query handlers — data fetched via REST API, not direct DB."""

from __future__ import annotations

from datetime import datetime, timedelta
import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

_log = logging.getLogger(__name__)

from query_routing.intent_classifier import IntentType
from executors.db_support import api_client
from executors.db_support import query_parsing as db_parsing
from executors.db_support import time_windows as db_time_windows
from executors.db_support import response_helpers as db_helpers
from executors.db_support.response_helpers import is_diagnostic_query_text
from executors import metric_registry


def _base_result(metric_alias: str, window_label: str) -> Dict[str, Any]:
    return {
        "operation_type": "aggregation",
        "rows": [],
        "fallback_answer": f"I couldn't find {metric_alias} data for {window_label}.",
        "correlation_data": None,
        "metric_alias": metric_alias,
        "metrics_used": [metric_alias],
        "window_start": None,
        "window_end": None,
        "window_label": window_label,
        "compared_spaces": [],
    }


def _is_current_snapshot_query(question: str) -> bool:
    q = str(question or "").lower()
    current_tokens = (
        "current",
        "now",
        "right now",
        "at this moment",
        "latest",
        "most recent",
    )
    snapshot_nouns = (
        "reading",
        "readings",
        "value",
        "values",
        "level",
        "levels",
        "measurement",
        "measurements",
        "status",
    )
    return any(token in q for token in current_tokens) or any(noun in q for noun in snapshot_nouns)


def _is_historical_window_summary_query(question: str, window_label: str, window_start: datetime, window_end: datetime) -> bool:
    q = str(question or "").lower()
    label = str(window_label or "").lower()
    currentness_tokens = (
        "current",
        "now",
        "right now",
        "at this moment",
        "latest",
        "most recent",
    )
    if any(token in q for token in currentness_tokens):
        return False
    if any(token in label for token in ("last week", "this week", "last month", "this month", "yesterday", "today")):
        return True
    if any(token in q for token in ("last week", "this week", "last month", "this month", "yesterday", "today", "past ", "last ")):
        return True
    try:
        window_hours = max(0.0, (window_end - window_start).total_seconds() / 3600.0)
    except Exception:
        window_hours = 0.0
    return window_hours >= 12.0


def _is_full_assessment_query(question: str) -> bool:
    q = str(question or "").lower()
    full_assessment_tokens = (
        "complete assessment",
        "full assessment",
        "full picture",
        "everything you have",
        "environmental assessment",
    )
    return any(token in q for token in full_assessment_tokens)


def _requested_metrics(
    question: str,
    explicit_metrics: List[str],
    hinted_metrics: List[str],
    intent: IntentType,
) -> List[str]:
    q = str(question or "").lower()
    if _is_full_assessment_query(question):
        full_pack = ["ieq", "co2", "pm25", "voc", "humidity", "temperature", "sound", "light"]
        return full_pack

    metrics = list(explicit_metrics) + [m for m in hinted_metrics if m not in explicit_metrics]
    is_comfort_assessment_query = db_helpers.is_comfort_assessment_query_text(question)
    analytical_intents = {
        IntentType.AGGREGATION_DB,
        IntentType.COMPARISON_DB,
        IntentType.ANOMALY_ANALYSIS_DB,
    }
    if explicit_metrics and hinted_metrics and len(hinted_metrics) > len(explicit_metrics) and intent in analytical_intents:
        metrics = list(hinted_metrics) + [m for m in explicit_metrics if m not in hinted_metrics]
    if explicit_metrics:
        explicit_air_metric = any(m in {"co2", "pm25", "voc"} for m in explicit_metrics)
        trend_like_phrase = any(
            token in q
            for token in (
                "trend",
                "trended",
                "over time",
                "this week",
                "last week",
                "this month",
                "last month",
                "past ",
                "last ",
            )
        )
        planner_context_expansion = (
            bool(hinted_metrics)
            and len(hinted_metrics) > len(explicit_metrics)
            and intent in analytical_intents
        )
        if not (
            is_diagnostic_query_text(question)
            or (
                intent == IntentType.COMPARISON_DB
                and explicit_air_metric
                and any(token in q for token in ("compare", "vs", "versus"))
            )
            or (
                intent == IntentType.AGGREGATION_DB
                and explicit_air_metric
                and trend_like_phrase
            )
            or planner_context_expansion
            or (
                intent == IntentType.COMPARISON_DB
                and is_comfort_assessment_query
            )
        ):
            return metrics
    if is_diagnostic_query_text(question):
        required_pack = ["co2", "pm25", "voc", "humidity", "temperature", "ieq", "sound", "light"]
        return required_pack + [m for m in metrics if m not in required_pack]
    is_air_quality_query = db_helpers.is_air_quality_query_text(question)
    if not is_air_quality_query:
        q = str(question or "").lower()
        if (
            intent == IntentType.COMPARISON_DB
            and any(m in {"co2", "pm25", "voc"} for m in metrics)
            and any(token in q for token in ("compare", "vs", "versus"))
        ):
            is_air_quality_query = True
        elif intent == IntentType.AGGREGATION_DB and len(explicit_metrics) == 1 and any(
            m in {"co2", "pm25", "voc"} for m in explicit_metrics
        ) and any(
            token in q
            for token in ("trend", "over time", "this week", "last week", "this month", "last month", "past ", "last ")
        ):
            is_air_quality_query = True
    if not (is_air_quality_query or is_comfort_assessment_query):
        return metrics
    required_pack = (
        ["ieq", "itc", "iaq", "temperature", "humidity", "co2", "pm25", "voc", "sound", "light"]
        if is_comfort_assessment_query
        else ["co2", "pm25", "voc", "humidity", "ieq"]
    )
    return required_pack + [m for m in metrics if m not in required_pack]


def _handle_diagnostic(
    *,
    question: str,
    intent: IntentType,
    requested_metrics: List[str],
    window_start: datetime,
    window_end: datetime,
    window_label: str,
    resolved_lab_name: Optional[str],
) -> Optional[Dict[str, Any]]:
    if not is_diagnostic_query_text(question):
        return None
    core_metrics = ["ieq", "co2", "pm25", "voc", "humidity", "temperature", "sound", "light"]
    # Only fetch metrics the API supports
    fetchable = [m for m in core_metrics if api_client._api_sensor_slug(m) or api_client._score_type(m)]

    if not resolved_lab_name or not fetchable:
        return {
            "operation_type": "diagnostic",
            "rows": [],
            "fallback_answer": f"I couldn't map diagnostic metrics for {window_label}.",
            "correlation_data": None,
            "metric_alias": "ieq",
            "metrics_used": [],
            "window_start": window_start,
            "window_end": window_end,
            "window_label": window_label,
            "compared_spaces": [],
        }

    rows = api_client.fetch_merged_timeseries(resolved_lab_name, fetchable, window_start, window_end)

    if not rows:
        return {
            "operation_type": "diagnostic",
            "rows": [],
            "fallback_answer": f"I couldn't find IEQ diagnostic readings for {window_label}.",
            "correlation_data": None,
            "metric_alias": "ieq",
            "metrics_used": fetchable,
            "window_start": window_start,
            "window_end": window_end,
            "window_label": window_label,
            "compared_spaces": [],
        }

    correlation_analysis = db_helpers.correlate_metrics_with_ieq(rows=rows, metrics=core_metrics)
    fallback = db_helpers.build_diagnostic_answer(
        rows=rows,
        correlation_analysis=correlation_analysis,
        window_label=window_label,
        lab_name=resolved_lab_name,
    )
    return {
        "operation_type": "diagnostic",
        "rows": rows,
        "fallback_answer": fallback,
        "correlation_data": correlation_analysis,
        "metric_alias": "ieq",
        "metrics_used": fetchable,
        "window_start": window_start,
        "window_end": window_end,
        "window_label": window_label,
        "compared_spaces": [],
    }


def _handle_correlation(
    *,
    question: str,
    window_start: datetime,
    window_end: datetime,
    window_label: str,
    resolved_lab_name: Optional[str],
    requested_metrics: List[str],
) -> Optional[Dict[str, Any]]:
    if not (db_parsing.wants_correlation(question) and len(requested_metrics) >= 2):
        return None
    metric_x, metric_y = requested_metrics[0], requested_metrics[1]
    if not resolved_lab_name:
        return {
            "operation_type": "correlation",
            "rows": [],
            "fallback_answer": f"I couldn't map requested metrics for correlation: {metric_x} vs {metric_y}.",
            "correlation_data": None,
            "metric_alias": f"{metric_x}_vs_{metric_y}",
            "metrics_used": [metric_x, metric_y],
        }

    fetch_start, fetch_end = db_time_windows.widen_window_to_min_span(window_start, window_end, 6)
    series_x = api_client.fetch_timeseries_rows(resolved_lab_name, metric_x, fetch_start, fetch_end)
    series_y = api_client.fetch_timeseries_rows(resolved_lab_name, metric_y, fetch_start, fetch_end)

    # Merge by timestamp bucket
    by_bucket_x = {str(r["bucket"]): r["value"] for r in series_x}
    by_bucket_y = {str(r["bucket"]): r["value"] for r in series_y}
    common = sorted(set(by_bucket_x) & set(by_bucket_y))
    rows = [
        {"lab_space": resolved_lab_name, "bucket": b, "x_value": by_bucket_x[b], "y_value": by_bucket_y[b]}
        for b in common
        if by_bucket_x[b] is not None and by_bucket_y[b] is not None
    ]

    corr_value: Optional[float] = None
    if len(rows) >= 3:
        df = pd.DataFrame(rows)
        corr = df["x_value"].corr(df["y_value"])
        if corr is not None and not pd.isna(corr):
            corr_value = float(corr)

    correlation_data = {
        "metric_x": metric_x,
        "metric_y": metric_y,
        "correlation": corr_value,
        "row_count": len(rows),
    }
    return {
        "operation_type": "correlation",
        "rows": rows,
        "fallback_answer": db_helpers.build_correlation_answer(
            metric_x=metric_x,
            metric_y=metric_y,
            correlation=corr_value,
            row_count=len(rows),
            window_label=window_label,
            lab_name=resolved_lab_name,
        ),
        "correlation_data": correlation_data,
        "metric_alias": f"{metric_x}_vs_{metric_y}",
        "metrics_used": [metric_x, metric_y],
    }


def _handle_comparison_multi(
    *,
    question: str,
    intent: IntentType,
    requested_metrics: List[str],
    window_start: datetime,
    window_end: datetime,
    window_label: str,
    resolved_lab_name: Optional[str],
) -> Optional[Dict[str, Any]]:
    if not (intent == IntentType.COMPARISON_DB and len(requested_metrics) >= 2):
        return None
    if db_parsing.is_baseline_reference_query(question):
        return None
    if db_parsing.is_temporal_period_comparison(question):
        return None
    if db_helpers.is_comfort_assessment_query_text(question):
        compare_metrics = requested_metrics[:8]
    elif db_helpers.is_air_quality_query_text(question):
        compare_metrics = requested_metrics[:5]
    else:
        compare_metrics = requested_metrics[:4]

    # Only include metrics that have a mapping (keeps metrics_used consistent)
    metric_names = [m for m in compare_metrics if metric_registry.metric_column(m) is not None]
    if not metric_names:
        return {
            "operation_type": "comparison_multi_metric",
            "rows": [],
            "fallback_answer": "I couldn't map planner-selected metrics for comparison.",
            "metrics_used": [],
        }

    compared_spaces = db_parsing.extract_compared_spaces(question)
    if not compared_spaces and resolved_lab_name:
        compared_spaces = [resolved_lab_name]

    if len(compared_spaces) < 2:
        if resolved_lab_name:
            row = api_client.fetch_multi_metric_agg_row(resolved_lab_name, metric_names, window_start, window_end)
            rows = [row] if row.get(metric_names[0]) is not None else []
            return {
                "operation_type": "comparison_multi_metric",
                "rows": rows,
                "fallback_answer": db_helpers.build_multi_metric_aggregation_answer(
                    metric_aliases=metric_names,
                    row=rows[0] if rows else {},
                    window_label=window_label,
                ),
                "metric_alias": metric_names[0],
                "metrics_used": metric_names,
                "compared_spaces": [resolved_lab_name],
            }
        return {
            "operation_type": "comparison_multi_metric",
            "rows": [],
            "fallback_answer": (
                "I need two explicit spaces for cross-space comparison (for example: "
                "'smart_lab vs concrete_lab')."
            ),
            "metric_alias": metric_names[0],
            "metrics_used": metric_names,
            "compared_spaces": compared_spaces,
        }

    rows = [
        api_client.fetch_multi_metric_agg_row(slug, metric_names, window_start, window_end)
        for slug in compared_spaces[:2]
    ]
    rows = [r for r in rows if any(r.get(m) is not None for m in metric_names)]
    return {
        "operation_type": "comparison_multi_metric",
        "rows": rows,
        "fallback_answer": db_helpers.build_multi_metric_comparison_answer(
            metric_aliases=metric_names,
            rows=rows,
            window_label=window_label,
        ),
        "metric_alias": metric_names[0],
        "metrics_used": metric_names,
        "compared_spaces": compared_spaces[:2],
    }


def _handle_baseline_reference_comparison(
    *,
    question: str,
    intent: IntentType,
    metric_alias: str,
    unit: str,
    window_start: datetime,
    window_end: datetime,
    window_label: str,
    resolved_lab_name: Optional[str],
) -> Optional[Dict[str, Any]]:
    if intent != IntentType.COMPARISON_DB:
        return None
    if not db_parsing.is_baseline_reference_query(question):
        return None
    if not resolved_lab_name:
        return {
            "operation_type": "baseline_reference_comparison",
            "rows": [],
            "fallback_answer": "Please specify one lab to compare against its baseline/reference.",
            "metrics_used": [metric_alias],
            "compared_spaces": [],
        }

    window_seconds = max(3600, int((window_end - window_start).total_seconds()))
    baseline_start = window_start - timedelta(seconds=window_seconds)

    # Fetch the full range (baseline + current) as a single time series
    all_rows = api_client.fetch_timeseries_rows(resolved_lab_name, metric_alias, baseline_start, window_end)

    def _ts(dt: datetime) -> str:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=__import__("datetime").timezone.utc)
        return dt.isoformat()

    current_vals = [
        r["value"] for r in all_rows
        if r.get("value") is not None and str(r.get("bucket", "")) >= _ts(window_start)
    ]
    baseline_vals = [
        r["value"] for r in all_rows
        if r.get("value") is not None and str(r.get("bucket", "")) < _ts(window_start)
    ]

    if not current_vals or not baseline_vals:
        return {
            "operation_type": "baseline_reference_comparison",
            "rows": [],
            "fallback_answer": (
                f"I couldn't compute a baseline comparison for {metric_alias} in {resolved_lab_name} "
                f"for {window_label}."
            ),
            "metrics_used": [metric_alias],
            "compared_spaces": [resolved_lab_name],
        }

    current_value = sum(current_vals) / len(current_vals)
    baseline_value = sum(baseline_vals) / len(baseline_vals)
    import statistics
    baseline_stddev = statistics.pstdev(baseline_vals) if len(baseline_vals) > 1 else None
    delta = current_value - baseline_value
    pct = (delta / baseline_value * 100.0) if baseline_value != 0 else None

    rows = [
        {
            "lab_space": resolved_lab_name,
            "current_avg": current_value,
            "baseline_avg": baseline_value,
            "delta_value": delta,
            "delta_percent": pct,
            "baseline_stddev": baseline_stddev,
            "current_count": len(current_vals),
            "baseline_count": len(baseline_vals),
        }
    ]
    pct_text = f"{pct:+.1f}%" if pct is not None else "n/a"
    direction = "higher" if delta >= 0 else "lower"
    return {
        "operation_type": "baseline_reference_comparison",
        "rows": rows,
        "fallback_answer": (
            f"In {window_label}, {resolved_lab_name} has average {metric_alias} {abs(delta):.2f} {unit} "
            f"({pct_text}) {direction} than its baseline/reference window."
        ),
        "metric_alias": metric_alias,
        "metrics_used": [metric_alias],
        "compared_spaces": [resolved_lab_name],
    }


def _handle_aggregation_multi(
    *,
    question: str,
    intent: IntentType,
    requested_metrics: List[str],
    compared_spaces: List[str],
    window_start: datetime,
    window_end: datetime,
    window_label: str,
    resolved_lab_name: Optional[str],
) -> Optional[Dict[str, Any]]:
    if not (intent == IntentType.AGGREGATION_DB and len(requested_metrics) >= 2 and len(compared_spaces) < 2):
        return None
    if _is_full_assessment_query(question):
        selected_metrics = requested_metrics[:8]
    elif db_helpers.is_comfort_assessment_query_text(question):
        selected_metrics = requested_metrics[:8]
    elif db_helpers.is_air_quality_query_text(question):
        selected_metrics = requested_metrics[:5]
    else:
        selected_metrics = requested_metrics[:4]

    metric_names = [m for m in selected_metrics if metric_registry.metric_column(m) is not None]
    if not metric_names:
        return {
            "operation_type": "aggregation_multi_metric",
            "rows": [],
            "fallback_answer": "I couldn't map requested metrics for analysis.",
            "metrics_used": [],
        }

    if resolved_lab_name:
        row = api_client.fetch_multi_metric_agg_row(resolved_lab_name, metric_names, window_start, window_end)
        rows = [row]
    else:
        row = api_client.fetch_all_spaces_avg_row(metric_names, window_start, window_end)
        rows = [row]

    return {
        "operation_type": "aggregation_multi_metric",
        "rows": rows,
        "fallback_answer": db_helpers.build_multi_metric_aggregation_answer(
            metric_aliases=metric_names,
            row=rows[0] if rows else {},
            window_label=window_label,
        ),
        "metric_alias": metric_names[0],
        "metrics_used": metric_names,
    }


def _handle_point_lookup(
    *,
    question: str,
    intent: IntentType,
    metric_alias: str,
    unit: str,
    requested_metrics: List[str],
    window_start: datetime,
    window_end: datetime,
    window_label: str,
    resolved_lab_name: Optional[str],
) -> Optional[Dict[str, Any]]:
    if intent not in {IntentType.POINT_LOOKUP_DB, IntentType.CURRENT_STATUS_DB}:
        return None

    current_snapshot_query = _is_current_snapshot_query(question)
    historical_summary_query = _is_historical_window_summary_query(
        question=question,
        window_label=window_label,
        window_start=window_start,
        window_end=window_end,
    )

    if historical_summary_query and intent == IntentType.POINT_LOOKUP_DB:
        if len(requested_metrics) >= 2:
            if db_helpers.is_comfort_assessment_query_text(question):
                selected_metrics = requested_metrics[:8]
            else:
                selected_metrics = requested_metrics[:6]
            metric_names = [m for m in selected_metrics if metric_registry.metric_column(m) is not None]
            if not metric_names:
                return {
                    "operation_type": "aggregation_multi_metric",
                    "rows": [],
                    "fallback_answer": "I couldn't map requested metrics for window analysis.",
                    "metrics_used": [],
                }
            if resolved_lab_name:
                row = api_client.fetch_multi_metric_agg_row(resolved_lab_name, metric_names, window_start, window_end)
                rows = [row]
            else:
                row = api_client.fetch_all_spaces_avg_row(metric_names, window_start, window_end)
                rows = [row]
            return {
                "operation_type": "aggregation_multi_metric",
                "rows": rows,
                "fallback_answer": db_helpers.build_multi_metric_aggregation_answer(
                    metric_aliases=metric_names,
                    row=rows[0] if rows else {},
                    window_label=window_label,
                ),
                "metric_alias": metric_names[0],
                "metrics_used": metric_names,
            }

        # Single metric historical summary
        if resolved_lab_name:
            agg = api_client.fetch_aggregation_row(resolved_lab_name, metric_alias, window_start, window_end)
            rows = [agg] if agg else []
        else:
            rows = api_client.fetch_all_spaces_agg_rows_for_metric(metric_alias, window_start, window_end)[:10]
        return {
            "operation_type": "aggregation",
            "rows": rows,
            "fallback_answer": db_helpers.build_aggregation_answer(metric_alias, rows, window_label),
            "metrics_used": [metric_alias],
        }

    is_multi = (
        db_helpers.is_air_quality_query_text(question)
        or db_helpers.is_comfort_assessment_query_text(question)
        or db_helpers.is_issue_triage_query_text(question)
        or (len(requested_metrics) >= 2 and current_snapshot_query)
    )
    if is_multi:
        selected_metrics = requested_metrics[:8] if db_helpers.is_comfort_assessment_query_text(question) else requested_metrics[:5]
        metric_names = [m for m in selected_metrics if metric_registry.metric_column(m) is not None]
        if "ieq" in metric_names:
            for sub in ("iaq", "itc", "iac", "iil"):
                if sub not in metric_names:
                    metric_names.append(sub)
        if not metric_names:
            return {
                "operation_type": "point_lookup_multi_metric",
                "rows": [],
                "fallback_answer": "I couldn't map requested air-quality metrics for point lookup.",
                "metrics_used": [],
            }

        active_window_label = window_label
        window_note = ""
        if resolved_lab_name:
            row = api_client.fetch_multi_metric_point_row(resolved_lab_name, metric_names)
            rows = [row] if any(row.get(m) is not None for m in metric_names) else []
        else:
            # No specific lab — use all-spaces average from /spaces/ endpoint
            spaces_data = api_client.fetch_spaces()
            if spaces_data:
                # Build an aggregate row from first available space or aggregate
                row = api_client.fetch_multi_metric_point_row(spaces_data[0]["slug"], metric_names)
                rows = [row] if any(row.get(m) is not None for m in metric_names) else []
            else:
                rows = []

        fallback_answer = db_helpers.build_multi_metric_aggregation_answer(
            metric_aliases=metric_names,
            row=rows[0] if rows else {},
            window_label=active_window_label,
        )
        if window_note:
            fallback_answer = f"{fallback_answer}{window_note}"
        return {
            "operation_type": "point_lookup_multi_metric",
            "rows": rows,
            "fallback_answer": fallback_answer,
            "metric_alias": metric_names[0],
            "metrics_used": metric_names,
            "window_start": window_start,
            "window_end": window_end,
            "window_label": active_window_label,
        }

    # Single-metric point lookup
    active_window_label = window_label
    window_note = ""
    rows: List[Dict[str, Any]] = []
    trend_rows: List[Dict[str, Any]] = []
    api_trend_pct: Optional[float] = None

    has_time_window = db_parsing.has_explicit_time_hint(question)
    fetch_start, fetch_end = db_time_windows.widen_window_to_min_span(window_start, window_end, 6)
    if resolved_lab_name:
        trend_rows = api_client.fetch_timeseries_rows(resolved_lab_name, metric_alias, fetch_start, fetch_end)

    if not has_time_window and resolved_lab_name and intent in {IntentType.CURRENT_STATUS_DB, IntentType.POINT_LOOKUP_DB}:
        # No explicit time window — use /spaces/{slug}/metrics for the live current value.
        space = api_client.fetch_space_metrics(resolved_lab_name)
        if space:
            api_slug = api_client._api_sensor_slug(metric_alias)
            avg_by_type = {m["type"]: m["avg_value"] for m in (space.get("avg_metrics") or [])}
            trend_by_type = {m["type"]: m.get("trend") for m in (space.get("avg_metrics") or [])}
            if api_slug and trend_by_type.get(api_slug) is not None:
                try:
                    api_trend_pct = float(trend_by_type[api_slug])
                except (TypeError, ValueError):
                    api_trend_pct = None
            val = avg_by_type.get(api_slug) if api_slug else None
            if val is not None:
                rows = [{"lab_space": resolved_lab_name, "bucket": space.get("last_updated"), "value": val}]

    if not rows:
        rows = [trend_rows[-1]] if trend_rows else []

    series_name = resolved_lab_name or (rows[0].get("lab_space") if rows else "selected_scope")
    fallback_answer = db_helpers.build_point_lookup_answer(metric_alias, rows[0] if rows else {}, active_window_label)
    if window_note:
        fallback_answer = f"{fallback_answer}{window_note}"
    return {
        "operation_type": "point_lookup",
        "rows": rows,
        "time_series_rows": trend_rows,
        "api_trend_pct": api_trend_pct,
        "fallback_answer": fallback_answer,
        "metrics_used": [metric_alias],
        "window_start": window_start,
        "window_end": window_end,
        "window_label": active_window_label,
    }


_ANOMALY_CORE_METRICS = ["ieq", "co2", "pm25", "voc", "humidity", "temperature"]


def _handle_anomaly_multi(
    *,
    intent: IntentType,
    explicit_metrics: List[str],
    requested_metrics: List[str],
    window_start: datetime,
    window_end: datetime,
    window_label: str,
    resolved_lab_name: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Anomaly scan across all core metrics when the user didn't request a single specific one."""
    if intent != IntentType.ANOMALY_ANALYSIS_DB:
        return None
    if not resolved_lab_name:
        return None
    # Defer to single-metric handler when user explicitly named exactly one metric.
    if len(explicit_metrics) == 1:
        return None

    # When no metric was explicitly requested, always scan all core metrics so
    # general questions like "any anomaly in the room?" get a full picture.
    if explicit_metrics:
        metrics_to_check = [m for m in requested_metrics if m in _ANOMALY_CORE_METRICS] or _ANOMALY_CORE_METRICS
    else:
        metrics_to_check = _ANOMALY_CORE_METRICS

    fetch_start, fetch_end = db_time_windows.widen_window_to_min_span(window_start, window_end, 6)

    metric_results = []
    all_rows: List[Dict[str, Any]] = []
    metrics_used: List[str] = []
    for metric in metrics_to_check:
        rows = api_client.fetch_timeseries_rows(resolved_lab_name, metric, fetch_start, fetch_end)
        if not rows:
            continue
        anomalies = db_helpers.detect_anomaly_points(rows)
        metric_results.append({"metric": metric, "rows": rows, "anomalies": anomalies})
        all_rows.extend([{**row, "metric": metric} for row in rows])
        metrics_used.append(metric)

    if not metric_results:
        return None

    return {
        "operation_type": "anomaly_multi",
        "rows": all_rows,
        "metric_results": metric_results,
        "fallback_answer": db_helpers.build_multi_metric_anomaly_answer(
            metric_results=metric_results,
            window_label=window_label,
            lab_name=resolved_lab_name,
        ),
        "metrics_used": metrics_used,
    }


def _handle_anomaly(
    *,
    intent: IntentType,
    metric_alias: str,
    window_start: datetime,
    window_end: datetime,
    window_label: str,
    resolved_lab_name: Optional[str],
) -> Optional[Dict[str, Any]]:
    if intent != IntentType.ANOMALY_ANALYSIS_DB:
        return None
    if not resolved_lab_name:
        return None

    # The API requires at least 5 hours to return completed hourly buckets;
    # anomaly detection also needs enough points to compute a meaningful baseline.
    fetch_start, fetch_end = db_time_windows.widen_window_to_min_span(window_start, window_end, 6)
    rows = api_client.fetch_timeseries_rows(resolved_lab_name, metric_alias, fetch_start, fetch_end)
    anomalies = db_helpers.detect_anomaly_points(rows)
    return {
        "operation_type": "anomaly",
        "rows": rows,
        "time_series_rows": rows,
        "fallback_answer": db_helpers.build_anomaly_answer(
            metric_alias=metric_alias,
            rows=rows,
            anomalies=anomalies,
            window_label=window_label,
            lab_name=resolved_lab_name,
        ),
        "metrics_used": [metric_alias],
    }


def _handle_temporal_comparison(
    *,
    question: str,
    intent: IntentType,
    metric_alias: str,
    unit: str,
    window_start: datetime,
    window_end: datetime,
    window_label: str,
    resolved_lab_name: Optional[str],
    requested_metrics: List[str],
) -> Optional[Dict[str, Any]]:
    """Handle within-lab temporal period comparisons (today vs last week, this week vs last month, etc.)."""
    if intent != IntentType.COMPARISON_DB:
        return None
    if not db_parsing.is_temporal_period_comparison(question):
        return None
    if db_parsing.is_baseline_reference_query(question):
        return None

    if not resolved_lab_name:
        return {
            "operation_type": "temporal_comparison",
            "rows": [],
            "fallback_answer": (
                "Please specify a lab to run the period comparison "
                "(e.g., 'today vs last week in smart_lab')."
            ),
            "metrics_used": [metric_alias],
            "compared_spaces": [],
        }

    windows = db_parsing.extract_temporal_comparison_windows(question)
    if not windows:
        return None
    current_start, current_end, current_label, ref_start, ref_end, ref_label = windows

    is_multi = (
        db_helpers.is_air_quality_query_text(question)
        or db_helpers.is_comfort_assessment_query_text(question)
    )
    if is_multi and len(requested_metrics) >= 2:
        cap = 8 if db_helpers.is_comfort_assessment_query_text(question) else 5
        compare_metrics = [m for m in requested_metrics[:cap] if metric_registry.metric_column(m) is not None]
    else:
        compare_metrics = [metric_alias]

    from datetime import timezone as _utc

    def _parse_bucket_utc(bucket: Any) -> Optional[datetime]:
        try:
            return datetime.fromisoformat(str(bucket).replace("Z", "+00:00")).astimezone(_utc.utc)
        except (ValueError, TypeError, AttributeError):
            return None

    current_start_utc = current_start.astimezone(_utc.utc)
    current_end_utc = current_end.astimezone(_utc.utc)
    ref_start_utc = ref_start.astimezone(_utc.utc)
    ref_end_utc = ref_end.astimezone(_utc.utc)

    full_start = min(current_start, ref_start)
    full_end = max(current_end, ref_end)

    if len(compare_metrics) == 1:
        metric = compare_metrics[0]
        all_rows = api_client.fetch_timeseries_rows(resolved_lab_name, metric, full_start, full_end)

        def _in_period(row: dict, start_utc: datetime, end_utc: datetime) -> bool:
            bucket_dt = _parse_bucket_utc(row.get("bucket"))
            return bucket_dt is not None and start_utc <= bucket_dt < end_utc

        current_vals = [float(r["value"]) for r in all_rows if r.get("value") is not None and _in_period(r, current_start_utc, current_end_utc)]
        ref_vals = [float(r["value"]) for r in all_rows if r.get("value") is not None and _in_period(r, ref_start_utc, ref_end_utc)]

        if not current_vals or not ref_vals:
            return {
                "operation_type": "temporal_comparison",
                "rows": [],
                "fallback_answer": (
                    f"I couldn't find enough {metric} data in {resolved_lab_name} "
                    f"to compare {current_label} against {ref_label}."
                ),
                "metric_alias": metric,
                "metrics_used": [metric],
                "compared_spaces": [resolved_lab_name],
                "window_start": ref_start,
                "window_end": current_end,
                "window_label": f"{current_label} vs {ref_label}",
            }

        current_avg = sum(current_vals) / len(current_vals)
        ref_avg = sum(ref_vals) / len(ref_vals)
        delta = current_avg - ref_avg
        pct = (delta / ref_avg * 100.0) if ref_avg != 0 else None

        rows = [{
            "lab_space": resolved_lab_name,
            "period": current_label,
            "avg_value": current_avg,
            "reference_period": ref_label,
            "reference_avg": ref_avg,
            "delta_value": delta,
            "delta_percent": pct,
        }]
        return {
            "operation_type": "temporal_comparison",
            "rows": rows,
            "fallback_answer": db_helpers.build_temporal_comparison_answer(
                metric_aliases=[metric],
                rows=rows,
                current_label=current_label,
                ref_label=ref_label,
                lab_name=resolved_lab_name,
                unit=unit,
            ),
            "metric_alias": metric,
            "metrics_used": [metric],
            "compared_spaces": [resolved_lab_name],
            "window_start": ref_start,
            "window_end": current_end,
            "window_label": f"{current_label} vs {ref_label}",
        }

    # Multi-metric: fetch merged timeseries and split by period
    all_rows = api_client.fetch_merged_timeseries(resolved_lab_name, compare_metrics, full_start, full_end)

    def _in_period(row: dict, start_utc: datetime, end_utc: datetime) -> bool:
        bucket_dt = _parse_bucket_utc(row.get("bucket"))
        return bucket_dt is not None and start_utc <= bucket_dt < end_utc

    current_period_rows = [r for r in all_rows if _in_period(r, current_start_utc, current_end_utc)]
    ref_period_rows = [r for r in all_rows if _in_period(r, ref_start_utc, ref_end_utc)]

    comparison_row: Dict[str, Any] = {
        "lab_space": resolved_lab_name,
        "period": current_label,
        "reference_period": ref_label,
    }
    for metric in compare_metrics:
        c_vals = [float(r[metric]) for r in current_period_rows if r.get(metric) is not None]
        r_vals = [float(r[metric]) for r in ref_period_rows if r.get(metric) is not None]
        if c_vals and r_vals:
            c_avg = sum(c_vals) / len(c_vals)
            r_avg = sum(r_vals) / len(r_vals)
            comparison_row[f"{metric}_current"] = c_avg
            comparison_row[f"{metric}_reference"] = r_avg
            comparison_row[f"{metric}_delta"] = c_avg - r_avg
            comparison_row[metric] = c_avg

    rows = [comparison_row]
    return {
        "operation_type": "temporal_comparison",
        "rows": rows,
        "fallback_answer": db_helpers.build_temporal_comparison_answer(
            metric_aliases=compare_metrics,
            rows=rows,
            current_label=current_label,
            ref_label=ref_label,
            lab_name=resolved_lab_name,
            unit=unit,
        ),
        "metric_alias": compare_metrics[0],
        "metrics_used": compare_metrics,
        "compared_spaces": [resolved_lab_name],
        "window_start": ref_start,
        "window_end": current_end,
        "window_label": f"{current_label} vs {ref_label}",
    }


def _handle_comparison(
    *,
    question: str,
    intent: IntentType,
    metric_alias: str,
    unit: str,
    window_start: datetime,
    window_end: datetime,
    window_label: str,
    resolved_lab_name: Optional[str],
) -> Optional[Dict[str, Any]]:
    if intent != IntentType.COMPARISON_DB:
        return None
    if db_parsing.is_baseline_reference_query(question):
        return None
    if db_parsing.is_temporal_period_comparison(question):
        return None
    # Single-space fallback: fetch aggregation for the resolved lab
    if resolved_lab_name:
        agg = api_client.fetch_aggregation_row(resolved_lab_name, metric_alias, window_start, window_end)
        rows = [agg] if agg else []
        fetch_start, fetch_end = db_time_windows.widen_window_to_min_span(window_start, window_end, 6)
        time_series_rows = api_client.fetch_timeseries_rows(
            resolved_lab_name,
            metric_alias,
            fetch_start,
            fetch_end,
        )
        return {
            "operation_type": "aggregation",
            "rows": rows,
            "time_series_rows": time_series_rows,
            "aggregation_summary": agg,
            "fallback_answer": db_helpers.build_aggregation_answer(metric_alias, rows, window_label),
            "metrics_used": [metric_alias],
            "compared_spaces": [resolved_lab_name],
        }
    return None


def _handle_default(
    *,
    question: str,
    intent: IntentType,
    metric_alias: str,
    window_start: datetime,
    window_end: datetime,
    window_label: str,
    resolved_lab_name: Optional[str],
    compared_spaces: List[str],
) -> Dict[str, Any]:
    fetch_start, fetch_end = db_time_windows.widen_window_to_min_span(window_start, window_end, 6)

    if intent == IntentType.AGGREGATION_DB and len(compared_spaces) >= 2:
        rows = []
        for slug in compared_spaces[:2]:
            agg = api_client.fetch_aggregation_row(slug, metric_alias, window_start, window_end)
            if agg:
                rows.append(agg)
        rows.sort(key=lambda r: (r.get("avg_value") or 0), reverse=True)
        return {
            "operation_type": "comparison",
            "rows": rows,
            "fallback_answer": db_helpers.build_comparison_answer(metric_alias, rows, window_label),
            "metrics_used": [metric_alias],
            "compared_spaces": compared_spaces[:2],
        }

    if db_parsing.wants_time_series(question):
        slug = resolved_lab_name or ""
        rows = api_client.fetch_timeseries_rows(slug, metric_alias, fetch_start, fetch_end)
        return {
            "operation_type": "timeseries",
            "rows": rows,
            "time_series_rows": rows,
            "fallback_answer": db_helpers.build_timeseries_answer(metric_alias, rows, window_label),
            "metrics_used": [metric_alias],
        }

    # Default aggregation — include IEQ contribution sub-scores from /spaces/
    time_series_rows: List[Dict[str, Any]] = []
    aggregation_summary: Optional[Dict[str, Any]] = None
    if resolved_lab_name:
        agg = api_client.fetch_aggregation_row(resolved_lab_name, metric_alias, window_start, window_end)
        rows = []
        if agg:
            aggregation_summary = agg
            # Enrich with contribution fields from /spaces/ summary
            spaces = api_client.fetch_spaces()
            space_summary = next((s for s in spaces if s.get("slug") == resolved_lab_name), None)
            if space_summary:
                sub = space_summary.get("metrics") or {}
                agg["contri_thermal"] = sub.get("itc")
                agg["contri_light"] = sub.get("iil")
                agg["contri_air"] = sub.get("iaq")
                agg["contri_acoustic"] = sub.get("iac")
                agg["contri_acustic"] = sub.get("iac")
            rows = [agg]
        time_series_rows = api_client.fetch_timeseries_rows(
            resolved_lab_name,
            metric_alias,
            fetch_start,
            fetch_end,
        )
    else:
        rows = api_client.fetch_all_spaces_agg_rows_for_metric(metric_alias, window_start, window_end)[:10]

    return {
        "operation_type": "aggregation",
        "rows": rows,
        "time_series_rows": time_series_rows,
        "aggregation_summary": aggregation_summary,
        "fallback_answer": db_helpers.build_aggregation_answer(metric_alias, rows, window_label),
        "metrics_used": [metric_alias],
    }


def _handle_forecast(
    *,
    intent: IntentType,
    metric_alias: str,
    resolved_lab_name: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Fetch 6-hour ahead predictions from the Smart CRG predictions API."""
    if intent != IntentType.FORECAST_DB:
        return None
    if not resolved_lab_name:
        return {
            "operation_type": "forecast",
            "rows": [],
            "fallback_answer": (
                "I need a lab name to fetch the forecast. "
                "Which lab should I check (for example: smart_lab, concrete_lab)?"
            ),
            "metrics_used": [metric_alias],
            "time_series_rows": [],
        }

    data = api_client.fetch_predictions(resolved_lab_name, metric_alias)
    if data is None:
        return {
            "operation_type": "forecast",
            "rows": [],
            "fallback_answer": (
                f"The forecast service is currently unavailable for {metric_alias} "
                f"in {resolved_lab_name}. Please try again in a moment."
            ),
            "metrics_used": [metric_alias],
            "time_series_rows": [],
        }

    # Normalise varying API response shapes
    predictions = (
        data.get("predictions")
        or data.get("forecast")
        or (data.get("data") or {}).get("predictions")
        or []
    )
    rows = []
    for item in predictions:
        bucket = item.get("timestamp") or item.get("time") or item.get("bucket")
        value = (
            item.get("predicted_value")
            or item.get("value")
            or item.get("prediction")
        )
        if bucket is not None:
            rows.append(
                {
                    "lab_space": resolved_lab_name,
                    "bucket": bucket,
                    "value": value,
                    "is_forecast": True,
                }
            )

    display = metric_registry.metric_display(metric_alias)
    if not rows:
        fallback = (
            f"No prediction data was returned for {display} in {resolved_lab_name}. "
            "The model may not have generated forecasts yet."
        )
    else:
        n = len(rows)
        fallback = (
            f"Here are the next {n} predicted {display} readings for {resolved_lab_name} "
            f"(covering up to 6 hours ahead)."
        )

    return {
        "operation_type": "forecast",
        "rows": rows,
        "fallback_answer": fallback,
        "metrics_used": [metric_alias],
        "time_series_rows": rows,
    }


def execute_intent_query(
    *,
    question: str,
    intent: IntentType,
    metric_alias: str,
    metric_column: str,
    unit: str,
    window_start: datetime,
    window_end: datetime,
    window_label: str,
    resolved_lab_name: Optional[str],
    compared_spaces: List[str],
    explicit_metrics: List[str],
    hinted_metrics: List[str],
    # Legacy parameter kept for backwards compatibility — no longer used
    cur: Any = None,
) -> Dict[str, Any]:
    result = _base_result(metric_alias=metric_alias, window_label=window_label)
    result["window_start"] = window_start
    result["window_end"] = window_end
    result["compared_spaces"] = list(compared_spaces)

    requested_metrics = _requested_metrics(
        question,
        explicit_metrics,
        hinted_metrics,
        intent,
    )

    handlers: List[Tuple[str, Any]] = [
        ("forecast", lambda: _handle_forecast(
            intent=intent,
            metric_alias=metric_alias,
            resolved_lab_name=resolved_lab_name,
        )),
        ("diagnostic", lambda: _handle_diagnostic(
            question=question,
            intent=intent,
            requested_metrics=requested_metrics,
            window_start=window_start,
            window_end=window_end,
            window_label=window_label,
            resolved_lab_name=resolved_lab_name,
        )),
        ("correlation", lambda: _handle_correlation(
            question=question,
            window_start=window_start,
            window_end=window_end,
            window_label=window_label,
            resolved_lab_name=resolved_lab_name,
            requested_metrics=requested_metrics,
        )),
        ("comparison_multi", lambda: _handle_comparison_multi(
            question=question,
            intent=intent,
            requested_metrics=requested_metrics,
            window_start=window_start,
            window_end=window_end,
            window_label=window_label,
            resolved_lab_name=resolved_lab_name,
        )),
        ("baseline_reference_comparison", lambda: _handle_baseline_reference_comparison(
            question=question,
            intent=intent,
            metric_alias=metric_alias,
            unit=unit,
            window_start=window_start,
            window_end=window_end,
            window_label=window_label,
            resolved_lab_name=resolved_lab_name,
        )),
        ("temporal_comparison", lambda: _handle_temporal_comparison(
            question=question,
            intent=intent,
            metric_alias=metric_alias,
            unit=unit,
            window_start=window_start,
            window_end=window_end,
            window_label=window_label,
            resolved_lab_name=resolved_lab_name,
            requested_metrics=requested_metrics,
        )),
        ("aggregation_multi", lambda: _handle_aggregation_multi(
            question=question,
            intent=intent,
            requested_metrics=requested_metrics,
            compared_spaces=compared_spaces,
            window_start=window_start,
            window_end=window_end,
            window_label=window_label,
            resolved_lab_name=resolved_lab_name,
        )),
        ("point_lookup", lambda: _handle_point_lookup(
            question=question,
            intent=intent,
            metric_alias=metric_alias,
            unit=unit,
            requested_metrics=requested_metrics,
            window_start=window_start,
            window_end=window_end,
            window_label=window_label,
            resolved_lab_name=resolved_lab_name,
        )),
        ("anomaly_multi", lambda: _handle_anomaly_multi(
            intent=intent,
            explicit_metrics=explicit_metrics,
            requested_metrics=requested_metrics,
            window_start=window_start,
            window_end=window_end,
            window_label=window_label,
            resolved_lab_name=resolved_lab_name,
        )),
        ("anomaly", lambda: _handle_anomaly(
            intent=intent,
            metric_alias=metric_alias,
            window_start=window_start,
            window_end=window_end,
            window_label=window_label,
            resolved_lab_name=resolved_lab_name,
        )),
        ("comparison", lambda: _handle_comparison(
            question=question,
            intent=intent,
            metric_alias=metric_alias,
            unit=unit,
            window_start=window_start,
            window_end=window_end,
            window_label=window_label,
            resolved_lab_name=resolved_lab_name,
        )),
    ]

    selected: Optional[Dict[str, Any]] = None
    for handler_name, handle in handlers:
        candidate = handle()
        if candidate:
            _log.debug("execute_intent_query: selected handler=%s intent=%s", handler_name, intent)
            selected = candidate
            break
    if not selected:
        selected = _handle_default(
            question=question,
            intent=intent,
            metric_alias=metric_alias,
            window_start=window_start,
            window_end=window_end,
            window_label=window_label,
            resolved_lab_name=resolved_lab_name,
            compared_spaces=compared_spaces,
        )

    result.update(selected)
    result.setdefault("window_start", window_start)
    result.setdefault("window_end", window_end)
    result.setdefault("window_label", window_label)
    result.setdefault("compared_spaces", list(compared_spaces))
    result.setdefault("correlation_data", None)
    return result
