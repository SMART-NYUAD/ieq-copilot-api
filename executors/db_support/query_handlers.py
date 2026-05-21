"""Intent-specific query handlers — data fetched via REST API, not direct DB."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    from query_routing.intent_classifier import IntentType
except ImportError:
    from ...query_routing.intent_classifier import IntentType

try:
    from executors.db_support import api_client
    from executors.db_support import charts as db_charts
    from executors.db_support import query_parsing as db_parsing
    from executors.db_support import response_helpers as db_helpers
    from executors.db_support.response_helpers import is_diagnostic_query_text
    from executors import metric_registry
except ImportError:
    from . import api_client
    from . import charts as db_charts
    from . import query_parsing as db_parsing
    from . import response_helpers as db_helpers
    from .response_helpers import is_diagnostic_query_text
    from .. import metric_registry


def _base_result(metric_alias: str, window_label: str) -> Dict[str, Any]:
    return {
        "operation_type": "aggregation",
        "rows": [],
        "fallback_answer": f"I couldn't find {metric_alias} data for {window_label}.",
        "chart_payload": db_charts.empty_chart(),
        "forecast_data": None,
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
        full_pack = ["ieq", "co2", "pm25", "tvoc", "humidity", "temperature", "sound", "light"]
        return full_pack

    metrics = list(explicit_metrics) + [m for m in hinted_metrics if m not in explicit_metrics]
    analytical_intents = {
        IntentType.AGGREGATION_DB,
        IntentType.COMPARISON_DB,
        IntentType.ANOMALY_ANALYSIS_DB,
    }
    if explicit_metrics and hinted_metrics and len(hinted_metrics) > len(explicit_metrics) and intent in analytical_intents:
        metrics = list(hinted_metrics) + [m for m in explicit_metrics if m not in hinted_metrics]
    if explicit_metrics:
        explicit_air_metric = any(m in {"co2", "pm25", "tvoc"} for m in explicit_metrics)
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
        ):
            return metrics
    if is_diagnostic_query_text(question):
        required_pack = ["co2", "pm25", "tvoc", "humidity", "temperature", "ieq", "sound", "light"]
        return required_pack + [m for m in metrics if m not in required_pack]
    is_air_quality_query = db_helpers.is_air_quality_query_text(question)
    is_comfort_assessment_query = db_helpers.is_comfort_assessment_query_text(question)
    if not is_air_quality_query:
        q = str(question or "").lower()
        if (
            intent == IntentType.COMPARISON_DB
            and any(m in {"co2", "pm25", "tvoc"} for m in metrics)
            and any(token in q for token in ("compare", "vs", "versus"))
        ):
            is_air_quality_query = True
        elif intent == IntentType.AGGREGATION_DB and len(explicit_metrics) == 1 and any(
            m in {"co2", "pm25", "tvoc"} for m in explicit_metrics
        ) and any(
            token in q
            for token in ("trend", "over time", "this week", "last week", "this month", "last month", "past ", "last ")
        ):
            is_air_quality_query = True
    if not (is_air_quality_query or is_comfort_assessment_query):
        return metrics
    required_pack = (
        ["ieq", "temperature", "humidity", "co2", "pm25", "tvoc", "sound", "light"]
        if is_comfort_assessment_query
        else ["co2", "pm25", "tvoc", "humidity", "ieq"]
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
    max_chart_lookback_points: int,
) -> Optional[Dict[str, Any]]:
    if not is_diagnostic_query_text(question):
        return None
    core_metrics = ["ieq", "co2", "pm25", "tvoc", "humidity", "temperature", "sound", "light"]
    # Only fetch metrics the API supports
    fetchable = [m for m in core_metrics if api_client._api_sensor_slug(m) or api_client._score_type(m)]

    if not resolved_lab_name or not fetchable:
        return {
            "operation_type": "diagnostic",
            "rows": [],
            "fallback_answer": f"I couldn't map diagnostic metrics for {window_label}.",
            "chart_payload": db_charts.empty_chart(),
            "forecast_data": None,
            "correlation_data": None,
            "metric_alias": "ieq",
            "metrics_used": [],
            "window_start": window_start,
            "window_end": window_end,
            "window_label": window_label,
            "compared_spaces": [],
        }

    window_hours = api_client.window_hours_from_datetimes(window_start, window_end)
    rows = api_client.fetch_merged_timeseries(resolved_lab_name, fetchable, window_hours)

    if not rows:
        return {
            "operation_type": "diagnostic",
            "rows": [],
            "fallback_answer": f"I couldn't find IEQ diagnostic readings for {window_label}.",
            "chart_payload": db_charts.empty_chart(),
            "forecast_data": None,
            "correlation_data": None,
            "metric_alias": "ieq",
            "metrics_used": fetchable,
            "window_start": window_start,
            "window_end": window_end,
            "window_label": window_label,
            "compared_spaces": [],
        }

    correlation_analysis = db_helpers.correlate_metrics_with_ieq(rows=rows, metrics=core_metrics)
    top_culprits = list(correlation_analysis.get("top_culprits") or [])
    chart = db_helpers.build_diagnostic_chart(
        rows=rows,
        culprit_metrics=top_culprits[:3],
        window_label=window_label,
        lab_name=resolved_lab_name,
        max_lookback=max_chart_lookback_points,
    )
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
        "chart_payload": chart,
        "forecast_data": None,
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
    max_chart_lookback_points: int,
) -> Optional[Dict[str, Any]]:
    if not (db_parsing.wants_correlation(question) and len(requested_metrics) >= 2):
        return None
    metric_x, metric_y = requested_metrics[0], requested_metrics[1]
    if not resolved_lab_name:
        return {
            "operation_type": "correlation",
            "rows": [],
            "fallback_answer": f"I couldn't map requested metrics for correlation: {metric_x} vs {metric_y}.",
            "chart_payload": db_charts.empty_chart(),
            "correlation_data": None,
            "metric_alias": f"{metric_x}_vs_{metric_y}",
            "metrics_used": [metric_x, metric_y],
        }

    window_hours = max(api_client.window_hours_from_datetimes(window_start, window_end), 6)
    series_x = api_client.fetch_timeseries_rows(resolved_lab_name, metric_x, window_hours)
    series_y = api_client.fetch_timeseries_rows(resolved_lab_name, metric_y, window_hours)

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
        "chart_payload": db_charts.build_scatter_chart(
            metric_x=metric_x,
            metric_y=metric_y,
            unit_x=db_helpers.metric_unit(metric_x),
            unit_y=db_helpers.metric_unit(metric_y),
            window_label=window_label,
            rows=rows,
            correlation=corr_value,
            lookback_points=max_chart_lookback_points,
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
            "chart_payload": db_charts.empty_chart(),
            "metrics_used": [],
        }

    compared_spaces = db_parsing.extract_compared_spaces(question)
    if not compared_spaces and resolved_lab_name:
        compared_spaces = [resolved_lab_name]

    window_hours = api_client.window_hours_from_datetimes(window_start, window_end)

    if len(compared_spaces) < 2:
        if resolved_lab_name:
            row = api_client.fetch_multi_metric_agg_row(resolved_lab_name, metric_names, window_hours)
            rows = [row] if row.get(metric_names[0]) is not None else []
            return {
                "operation_type": "comparison_multi_metric",
                "rows": rows,
                "fallback_answer": db_helpers.build_multi_metric_aggregation_answer(
                    metric_aliases=metric_names,
                    row=rows[0] if rows else {},
                    window_label=window_label,
                ),
                "chart_payload": db_helpers.build_multi_metric_snapshot_chart(
                    metric_aliases=metric_names,
                    unit_by_metric={m: db_helpers.metric_unit(m) for m in metric_names},
                    window_label=window_label,
                    row=rows[0] if rows else {},
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
            "chart_payload": db_charts.empty_chart(),
            "metric_alias": metric_names[0],
            "metrics_used": metric_names,
            "compared_spaces": compared_spaces,
        }

    rows = [
        api_client.fetch_multi_metric_agg_row(slug, metric_names, window_hours)
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
        "chart_payload": db_helpers.build_multi_metric_bar_chart(
            metric_aliases=metric_names,
            unit_by_metric={m: db_helpers.metric_unit(m) for m in metric_names},
            window_label=window_label,
            rows=rows,
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
            "chart_payload": db_charts.empty_chart(),
            "metrics_used": [metric_alias],
            "compared_spaces": [],
        }

    window_seconds = max(3600, int((window_end - window_start).total_seconds()))
    baseline_start = window_start - timedelta(seconds=window_seconds)

    # Fetch the full range (baseline + current) as a single time series
    full_window_hours = api_client.window_hours_from_datetimes(baseline_start, window_end)
    all_rows = api_client.fetch_timeseries_rows(resolved_lab_name, metric_alias, full_window_hours)

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
            "chart_payload": db_charts.empty_chart(),
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
        "chart_payload": {
            "visualization_type": "bar",
            "chart": {
                "title": f"{metric_alias} vs baseline ({window_label})",
                "x_label": "reference",
                "y_label": f"{metric_alias} ({unit})",
                "series": [
                    {
                        "name": resolved_lab_name,
                        "points": [
                            {"x": "baseline", "y": baseline_value},
                            {"x": "current", "y": current_value},
                        ],
                    }
                ],
            },
        },
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
            "chart_payload": db_charts.empty_chart(),
            "metrics_used": [],
        }

    window_hours = api_client.window_hours_from_datetimes(window_start, window_end)

    if resolved_lab_name:
        row = api_client.fetch_multi_metric_agg_row(resolved_lab_name, metric_names, window_hours)
        rows = [row]
    else:
        row = api_client.fetch_all_spaces_avg_row(metric_names, window_hours)
        rows = [row]

    return {
        "operation_type": "aggregation_multi_metric",
        "rows": rows,
        "fallback_answer": db_helpers.build_multi_metric_aggregation_answer(
            metric_aliases=metric_names,
            row=rows[0] if rows else {},
            window_label=window_label,
        ),
        "chart_payload": db_helpers.build_multi_metric_snapshot_chart(
            metric_aliases=metric_names,
            unit_by_metric={m: db_helpers.metric_unit(m) for m in metric_names},
            window_label=window_label,
            row=rows[0] if rows else {},
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
    max_chart_lookback_points: int,
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
    window_hours = api_client.window_hours_from_datetimes(window_start, window_end)

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
                    "chart_payload": db_charts.empty_chart(),
                    "metrics_used": [],
                }
            if resolved_lab_name:
                row = api_client.fetch_multi_metric_agg_row(resolved_lab_name, metric_names, window_hours)
                rows = [row]
            else:
                row = api_client.fetch_all_spaces_avg_row(metric_names, window_hours)
                rows = [row]
            return {
                "operation_type": "aggregation_multi_metric",
                "rows": rows,
                "fallback_answer": db_helpers.build_multi_metric_aggregation_answer(
                    metric_aliases=metric_names,
                    row=rows[0] if rows else {},
                    window_label=window_label,
                ),
                "chart_payload": db_helpers.build_multi_metric_snapshot_chart(
                    metric_aliases=metric_names,
                    unit_by_metric={m: db_helpers.metric_unit(m) for m in metric_names},
                    window_label=window_label,
                    row=rows[0] if rows else {},
                ),
                "metric_alias": metric_names[0],
                "metrics_used": metric_names,
            }

        # Single metric historical summary
        if resolved_lab_name:
            agg = api_client.fetch_aggregation_row(resolved_lab_name, metric_alias, window_hours)
            rows = [agg] if agg else []
        else:
            rows = api_client.fetch_all_spaces_agg_rows_for_metric(metric_alias, window_hours)[:10]
        return {
            "operation_type": "aggregation",
            "rows": rows,
            "fallback_answer": db_helpers.build_aggregation_answer(metric_alias, rows, window_label),
            "chart_payload": db_charts.build_bar_chart(
                metric_alias=metric_alias,
                unit=unit,
                window_label=window_label,
                rows=rows,
                value_key="avg_value",
            ),
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
        if not metric_names:
            return {
                "operation_type": "point_lookup_multi_metric",
                "rows": [],
                "fallback_answer": "I couldn't map requested air-quality metrics for point lookup.",
                "chart_payload": db_charts.empty_chart(),
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
            "chart_payload": db_helpers.build_multi_metric_snapshot_chart(
                metric_aliases=metric_names,
                unit_by_metric={m: db_helpers.metric_unit(m) for m in metric_names},
                window_label=active_window_label,
                row=rows[0] if rows else {},
            ),
            "metric_alias": metric_names[0],
            "metrics_used": metric_names,
            "window_start": window_start,
            "window_end": window_end,
            "window_label": active_window_label,
        }

    # Single-metric point lookup
    # Use a slightly wider window (2h) so we always catch the most recent completed bucket.
    fetch_hours = max(window_hours, 6)
    trend_rows = api_client.fetch_timeseries_rows(resolved_lab_name or "", metric_alias, fetch_hours)
    rows = [trend_rows[-1]] if trend_rows else []
    active_window_label = window_label
    window_note = ""

    # If still no timeseries data, fall back to the space-level metrics endpoint for current value.
    if not rows and resolved_lab_name and intent in {IntentType.CURRENT_STATUS_DB, IntentType.POINT_LOOKUP_DB}:
        space = api_client.fetch_space_metrics(resolved_lab_name)
        if space:
            api_slug = api_client._api_sensor_slug(metric_alias)
            avg_by_type = {m["type"]: m["avg_value"] for m in (space.get("avg_metrics") or [])}
            val = avg_by_type.get(api_slug) if api_slug else None
            if val is not None:
                rows = [{"lab_space": resolved_lab_name, "bucket": space.get("last_updated"), "value": val}]
                trend_rows = rows

    series_name = resolved_lab_name or (rows[0].get("lab_space") if rows else "selected_scope")
    fallback_answer = db_helpers.build_point_lookup_answer(metric_alias, rows[0] if rows else {}, active_window_label)
    if window_note:
        fallback_answer = f"{fallback_answer}{window_note}"
    return {
        "operation_type": "point_lookup",
        "rows": rows,
        "fallback_answer": fallback_answer,
        "chart_payload": db_charts.build_line_chart(
            metric_alias=metric_alias,
            unit=unit,
            window_label=active_window_label,
            series_rows=trend_rows,
            series_name=str(series_name),
            lookback_points=max_chart_lookback_points,
        ),
        "metrics_used": [metric_alias],
        "window_start": window_start,
        "window_end": window_end,
        "window_label": active_window_label,
    }


def _handle_anomaly(
    *,
    intent: IntentType,
    metric_alias: str,
    unit: str,
    window_start: datetime,
    window_end: datetime,
    window_label: str,
    resolved_lab_name: Optional[str],
    max_chart_lookback_points: int,
) -> Optional[Dict[str, Any]]:
    if intent != IntentType.ANOMALY_ANALYSIS_DB:
        return None
    if not resolved_lab_name:
        return None

    window_hours = api_client.window_hours_from_datetimes(window_start, window_end)
    # The API requires at least 5 hours to return completed hourly buckets;
    # anomaly detection also needs enough points to compute a meaningful baseline.
    fetch_hours = max(window_hours, 6)
    rows = api_client.fetch_timeseries_rows(resolved_lab_name, metric_alias, fetch_hours)
    anomalies = db_helpers.detect_anomaly_points(rows)
    series_name = resolved_lab_name
    return {
        "operation_type": "anomaly",
        "rows": rows,
        "fallback_answer": db_helpers.build_anomaly_answer(
            metric_alias=metric_alias,
            rows=rows,
            anomalies=anomalies,
            window_label=window_label,
            lab_name=resolved_lab_name,
        ),
        "chart_payload": db_charts.build_anomaly_chart(
            metric_alias=metric_alias,
            unit=unit,
            window_label=window_label,
            series_rows=rows,
            anomalies=anomalies,
            series_name=str(series_name),
            lookback_points=max_chart_lookback_points,
        ),
        "metrics_used": [metric_alias],
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
    compared_spaces = db_parsing.extract_compared_spaces(question)
    if len(compared_spaces) < 2:
        return {
            "operation_type": "comparison",
            "rows": [],
            "fallback_answer": (
                "I need two explicit spaces for cross-space comparison (for example: "
                "'smart_lab vs concrete_lab')."
            ),
            "chart_payload": db_charts.empty_chart(),
            "metrics_used": [metric_alias],
            "compared_spaces": compared_spaces,
        }

    window_hours = api_client.window_hours_from_datetimes(window_start, window_end)
    rows = []
    for slug in compared_spaces[:2]:
        agg = api_client.fetch_aggregation_row(slug, metric_alias, window_hours)
        if agg:
            rows.append(agg)
    rows.sort(key=lambda r: (r.get("avg_value") or 0), reverse=True)
    return {
        "operation_type": "comparison",
        "rows": rows,
        "fallback_answer": db_helpers.build_comparison_answer(metric_alias, rows, window_label),
        "chart_payload": db_charts.build_bar_chart(
            metric_alias=metric_alias,
            unit=unit,
            window_label=window_label,
            rows=rows,
            value_key="avg_value",
        ),
        "metrics_used": [metric_alias],
        "compared_spaces": compared_spaces[:2],
    }


def _handle_forecast(
    *,
    question: str,
    intent: IntentType,
    metric_alias: str,
    unit: str,
    window_start: datetime,
    window_end: datetime,
    window_label: str,
    resolved_lab_name: Optional[str],
    max_chart_lookback_points: int,
) -> Optional[Dict[str, Any]]:
    if not (intent == IntentType.FORECAST_DB or db_parsing.wants_forecast(question)):
        return None
    horizon_hours, _ = db_parsing.extract_forecast_horizon_hours(question)
    forecast_window_start, forecast_window_end, forecast_window_label = db_parsing.forecast_history_window(
        question=question,
        horizon_hours=horizon_hours,
        default_start=window_start,
        default_end=window_end,
        default_label=window_label,
    )

    slug = resolved_lab_name or ""
    forecast_window_hours = max(
        api_client.window_hours_from_datetimes(forecast_window_start, forecast_window_end), 6
    )
    model_rows = api_client.fetch_timeseries_rows(slug, metric_alias, forecast_window_hours)
    model_rows = model_rows[-1000:] if len(model_rows) > 1000 else model_rows

    forecast_data = db_helpers.build_forecast_from_rows(model_rows, horizon_hours=horizon_hours)
    chart_rows = model_rows
    if forecast_window_start != window_start or forecast_window_end != window_end:
        requested_hours = max(api_client.window_hours_from_datetimes(window_start, window_end), 6)
        requested_rows = api_client.fetch_timeseries_rows(slug, metric_alias, requested_hours)
        if requested_rows:
            chart_rows = requested_rows

    effective_horizon_hours = int((forecast_data or {}).get("horizon_hours") or horizon_hours)
    effective_horizon_label = f"next {effective_horizon_hours} hour(s)"
    series_name = resolved_lab_name or "selected_scope"
    fallback_text = db_helpers.build_forecast_answer(
        metric_alias=metric_alias,
        forecast=forecast_data,
        window_label=window_label,
        horizon_label=effective_horizon_label,
    )
    if forecast_window_label != window_label:
        fallback_text = (
            f"{fallback_text} "
            f"(Model history window: {forecast_window_label}.)"
        )
    return {
        "operation_type": "prediction",
        "rows": chart_rows,
        "fallback_answer": fallback_text,
        "chart_payload": db_charts.build_forecast_chart(
            metric_alias=metric_alias,
            unit=unit,
            window_label=f"{window_label} + {effective_horizon_label}",
            history_rows=chart_rows,
            forecast=forecast_data,
            series_name=str(series_name),
            lookback_points=max_chart_lookback_points,
        ),
        "forecast_data": forecast_data,
        "metrics_used": [metric_alias],
        "forecast_history_start": forecast_window_start,
        "forecast_history_end": forecast_window_end,
        "forecast_history_label": forecast_window_label,
    }


def _handle_default(
    *,
    question: str,
    intent: IntentType,
    metric_alias: str,
    unit: str,
    window_start: datetime,
    window_end: datetime,
    window_label: str,
    resolved_lab_name: Optional[str],
    compared_spaces: List[str],
    max_chart_lookback_points: int,
) -> Dict[str, Any]:
    window_hours = api_client.window_hours_from_datetimes(window_start, window_end)

    if intent == IntentType.AGGREGATION_DB and len(compared_spaces) >= 2:
        rows = []
        for slug in compared_spaces[:2]:
            agg = api_client.fetch_aggregation_row(slug, metric_alias, window_hours)
            if agg:
                rows.append(agg)
        rows.sort(key=lambda r: (r.get("avg_value") or 0), reverse=True)
        return {
            "operation_type": "comparison",
            "rows": rows,
            "fallback_answer": db_helpers.build_comparison_answer(metric_alias, rows, window_label),
            "chart_payload": db_charts.build_bar_chart(
                metric_alias=metric_alias,
                unit=unit,
                window_label=window_label,
                rows=rows,
                value_key="avg_value",
            ),
            "metrics_used": [metric_alias],
            "compared_spaces": compared_spaces[:2],
        }

    if db_parsing.wants_time_series(question):
        slug = resolved_lab_name or ""
        rows = api_client.fetch_timeseries_rows(slug, metric_alias, max(window_hours, 6))
        series_name = resolved_lab_name or (rows[0].get("lab_space") if rows else "selected_scope")
        return {
            "operation_type": "timeseries",
            "rows": rows,
            "fallback_answer": db_helpers.build_timeseries_answer(metric_alias, rows, window_label),
            "chart_payload": db_charts.build_line_chart(
                metric_alias=metric_alias,
                unit=unit,
                window_label=window_label,
                series_rows=rows,
                series_name=str(series_name),
                lookback_points=max_chart_lookback_points,
            ),
            "metrics_used": [metric_alias],
        }

    # Default aggregation — include IEQ contribution sub-scores from /spaces/
    if resolved_lab_name:
        agg = api_client.fetch_aggregation_row(resolved_lab_name, metric_alias, window_hours)
        rows = []
        if agg:
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
    else:
        rows = api_client.fetch_all_spaces_agg_rows_for_metric(metric_alias, window_hours)[:10]

    return {
        "operation_type": "aggregation",
        "rows": rows,
        "fallback_answer": db_helpers.build_aggregation_answer(metric_alias, rows, window_label),
        "chart_payload": db_charts.build_bar_chart(
            metric_alias=metric_alias,
            unit=unit,
            window_label=window_label,
            rows=rows,
            value_key="avg_value",
        ),
        "metrics_used": [metric_alias],
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
    max_chart_lookback_points: int,
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

    handlers = [
        lambda: _handle_diagnostic(
            question=question,
            intent=intent,
            requested_metrics=requested_metrics,
            window_start=window_start,
            window_end=window_end,
            window_label=window_label,
            resolved_lab_name=resolved_lab_name,
            max_chart_lookback_points=max_chart_lookback_points,
        ),
        lambda: _handle_correlation(
            question=question,
            window_start=window_start,
            window_end=window_end,
            window_label=window_label,
            resolved_lab_name=resolved_lab_name,
            requested_metrics=requested_metrics,
            max_chart_lookback_points=max_chart_lookback_points,
        ),
        lambda: _handle_comparison_multi(
            question=question,
            intent=intent,
            requested_metrics=requested_metrics,
            window_start=window_start,
            window_end=window_end,
            window_label=window_label,
            resolved_lab_name=resolved_lab_name,
        ),
        lambda: _handle_baseline_reference_comparison(
            question=question,
            intent=intent,
            metric_alias=metric_alias,
            unit=unit,
            window_start=window_start,
            window_end=window_end,
            window_label=window_label,
            resolved_lab_name=resolved_lab_name,
        ),
        lambda: _handle_aggregation_multi(
            question=question,
            intent=intent,
            requested_metrics=requested_metrics,
            compared_spaces=compared_spaces,
            window_start=window_start,
            window_end=window_end,
            window_label=window_label,
            resolved_lab_name=resolved_lab_name,
        ),
        lambda: _handle_point_lookup(
            question=question,
            intent=intent,
            metric_alias=metric_alias,
            unit=unit,
            requested_metrics=requested_metrics,
            window_start=window_start,
            window_end=window_end,
            window_label=window_label,
            resolved_lab_name=resolved_lab_name,
            max_chart_lookback_points=max_chart_lookback_points,
        ),
        lambda: _handle_anomaly(
            intent=intent,
            metric_alias=metric_alias,
            unit=unit,
            window_start=window_start,
            window_end=window_end,
            window_label=window_label,
            resolved_lab_name=resolved_lab_name,
            max_chart_lookback_points=max_chart_lookback_points,
        ),
        lambda: _handle_comparison(
            question=question,
            intent=intent,
            metric_alias=metric_alias,
            unit=unit,
            window_start=window_start,
            window_end=window_end,
            window_label=window_label,
            resolved_lab_name=resolved_lab_name,
        ),
        lambda: _handle_forecast(
            question=question,
            intent=intent,
            metric_alias=metric_alias,
            unit=unit,
            window_start=window_start,
            window_end=window_end,
            window_label=window_label,
            resolved_lab_name=resolved_lab_name,
            max_chart_lookback_points=max_chart_lookback_points,
        ),
    ]

    selected: Optional[Dict[str, Any]] = None
    for handle in handlers:
        candidate = handle()
        if candidate:
            selected = candidate
            break
    if not selected:
        selected = _handle_default(
            question=question,
            intent=intent,
            metric_alias=metric_alias,
            unit=unit,
            window_start=window_start,
            window_end=window_end,
            window_label=window_label,
            resolved_lab_name=resolved_lab_name,
            compared_spaces=compared_spaces,
            max_chart_lookback_points=max_chart_lookback_points,
        )

    result.update(selected)
    result.setdefault("window_start", window_start)
    result.setdefault("window_end", window_end)
    result.setdefault("window_label", window_label)
    result.setdefault("compared_spaces", list(compared_spaces))
    result.setdefault("forecast_data", None)
    result.setdefault("correlation_data", None)
    return result
