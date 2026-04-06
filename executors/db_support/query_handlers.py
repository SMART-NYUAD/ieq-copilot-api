"""Intent-specific DB query handlers extracted from db_query_executor."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    from query_routing.intent_classifier import IntentType
except ImportError:
    from ...query_routing.intent_classifier import IntentType

try:
    from executors.db_support import charts as db_charts
    from executors.db_support import query_parsing as db_parsing
    from executors.db_support import response_helpers as db_helpers
    from executors.db_support.response_helpers import is_diagnostic_query_text
except ImportError:
    from . import charts as db_charts
    from . import query_parsing as db_parsing
    from . import response_helpers as db_helpers
    from .response_helpers import is_diagnostic_query_text


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


def _requested_metrics(
    question: str,
    explicit_metrics: List[str],
    hinted_metrics: List[str],
    intent: IntentType,
) -> List[str]:
    metrics = list(explicit_metrics) + [m for m in hinted_metrics if m not in explicit_metrics]
    if explicit_metrics:
        # For comparison-style air-quality asks (for example: "compare CO2 levels ..."),
        # expand to a core multi-metric pack so responses include PM2.5/TVOC context.
        q = str(question or "").lower()
        explicit_air_metric = any(m in {"co2", "pm25", "tvoc"} for m in explicit_metrics)
        if not (
            is_diagnostic_query_text(question)
            or (
            intent == IntentType.COMPARISON_DB
            and explicit_air_metric
            and any(token in q for token in ("compare", "vs", "versus"))
            )
        ):
            return metrics
    if is_diagnostic_query_text(question):
        required_pack = ["co2", "pm25", "tvoc", "humidity", "temperature", "ieq", "sound", "light"]
        missing = [m for m in required_pack if m not in metrics]
        return metrics + missing
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
    if not (is_air_quality_query or is_comfort_assessment_query):
        return metrics
    required_pack = (
        ["ieq", "temperature", "humidity", "co2", "pm25", "tvoc"]
        if is_comfort_assessment_query
        else ["co2", "pm25", "tvoc", "humidity", "ieq"]
    )
    missing_core_metrics = [m for m in required_pack if m not in metrics]
    return metrics + missing_core_metrics


def _handle_diagnostic(
    *,
    cur: Any,
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
    metric_columns = [
        (metric, db_parsing.CANONICAL_METRIC_COLUMN_MAP.get(metric))
        for metric in core_metrics
        if db_parsing.CANONICAL_METRIC_COLUMN_MAP.get(metric)
    ]
    if not metric_columns:
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
    select_metrics_sql = ", ".join([f"AVG({column}) AS {metric}" for metric, column in metric_columns])
    sql = f"""
        SELECT bucket, {select_metrics_sql}
        FROM lab_ieq_final
        WHERE bucket >= %s
          AND bucket < %s
          AND (%s IS NULL OR lab_space = %s)
        GROUP BY bucket
        ORDER BY bucket ASC
        LIMIT 2000
    """
    cur.execute(sql, (window_start, window_end, resolved_lab_name, resolved_lab_name))
    rows = [dict(row) for row in cur.fetchall()]
    if not rows:
        return {
            "operation_type": "diagnostic",
            "rows": [],
            "fallback_answer": f"I couldn't find IEQ diagnostic readings for {window_label}.",
            "chart_payload": db_charts.empty_chart(),
            "forecast_data": None,
            "correlation_data": None,
            "metric_alias": "ieq",
            "metrics_used": [m for m, _ in metric_columns],
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
        "metrics_used": [m for m, _ in metric_columns],
        "window_start": window_start,
        "window_end": window_end,
        "window_label": window_label,
        "compared_spaces": [],
    }


def _handle_correlation(
    *,
    cur: Any,
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
    column_x = db_parsing.CANONICAL_METRIC_COLUMN_MAP.get(metric_x)
    column_y = db_parsing.CANONICAL_METRIC_COLUMN_MAP.get(metric_y)
    if not column_x or not column_y:
        return {
            "operation_type": "correlation",
            "rows": [],
            "fallback_answer": f"I couldn't map requested metrics for correlation: {metric_x} vs {metric_y}.",
            "chart_payload": db_charts.empty_chart(),
            "correlation_data": None,
            "metric_alias": f"{metric_x}_vs_{metric_y}",
            "metrics_used": [metric_x, metric_y],
        }

    sql = f"""
        SELECT bucket, {column_x} AS x_value, {column_y} AS y_value
        FROM lab_ieq_final
        WHERE bucket >= %s
          AND bucket < %s
          AND (%s IS NULL OR lab_space = %s)
          AND {column_x} IS NOT NULL
          AND {column_y} IS NOT NULL
        ORDER BY bucket ASC
        LIMIT 5000
    """
    cur.execute(sql, (window_start, window_end, resolved_lab_name, resolved_lab_name))
    rows = [dict(row) for row in cur.fetchall()]
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
    cur: Any,
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
    compare_metrics = requested_metrics[:4]
    metric_columns = [
        (metric, db_parsing.CANONICAL_METRIC_COLUMN_MAP.get(metric))
        for metric in compare_metrics
        if db_parsing.CANONICAL_METRIC_COLUMN_MAP.get(metric)
    ]
    if len(metric_columns) < 1:
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
    select_metrics_sql = ", ".join([f"AVG({column}) AS {metric}" for metric, column in metric_columns])
    metric_names = [m for m, _ in metric_columns]
    if len(compared_spaces) < 2:
        if resolved_lab_name:
            sql = f"""
                SELECT lab_space, {select_metrics_sql}
                FROM lab_ieq_final
                WHERE lab_space = %s
                  AND bucket >= %s
                  AND bucket < %s
                GROUP BY lab_space
                LIMIT 1
            """
            cur.execute(sql, (resolved_lab_name, window_start, window_end))
            row = cur.fetchone()
            rows = [dict(row)] if row else []
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
    if len(compared_spaces) >= 2:
        sql = f"""
            SELECT lab_space, {select_metrics_sql}
            FROM lab_ieq_final
            WHERE lab_space = ANY(%s)
              AND bucket >= %s
              AND bucket < %s
            GROUP BY lab_space
            ORDER BY lab_space ASC
            LIMIT 2
        """
        cur.execute(sql, (compared_spaces, window_start, window_end))
    rows = [dict(row) for row in cur.fetchall()]
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
        "compared_spaces": compared_spaces,
    }


def _handle_baseline_reference_comparison(
    *,
    cur: Any,
    question: str,
    intent: IntentType,
    metric_alias: str,
    metric_column: str,
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
    baseline_end = window_start
    sql = f"""
        SELECT
            AVG(CASE WHEN bucket >= %s AND bucket < %s THEN {metric_column} END) AS current_avg,
            AVG(CASE WHEN bucket >= %s AND bucket < %s THEN {metric_column} END) AS baseline_avg,
            STDDEV_POP(CASE WHEN bucket >= %s AND bucket < %s THEN {metric_column} END) AS baseline_stddev,
            COUNT(CASE WHEN bucket >= %s AND bucket < %s THEN 1 END) AS current_count,
            COUNT(CASE WHEN bucket >= %s AND bucket < %s THEN 1 END) AS baseline_count
        FROM lab_ieq_final
        WHERE lab_space = %s
          AND bucket >= %s
          AND bucket < %s
          AND {metric_column} IS NOT NULL
    """
    cur.execute(
        sql,
        (
            window_start,
            window_end,
            baseline_start,
            baseline_end,
            baseline_start,
            baseline_end,
            window_start,
            window_end,
            baseline_start,
            baseline_end,
            resolved_lab_name,
            baseline_start,
            window_end,
        ),
    )
    row = dict(cur.fetchone() or {})
    current_avg = row.get("current_avg")
    baseline_avg = row.get("baseline_avg")
    if current_avg is None or baseline_avg is None:
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
    current_value = float(current_avg)
    baseline_value = float(baseline_avg)
    delta = current_value - baseline_value
    pct = (delta / baseline_value * 100.0) if baseline_value != 0 else None
    rows = [
        {
            "lab_space": resolved_lab_name,
            "current_avg": current_value,
            "baseline_avg": baseline_value,
            "delta_value": delta,
            "delta_percent": pct,
            "baseline_stddev": row.get("baseline_stddev"),
            "current_count": int(row.get("current_count") or 0),
            "baseline_count": int(row.get("baseline_count") or 0),
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
    cur: Any,
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
    if db_helpers.is_comfort_assessment_query_text(question):
        selected_metrics = requested_metrics[:6]
    elif db_helpers.is_air_quality_query_text(question):
        selected_metrics = requested_metrics[:5]
    else:
        selected_metrics = requested_metrics[:4]
    metric_columns = [
        (metric, db_parsing.CANONICAL_METRIC_COLUMN_MAP.get(metric))
        for metric in selected_metrics
        if db_parsing.CANONICAL_METRIC_COLUMN_MAP.get(metric)
    ]
    if len(metric_columns) < 1:
        return {
            "operation_type": "aggregation_multi_metric",
            "rows": [],
            "fallback_answer": "I couldn't map requested metrics for analysis.",
            "chart_payload": db_charts.empty_chart(),
            "metrics_used": [],
        }
    select_metrics_sql = ", ".join(
        [
            (
                f"AVG({column}) AS {metric}, "
                f"MIN({column}) AS {metric}_min, "
                f"MAX({column}) AS {metric}_max, "
                f"STDDEV_POP({column}) AS {metric}_stddev"
            )
            for metric, column in metric_columns
        ]
    )
    sql = f"""
        SELECT lab_space, {select_metrics_sql}, COUNT(*) AS reading_count
        FROM lab_ieq_final
        WHERE bucket >= %s
          AND bucket < %s
          AND (%s IS NULL OR lab_space = %s)
        GROUP BY lab_space
        ORDER BY reading_count DESC
        LIMIT 1
    """
    cur.execute(sql, (window_start, window_end, resolved_lab_name, resolved_lab_name))
    row = cur.fetchone()
    rows = [dict(row)] if row else []
    metric_names = [m for m, _ in metric_columns]
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
    cur: Any,
    question: str,
    intent: IntentType,
    metric_alias: str,
    metric_column: str,
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
    is_multi = (
        db_helpers.is_air_quality_query_text(question)
        or db_helpers.is_comfort_assessment_query_text(question)
        or db_helpers.is_issue_triage_query_text(question)
    )
    if is_multi:
        selected_metrics = requested_metrics[:6] if db_helpers.is_comfort_assessment_query_text(question) else requested_metrics[:5]
        metric_columns = [
            (metric, db_parsing.CANONICAL_METRIC_COLUMN_MAP.get(metric))
            for metric in selected_metrics
            if db_parsing.CANONICAL_METRIC_COLUMN_MAP.get(metric)
        ]
        if len(metric_columns) < 1:
            return {
                "operation_type": "point_lookup_multi_metric",
                "rows": [],
                "fallback_answer": "I couldn't map requested air-quality metrics for point lookup.",
                "chart_payload": db_charts.empty_chart(),
                "metrics_used": [],
            }
        select_metrics_sql = ", ".join([f"{column} AS {metric}" for metric, column in metric_columns])
        sql = f"""
            SELECT lab_space, bucket, {select_metrics_sql}
            FROM lab_ieq_final
            WHERE (%s IS NULL OR lab_space = %s)
              AND bucket >= %s
              AND bucket < %s
            ORDER BY bucket DESC
            LIMIT 1
        """
        cur.execute(sql, (resolved_lab_name, resolved_lab_name, window_start, window_end))
        row = cur.fetchone()
        rows = [dict(row)] if row else []
        metric_names = [m for m, _ in metric_columns]
        return {
            "operation_type": "point_lookup_multi_metric",
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

    sql = f"""
        SELECT lab_space, bucket, {metric_column} AS value
        FROM lab_ieq_final
        WHERE (%s IS NULL OR lab_space = %s)
          AND bucket >= %s
          AND bucket < %s
        ORDER BY bucket DESC
        LIMIT 1
    """
    cur.execute(sql, (resolved_lab_name, resolved_lab_name, window_start, window_end))
    row = cur.fetchone()
    rows = [dict(row)] if row else []
    trend_sql = f"""
        SELECT bucket, {metric_column} AS value
        FROM lab_ieq_final
        WHERE (%s IS NULL OR lab_space = %s)
          AND bucket >= %s
          AND bucket < %s
        ORDER BY bucket ASC
        LIMIT 500
    """
    cur.execute(trend_sql, (resolved_lab_name, resolved_lab_name, window_start, window_end))
    trend_rows = [dict(item) for item in cur.fetchall()]
    series_name = resolved_lab_name or (rows[0].get("lab_space") if rows else "selected_scope")
    return {
        "operation_type": "point_lookup",
        "rows": rows,
        "fallback_answer": db_helpers.build_point_lookup_answer(metric_alias, row or {}, window_label),
        "chart_payload": db_charts.build_line_chart(
            metric_alias=metric_alias,
            unit=unit,
            window_label=window_label,
            series_rows=trend_rows,
            series_name=str(series_name),
            lookback_points=max_chart_lookback_points,
        ),
        "metrics_used": [metric_alias],
    }


def _handle_anomaly(
    *,
    cur: Any,
    intent: IntentType,
    metric_alias: str,
    metric_column: str,
    unit: str,
    window_start: datetime,
    window_end: datetime,
    window_label: str,
    resolved_lab_name: Optional[str],
    max_chart_lookback_points: int,
) -> Optional[Dict[str, Any]]:
    if intent != IntentType.ANOMALY_ANALYSIS_DB:
        return None
    sql = f"""
        SELECT lab_space, bucket, {metric_column} AS value
        FROM lab_ieq_final
        WHERE (%s IS NULL OR lab_space = %s)
          AND bucket >= %s
          AND bucket < %s
          AND {metric_column} IS NOT NULL
        ORDER BY bucket ASC
        LIMIT 5000
    """
    cur.execute(sql, (resolved_lab_name, resolved_lab_name, window_start, window_end))
    rows = [dict(item) for item in cur.fetchall()]
    anomalies = db_helpers.detect_anomaly_points(rows)
    series_name = resolved_lab_name or (rows[0].get("lab_space") if rows else "selected_scope")
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
    cur: Any,
    question: str,
    intent: IntentType,
    metric_alias: str,
    metric_column: str,
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
    if len(compared_spaces) >= 2:
        sql = f"""
            SELECT lab_space, AVG({metric_column}) AS avg_value
            FROM lab_ieq_final
            WHERE lab_space = ANY(%s)
              AND bucket >= %s
              AND bucket < %s
            GROUP BY lab_space
            ORDER BY avg_value DESC
            LIMIT 2
        """
        cur.execute(sql, (compared_spaces, window_start, window_end))
    rows = [dict(row) for row in cur.fetchall()]
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
        "compared_spaces": compared_spaces,
    }


def _handle_forecast(
    *,
    cur: Any,
    question: str,
    intent: IntentType,
    metric_alias: str,
    metric_column: str,
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
    sql = f"""
        SELECT bucket, value
        FROM (
            SELECT bucket, AVG({metric_column}) AS value
            FROM lab_ieq_final
            WHERE bucket >= %s
              AND bucket < %s
              AND (%s IS NULL OR lab_space = %s)
            GROUP BY bucket
            ORDER BY bucket DESC
            LIMIT 1000
        ) recent
        ORDER BY bucket ASC
    """
    cur.execute(sql, (forecast_window_start, forecast_window_end, resolved_lab_name, resolved_lab_name))
    model_rows = [dict(row) for row in cur.fetchall()]
    forecast_data = db_helpers.build_forecast_from_rows(model_rows, horizon_hours=horizon_hours)
    chart_rows = model_rows
    if forecast_window_start != window_start or forecast_window_end != window_end:
        # Keep chart/history display aligned to the requested window, while the model
        # can still use a broader history window internally.
        cur.execute(sql, (window_start, window_end, resolved_lab_name, resolved_lab_name))
        requested_rows = [dict(row) for row in cur.fetchall()]
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
    cur: Any,
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
    max_chart_lookback_points: int,
) -> Dict[str, Any]:
    if intent == IntentType.AGGREGATION_DB and len(compared_spaces) >= 2:
        sql = f"""
            SELECT lab_space, AVG({metric_column}) AS avg_value
            FROM lab_ieq_final
            WHERE lab_space = ANY(%s)
              AND bucket >= %s
              AND bucket < %s
            GROUP BY lab_space
            ORDER BY avg_value DESC
            LIMIT 2
        """
        cur.execute(sql, (compared_spaces, window_start, window_end))
        rows = [dict(row) for row in cur.fetchall()]
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
            "compared_spaces": compared_spaces,
        }

    if db_parsing.wants_time_series(question):
        sql = f"""
            SELECT lab_space, bucket, {metric_column} AS value
            FROM lab_ieq_final
            WHERE bucket >= %s
              AND bucket < %s
              AND (%s IS NULL OR lab_space = %s)
            ORDER BY bucket ASC
            LIMIT 1000
        """
        cur.execute(sql, (window_start, window_end, resolved_lab_name, resolved_lab_name))
        rows = [dict(row) for row in cur.fetchall()]
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

    sql = f"""
        SELECT lab_space,
               AVG({metric_column}) AS avg_value,
               MIN({metric_column}) AS min_value,
               MAX({metric_column}) AS max_value,
               AVG(contri_thermal) AS contri_thermal,
               AVG(contri_light) AS contri_light,
               AVG(contri_air) AS contri_air,
               AVG(contri_acoustic) AS contri_acoustic,
               AVG(contri_acoustic) AS contri_acustic,
               COUNT(*) AS reading_count
        FROM lab_ieq_final
        WHERE bucket >= %s
          AND bucket < %s
          AND (%s IS NULL OR lab_space = %s)
        GROUP BY lab_space
        ORDER BY avg_value DESC
        LIMIT 10
    """
    cur.execute(sql, (window_start, window_end, resolved_lab_name, resolved_lab_name))
    rows = [dict(row) for row in cur.fetchall()]
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
    cur: Any,
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
            cur=cur,
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
            cur=cur,
            question=question,
            window_start=window_start,
            window_end=window_end,
            window_label=window_label,
            resolved_lab_name=resolved_lab_name,
            requested_metrics=requested_metrics,
            max_chart_lookback_points=max_chart_lookback_points,
        ),
        lambda: _handle_comparison_multi(
            cur=cur,
            question=question,
            intent=intent,
            requested_metrics=requested_metrics,
            window_start=window_start,
            window_end=window_end,
            window_label=window_label,
            resolved_lab_name=resolved_lab_name,
        ),
        lambda: _handle_baseline_reference_comparison(
            cur=cur,
            question=question,
            intent=intent,
            metric_alias=metric_alias,
            metric_column=metric_column,
            unit=unit,
            window_start=window_start,
            window_end=window_end,
            window_label=window_label,
            resolved_lab_name=resolved_lab_name,
        ),
        lambda: _handle_aggregation_multi(
            cur=cur,
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
            cur=cur,
            question=question,
            intent=intent,
            metric_alias=metric_alias,
            metric_column=metric_column,
            unit=unit,
            requested_metrics=requested_metrics,
            window_start=window_start,
            window_end=window_end,
            window_label=window_label,
            resolved_lab_name=resolved_lab_name,
            max_chart_lookback_points=max_chart_lookback_points,
        ),
        lambda: _handle_anomaly(
            cur=cur,
            intent=intent,
            metric_alias=metric_alias,
            metric_column=metric_column,
            unit=unit,
            window_start=window_start,
            window_end=window_end,
            window_label=window_label,
            resolved_lab_name=resolved_lab_name,
            max_chart_lookback_points=max_chart_lookback_points,
        ),
        lambda: _handle_comparison(
            cur=cur,
            question=question,
            intent=intent,
            metric_alias=metric_alias,
            metric_column=metric_column,
            unit=unit,
            window_start=window_start,
            window_end=window_end,
            window_label=window_label,
            resolved_lab_name=resolved_lab_name,
        ),
        lambda: _handle_forecast(
            cur=cur,
            question=question,
            intent=intent,
            metric_alias=metric_alias,
            metric_column=metric_column,
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
            cur=cur,
            question=question,
            intent=intent,
            metric_alias=metric_alias,
            metric_column=metric_column,
            unit=unit,
            window_start=window_start,
            window_end=window_end,
            window_label=window_label,
            resolved_lab_name=resolved_lab_name,
            compared_spaces=compared_spaces,
            max_chart_lookback_points=max_chart_lookback_points,
        )

    result.update(selected)
    # Preserve original scope values when branch did not override them.
    result.setdefault("window_start", window_start)
    result.setdefault("window_end", window_end)
    result.setdefault("window_label", window_label)
    result.setdefault("compared_spaces", list(compared_spaces))
    result.setdefault("forecast_data", None)
    result.setdefault("correlation_data", None)
    return result
