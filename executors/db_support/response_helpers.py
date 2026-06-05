"""Reusable DB-executor response and payload helper functions."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
try:
    from executors.db_support import time_windows as db_time_windows
    from executors import metric_registry
except ImportError:
    from . import time_windows as db_time_windows
    from .. import metric_registry

try:
    from prompting.db_prompts import (
        DB_TOOL_RESPONSE_DIRECTIVE,
        DB_TOOL_RESPONSE_DIRECTIVE_POINT_LOOKUP,
        DB_TOOL_RESPONSE_DIRECTIVE_AIR_QUALITY_POINT_LOOKUP,
        DB_TOOL_RESPONSE_DIRECTIVE_COMPARISON,
        DB_TOOL_RESPONSE_DIRECTIVE_ANOMALY,
        DB_TOOL_RESPONSE_DIRECTIVE_DIAGNOSTIC,
        CITATION_FORMAT_INSTRUCTION,
        FRIENDLY_TONE_INSTRUCTION,
    )
except ImportError:
    from ...prompting.db_prompts import (
        DB_TOOL_RESPONSE_DIRECTIVE,
        DB_TOOL_RESPONSE_DIRECTIVE_POINT_LOOKUP,
        DB_TOOL_RESPONSE_DIRECTIVE_AIR_QUALITY_POINT_LOOKUP,
        DB_TOOL_RESPONSE_DIRECTIVE_COMPARISON,
        DB_TOOL_RESPONSE_DIRECTIVE_ANOMALY,
        DB_TOOL_RESPONSE_DIRECTIVE_DIAGNOSTIC,
        CITATION_FORMAT_INSTRUCTION,
        FRIENDLY_TONE_INSTRUCTION,
    )


def to_target_timezone(dt: datetime) -> datetime:
    return db_time_windows.to_target_timezone(dt)


def serialize_datetime_iso(dt: datetime) -> str:
    return db_time_windows.serialize_datetime_iso(dt)


def serialize_timestamp_value(value: Any) -> Any:
    return db_time_windows.serialize_timestamp_value(value)


def is_air_quality_query_text(question: str) -> bool:
    q = (question or "").lower()
    if is_issue_triage_query_text(q):
        return True
    if any(hint in q for hint in ("air quality", "indoor air quality", "ieq")):
        return True
    if re.search(r"\bhow\s+(?:is|was)\s+the\s+air\b", q):
        return True
    if re.search(r"\bthe\s+air\b", q) and ("_lab" in q or re.search(r"\b[a-z0-9]+\s+lab\b", q) is not None):
        return True
    return False


def is_issue_triage_query_text(question: str) -> bool:
    q = (question or "").lower()
    issue_hints = (
        "issue",
        "issues",
        "problem",
        "problems",
        "anything wrong",
        "any issue",
        "any issues",
        "wrong",
    )
    currentness_hints = ("right now", "now", "current", "currently", "latest", "today", "at this moment")
    has_issue_hint = any(hint in q for hint in issue_hints)
    has_currentness_hint = any(hint in q for hint in currentness_hints)
    has_lab_reference = ("_lab" in q) or (re.search(r"\b[a-z0-9]+\s+lab\b", q) is not None)
    return has_issue_hint and (has_currentness_hint or has_lab_reference)


def is_comfort_assessment_query_text(question: str) -> bool:
    q = (question or "").lower()
    comfort_hints = (
        "comfortable",
        "comfort",
        "too hot",
        "too cold",
        "stuffy",
        "dry",
        "humid",
    )
    return any(hint in q for hint in comfort_hints)


def is_diagnostic_query_text(question: str) -> bool:
    q = (question or "").lower()
    diagnostic_hints = (
        "driving",
        "causing",
        "cause of",
        "responsible for",
        "behind",
        "reason for",
        "why is",
        "why was",
        "what is affecting",
        "what affected",
        "contributing to",
        "root cause",
        "which metric",
        "what metric",
        "which factor",
        "what factor",
        "poor ieq",
        "low ieq",
        "bad ieq",
        "dropped",
        "decline",
        "dip in ieq",
        "drop in ieq",
        "drop in index",
        "what caused",
        "what's causing",
        "whats causing",
    )
    return any(hint in q for hint in diagnostic_hints)


def db_response_directive(intent: Any, question: str = "") -> str:
    if is_diagnostic_query_text(question):
        return DB_TOOL_RESPONSE_DIRECTIVE_DIAGNOSTIC
    intent_value = getattr(intent, "value", str(intent))
    if intent_value in {"point_lookup_db", "current_status_db"}:
        if (
            is_air_quality_query_text(question)
            or is_comfort_assessment_query_text(question)
            or is_issue_triage_query_text(question)
        ):
            return DB_TOOL_RESPONSE_DIRECTIVE_AIR_QUALITY_POINT_LOOKUP
        return DB_TOOL_RESPONSE_DIRECTIVE_POINT_LOOKUP
    if intent_value == "comparison_db":
        return DB_TOOL_RESPONSE_DIRECTIVE_COMPARISON
    if intent_value == "anomaly_analysis_db":
        return DB_TOOL_RESPONSE_DIRECTIVE_ANOMALY
    return DB_TOOL_RESPONSE_DIRECTIVE


def correlate_metrics_with_ieq(
    rows: List[Dict[str, Any]],
    metrics: List[str],
) -> Dict[str, Any]:
    """
    Compute Pearson correlation between each metric and ieq column.
    """
    if not rows:
        return {
            "correlations": {},
            "top_culprits": [],
            "top_culprit_scores": {},
            "dip_count": 0,
            "ieq_threshold_used": 0.0,
            "dip_periods": [],
        }
    df = pd.DataFrame(rows).copy()
    if "ieq" not in df.columns:
        return {
            "correlations": {},
            "top_culprits": [],
            "top_culprit_scores": {},
            "dip_count": 0,
            "ieq_threshold_used": 0.0,
            "dip_periods": [],
        }
    df["ieq"] = pd.to_numeric(df["ieq"], errors="coerce")
    df = df.dropna(subset=["ieq"])
    correlations: Dict[str, Optional[float]] = {}
    metric_scores: List[Tuple[str, float]] = []
    for metric in metrics:
        if metric == "ieq" or metric not in df.columns:
            continue
        metric_df = df[["ieq", metric]].copy()
        metric_df[metric] = pd.to_numeric(metric_df[metric], errors="coerce")
        metric_df = metric_df.dropna(subset=["ieq", metric])
        if len(metric_df) < 8:
            correlations[metric] = None
            continue
        corr_value = metric_df["ieq"].corr(metric_df[metric])
        if corr_value is None or pd.isna(corr_value):
            correlations[metric] = None
            continue
        rounded = round(float(corr_value), 3)
        correlations[metric] = rounded
        metric_scores.append((metric, rounded))
    metric_scores.sort(key=lambda item: item[1])
    top_culprits = [metric for metric, _ in metric_scores]
    top_culprit_scores = {metric: score for metric, score in metric_scores}

    ieq_min = float(df["ieq"].min())
    ieq_max = float(df["ieq"].max())
    ieq_mean = float(df["ieq"].mean())
    ieq_threshold = ieq_mean - 0.25 * (ieq_max - ieq_min)
    dip_df = df[df["ieq"] < ieq_threshold].copy()
    dip_count = int(len(dip_df))
    dip_periods: List[Dict[str, Any]] = []
    if dip_count > 0:
        columns = ["bucket", "ieq"] + [m for m in metrics if m != "ieq" and m in dip_df.columns]
        for _, row in dip_df.head(10)[columns].iterrows():
            period: Dict[str, Any] = {}
            for column in columns:
                value = row.get(column)
                if column == "bucket":
                    period[column] = db_time_windows.serialize_timestamp_value(value)
                else:
                    period[column] = None if pd.isna(value) else float(value)
            dip_periods.append(period)
    return {
        "correlations": correlations,
        "top_culprits": top_culprits,
        "top_culprit_scores": top_culprit_scores,
        "dip_count": dip_count,
        "ieq_threshold_used": round(float(ieq_threshold), 3),
        "dip_periods": dip_periods,
    }


def build_diagnostic_answer(
    rows: List[Dict[str, Any]],
    correlation_analysis: Dict[str, Any],
    window_label: str,
    lab_name: Optional[str],
) -> str:
    """
    Build deterministic fallback answer for diagnostic queries.
    Must include: scope, reading count, dip count, top culprit metric,
    correlation value, correlation strength label, direction label,
    secondary culprits if present.
    """
    scope = lab_name or (rows[0].get("lab_space") if rows else "selected scope")
    reading_count = len(rows)
    dip_count = int(correlation_analysis.get("dip_count") or 0)
    top_culprits = list(correlation_analysis.get("top_culprits") or [])
    scores = dict(correlation_analysis.get("top_culprit_scores") or {})
    if not rows:
        return f"I couldn't find IEQ and related metric readings for {scope} in {window_label}."
    if not top_culprits:
        return (
            f"Diagnostic scan for {scope} in {window_label} used {reading_count} readings with {dip_count} IEQ dips, "
            "but there were not enough paired datapoints to compute reliable metric correlations."
        )
    top_metric = top_culprits[0]
    top_score = scores.get(top_metric)
    abs_corr = abs(float(top_score or 0.0))
    if abs_corr >= 0.8:
        strength = "strong"
    elif abs_corr >= 0.5:
        strength = "moderate"
    elif abs_corr >= 0.3:
        strength = "weak"
    else:
        strength = "very weak"
    direction = "inverse" if float(top_score or 0.0) < 0 else "positive"
    secondary = [m for m in top_culprits[1:3] if m in scores]
    secondary_text = ""
    if secondary:
        secondary_text = " Secondary contributors: " + ", ".join(
            f"{metric} (r={float(scores.get(metric) or 0.0):.3f})" for metric in secondary
        ) + "."
    return (
        f"Diagnostic analysis for {scope} in {window_label} used {reading_count} readings and found {dip_count} IEQ dips. "
        f"Top likely driver is {top_metric} with r={float(top_score or 0.0):.3f} "
        f"({strength} {direction} relationship with IEQ).{secondary_text}"
    )


def build_point_lookup_answer(metric_alias: str, row: Dict, window_label: str) -> str:
    if not row:
        return f"I couldn't find {metric_alias} datapoints for {window_label} in the selected scope."
    return (
        f"Latest {metric_alias} for {row.get('lab_space', 'unknown')} is "
        f"{row.get('value')} at {row.get('bucket')} ({window_label})."
    )


def build_aggregation_answer(metric_alias: str, rows: List[Dict], window_label: str) -> str:
    if not rows:
        return f"I couldn't find {metric_alias} data for {window_label}."
    top = rows[0]
    return (
        f"Computed average {metric_alias} over {window_label} across {len(rows)} spaces. "
        f"Highest average is {top.get('lab_space')} with {top.get('avg_value')}."
    )


def build_timeseries_answer(metric_alias: str, rows: List[Dict], window_label: str) -> str:
    if not rows:
        return f"I couldn't find {metric_alias} values for {window_label}."
    first = rows[0]
    last = rows[-1]
    min_value = min(float(row.get("value") or 0.0) for row in rows if row.get("value") is not None)
    max_value = max(float(row.get("value") or 0.0) for row in rows if row.get("value") is not None)
    return (
        f"Found {len(rows)} {metric_alias} values for {first.get('lab_space', 'selected scope')} in {window_label}. "
        f"Range was {min_value:.2f} to {max_value:.2f}, from {first.get('bucket')} to {last.get('bucket')}."
    )


def normalize_series_rows(rows: List[Dict[str, Any]], metric_alias: str) -> List[Dict[str, Any]]:
    """Extract chronological bucket/value points from handler rows."""
    points: List[Dict[str, Any]] = []
    metric_key = str(metric_alias or "").strip().lower()
    for row in rows or []:
        row_metric = str(row.get("metric") or "").strip().lower()
        if metric_key and row_metric and row_metric != metric_key:
            continue
        bucket = row.get("bucket")
        value = row.get("value")
        if value is None and metric_key and row.get(metric_key) is not None:
            value = row.get(metric_key)
        if bucket is None or value is None:
            continue
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue
        points.append(
            {
                "at": bucket,
                "value": numeric_value,
                "lab_space": row.get("lab_space"),
            }
        )
    points.sort(key=lambda point: str(point.get("at") or ""))
    return points


def _trend_direction(percent_change: Optional[float]) -> str:
    if percent_change is None:
        return "unknown"
    if percent_change > 2.0:
        return "rising"
    if percent_change < -2.0:
        return "falling"
    return "stable"


def build_backend_semantic_state(analysis: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not analysis or not analysis.get("window_stats"):
        return None
    return {
        key: analysis[key]
        for key in (
            "window_stats",
            "change_analysis",
            "notable_events",
            "data_granularity",
            "granularity_note",
        )
        if key in analysis
    }


_MULTI_METRIC_OPERATIONS = frozenset(
    {
        "comparison",
        "comparison_multi_metric",
        "temporal_comparison",
        "baseline_reference_comparison",
        "anomaly_multi",
    }
)


def _should_attach_authoritative_bounds(
    *,
    operation_type: Optional[str],
    metrics_used: Optional[List[str]],
) -> bool:
    if str(operation_type or "").strip().lower() in _MULTI_METRIC_OPERATIONS:
        return False
    normalized = [str(m or "").strip().lower() for m in list(metrics_used or []) if m]
    return len(normalized) <= 1


def _build_authoritative_bounds_block(
    *,
    metric: str,
    unit: str,
    window_stats: Dict[str, Any],
    notable_events: List[Any],
    display_start: str = "",
    display_end: str = "",
    window_label: str = "",
) -> Dict[str, Any]:
    min_value = float(window_stats["min"])
    max_value = float(window_stats["max"])
    mean_raw = window_stats.get("mean")
    mean_value = float(mean_raw) if mean_raw is not None else None
    unit_display = unit or metric_registry.metric_unit(metric) or "value"
    rules = [
        f"These bounds apply only to {metric} ({unit_display}), not other metrics in the question.",
        "Every value you cite for this metric must fall within allowed_value_min and allowed_value_max.",
        "Use peak_value/peak_at and trough_value/trough_at for extrema; do not invent other peaks or times.",
        "If notable_events_count is 0, do not describe spikes, surges, or brief peaks.",
        "Values are hourly averages; do not claim sub-hour spikes.",
    ]
    if metric == "ieq":
        rules.append(
            "IEQ is a unitless index (0–100) where higher = better, lower = worse. "
            "Score bands: >75 high quality, 51–75 medium, 26–50 moderate, ≤25 low quality. "
            "Never append % or %RH to IEQ values. "
            "A high ITC score means GOOD thermal comfort (comfortable), NOT hot/warm/stuffy. "
            "These bands apply to IEQ and all sub-indices (IAQ, ITC, IAC, IIL)."
        )
    elif metric == "humidity" or unit_display.lower() in {"%rh", "%"}:
        rules.append("Express humidity in %RH only.")
    elif metric == "co2" or unit_display.lower() == "ppm":
        rules.append("Express CO2 in ppm only.")
    block: Dict[str, Any] = {
        "metric": metric,
        "unit": unit_display,
        "allowed_value_min": round(min_value, 4),
        "allowed_value_max": round(max_value, 4),
        "window_mean": round(mean_value, 4) if mean_value is not None else None,
        "peak_value": round(max_value, 4),
        "peak_at": window_stats.get("max_at"),
        "trough_value": round(min_value, 4),
        "trough_at": window_stats.get("min_at"),
        "notable_events_count": len(notable_events),
        "rules": rules,
    }
    if display_start and display_end:
        block["analysis_window_display"] = f"{display_start} to {display_end}"
    elif window_label:
        block["analysis_window_label"] = window_label
    return block


def enrich_backend_semantic_state(
    analysis: Optional[Dict[str, Any]],
    *,
    operation_type: Optional[str] = None,
    metrics_used: Optional[List[str]] = None,
    display_start: str = "",
    display_end: str = "",
    window_label: str = "",
) -> Optional[Dict[str, Any]]:
    """Attach authoritative_bounds for single-metric time-series answers only."""
    state = build_backend_semantic_state(analysis)
    if not state or not _should_attach_authoritative_bounds(
        operation_type=operation_type,
        metrics_used=metrics_used,
    ):
        return state

    ts = (analysis or {}).get("time_series") or {}
    metric = str(ts.get("metric") or "").strip().lower()
    unit = str(ts.get("unit") or "").strip()
    if not metric:
        return state

    state["authoritative_bounds"] = _build_authoritative_bounds_block(
        metric=metric,
        unit=unit,
        window_stats=state["window_stats"],
        notable_events=list(state.get("notable_events") or []),
        display_start=display_start,
        display_end=display_end,
        window_label=window_label,
    )
    return state


def build_time_series_analysis(
    *,
    series_rows: List[Dict[str, Any]],
    metric_alias: str,
    unit: str,
    api_trend_pct: Optional[float] = None,
    aggregation_summary: Optional[Dict[str, Any]] = None,
    max_points: int = 48,
) -> Dict[str, Any]:
    """Derive spike/trend context for the LLM from hourly bucket rows."""
    points = normalize_series_rows(series_rows, metric_alias)
    analysis: Dict[str, Any] = {
        "data_granularity": "hourly_average",
        "granularity_note": (
            "Values are hourly averages from the sensor API. "
            "Sub-hour spikes may not be visible in this data."
        ),
    }
    if not points:
        return analysis

    values = [float(point["value"]) for point in points]
    point_count = len(values)
    mean_value = sum(values) / point_count
    min_value = min(values)
    max_value = max(values)
    min_idx = values.index(min_value)
    max_idx = values.index(max_value)

    window_stats: Dict[str, Any] = {
        "min": round(min_value, 4),
        "max": round(max_value, 4),
        "mean": round(mean_value, 4),
        "range": round(max_value - min_value, 4),
        "point_count": point_count,
        "min_at": points[min_idx]["at"],
        "max_at": points[max_idx]["at"],
    }
    if point_count > 1:
        window_stats["stddev"] = round(float(pd.Series(values).std(ddof=0)), 4)
    if aggregation_summary:
        for source_key, target_key in (
            ("min_value", "api_min_value"),
            ("max_value", "api_max_value"),
            ("avg_value", "api_avg_value"),
        ):
            if aggregation_summary.get(source_key) is not None:
                window_stats[target_key] = aggregation_summary.get(source_key)

    first_value = values[0]
    last_value = values[-1]
    absolute_change = last_value - first_value
    percent_change = (absolute_change / first_value * 100.0) if first_value else None
    change_analysis: Dict[str, Any] = {
        "first_value": round(first_value, 4),
        "last_value": round(last_value, 4),
        "absolute_change": round(absolute_change, 4),
        "percent_change": round(percent_change, 2) if percent_change is not None else None,
        "direction": _trend_direction(percent_change),
    }
    if api_trend_pct is not None:
        try:
            change_analysis["api_trend_pct"] = round(float(api_trend_pct), 2)
        except (TypeError, ValueError):
            pass

    if point_count >= 4:
        recent_values = values[-3:]
        prior_values = values[-6:-3] if point_count >= 6 else values[:-3]
        if prior_values:
            recent_avg = sum(recent_values) / len(recent_values)
            prior_avg = sum(prior_values) / len(prior_values)
            change_analysis["recent_3h_avg"] = round(recent_avg, 4)
            change_analysis["prior_3h_avg"] = round(prior_avg, 4)
            if prior_avg:
                change_analysis["recent_vs_prior_pct"] = round(
                    (recent_avg - prior_avg) / prior_avg * 100.0,
                    2,
                )

    if point_count >= 2:
        max_jump = 0.0
        max_jump_at: Optional[Any] = None
        for idx in range(1, point_count):
            jump = abs(values[idx] - values[idx - 1])
            if jump > max_jump:
                max_jump = jump
                max_jump_at = points[idx]["at"]
        change_analysis["max_hour_over_hour_jump"] = round(max_jump, 4)
        change_analysis["max_jump_at"] = max_jump_at

    anomaly_input = [
        {"lab_space": point.get("lab_space"), "bucket": point["at"], "value": point["value"]}
        for point in points
    ]
    anomalies = detect_anomaly_points(anomaly_input)
    notable_events: List[Dict[str, Any]] = []
    for anomaly in anomalies:
        try:
            value = float(anomaly.get("value"))
        except (TypeError, ValueError):
            continue
        vs_mean_pct = ((value - mean_value) / mean_value * 100.0) if mean_value else None
        notable_events.append(
            {
                "type": "spike" if value > mean_value else "dip",
                "at": anomaly.get("bucket"),
                "value": round(value, 4),
                "z_score": round(float(anomaly.get("score") or 0.0), 2),
                "vs_window_mean_pct": round(vs_mean_pct, 2) if vs_mean_pct is not None else None,
            }
        )

    compact_points = points[-max_points:]
    analysis.update(
        {
            "window_stats": window_stats,
            "change_analysis": change_analysis,
            "notable_events": notable_events,
            "time_series": {
                "metric": metric_alias,
                "unit": unit,
                "interval": "1h",
                "points": [{"at": point["at"], "value": round(point["value"], 4)} for point in compact_points],
            },
        }
    )
    return analysis


def detect_anomaly_points(rows: List[Dict[str, Any]], z_threshold: float = 2.5) -> List[Dict[str, Any]]:
    if len(rows) < 8:
        return []
    values = pd.Series([float(row["value"]) for row in rows if row.get("value") is not None], dtype="float64")
    if len(values) < 8:
        return []
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    anomaly_indices: List[int] = []
    anomaly_scores: Dict[int, float] = {}
    if std > 0:
        z_scores = ((values - mean) / std).abs()
        for idx, score in z_scores.items():
            if float(score) >= z_threshold:
                anomaly_indices.append(int(idx))
                anomaly_scores[int(idx)] = float(score)
    if not anomaly_indices:
        q1 = float(values.quantile(0.25))
        q3 = float(values.quantile(0.75))
        iqr = q3 - q1
        if iqr <= 0:
            return []
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        for idx, value in values.items():
            if float(value) < lower or float(value) > upper:
                anomaly_indices.append(int(idx))
                anomaly_scores[int(idx)] = 0.0
    output: List[Dict[str, Any]] = []
    for idx in sorted(set(anomaly_indices)):
        if idx < 0 or idx >= len(rows):
            continue
        row = rows[idx]
        output.append(
            {
                "lab_space": row.get("lab_space"),
                "bucket": row.get("bucket"),
                "value": row.get("value"),
                "score": anomaly_scores.get(idx, 0.0),
            }
        )
    return output


def build_anomaly_answer(
    metric_alias: str,
    rows: List[Dict[str, Any]],
    anomalies: List[Dict[str, Any]],
    window_label: str,
    lab_name: Optional[str],
) -> str:
    scope = lab_name or (rows[0].get("lab_space") if rows else "selected scope")
    if not rows:
        return f"I couldn't find {metric_alias} readings for anomaly analysis in {window_label}."
    if not anomalies:
        return (
            f"No strong {metric_alias} anomalies were detected for {scope} in {window_label} "
            f"across {len(rows)} readings."
        )
    top = sorted(
        anomalies,
        key=lambda row: (float(row.get("score") or 0.0), float(row.get("value") or 0.0)),
        reverse=True,
    )[:3]
    details = ", ".join(
        [
            f"{item.get('bucket')} ({float(item.get('value') or 0.0):.2f})"
            for item in top
            if item.get("bucket") is not None
        ]
    )
    return (
        f"Detected {len(anomalies)} {metric_alias} anomalies for {scope} in {window_label} "
        f"from {len(rows)} readings. Top spikes: {details}."
    )


def build_multi_metric_anomaly_answer(
    metric_results: List[Dict[str, Any]],
    window_label: str,
    lab_name: Optional[str],
) -> str:
    """Summarize anomaly detection across multiple metrics."""
    scope = lab_name or "selected scope"
    flagged = [(r["metric"], r["anomalies"], r["rows"]) for r in metric_results if r.get("anomalies")]
    if not flagged:
        metrics_checked = ", ".join(r["metric"] for r in metric_results if r.get("rows"))
        return (
            f"No strong anomalies were detected for {scope} in {window_label} "
            f"across metrics checked: {metrics_checked}."
        )
    parts = []
    for metric, anomalies, rows in flagged:
        top = sorted(anomalies, key=lambda x: float(x.get("score") or 0.0), reverse=True)[:2]
        details = ", ".join(
            f"{item['bucket']} ({float(item.get('value') or 0.0):.2f})"
            for item in top
            if item.get("bucket") is not None
        )
        parts.append(f"{metric}: {len(anomalies)} anomaly point(s) — {details}")
    return (
        f"Anomalies detected for {scope} in {window_label}:\n"
        + "\n".join(f"  • {p}" for p in parts)
    )


def build_comparison_answer(metric_alias: str, rows: List[Dict], window_label: str) -> str:
    if len(rows) < 2:
        return (
            f"I couldn't compare spaces for {metric_alias}. "
            f"Try specifying two spaces like 'lab_a vs lab_b'."
        )
    left, right = rows[0], rows[1]
    delta = float(left.get("avg_value") or 0) - float(right.get("avg_value") or 0)
    direction = "higher" if delta >= 0 else "lower"
    return (
        f"In {window_label}, {left.get('lab_space')} is {abs(delta):.2f} {direction} than "
        f"{right.get('lab_space')} for average {metric_alias}."
    )


def build_temporal_comparison_answer(
    metric_aliases: List[str],
    rows: List[Dict[str, Any]],
    current_label: str,
    ref_label: str,
    lab_name: Optional[str],
    unit: str = "value",
) -> str:
    scope = lab_name or "selected scope"
    if not rows:
        return (
            f"I couldn't find enough data to compare {current_label} against {ref_label} "
            f"for {scope}."
        )
    row = rows[0]
    if len(metric_aliases) == 1:
        metric = metric_aliases[0]
        current_avg = row.get("avg_value")
        ref_avg = row.get("reference_avg")
        if current_avg is None or ref_avg is None:
            return (
                f"I couldn't find enough {metric} data to compare {current_label} against "
                f"{ref_label} for {scope}."
            )
        delta = float(row.get("delta_value") or (float(current_avg) - float(ref_avg)))
        pct = row.get("delta_percent")
        direction = "higher" if delta >= 0 else "lower"
        pct_text = f" ({abs(float(pct)):.1f}% {direction})" if pct is not None else f" ({direction})"
        return (
            f"In {scope}, average {metric} for {current_label} is {float(current_avg):.2f} {unit} "
            f"vs {float(ref_avg):.2f} {unit} for {ref_label}{pct_text}."
        )
    parts: List[str] = []
    for metric in metric_aliases:
        current_v = row.get(f"{metric}_current")
        ref_v = row.get(f"{metric}_reference")
        delta_v = row.get(f"{metric}_delta")
        if current_v is None or ref_v is None:
            continue
        if delta_v is None:
            delta_v = float(current_v) - float(ref_v)
        direction = "higher" if float(delta_v) >= 0 else "lower"
        parts.append(
            f"{metric}: {float(current_v):.2f} ({current_label}) vs {float(ref_v):.2f} ({ref_label}) "
            f"— {abs(float(delta_v)):.2f} {direction}"
        )
    if not parts:
        return (
            f"I found data for {scope} but couldn't compute comparable values "
            f"for {current_label} vs {ref_label}."
        )
    return (
        f"Period comparison for {scope} — {current_label} vs {ref_label}:\n"
        + "\n".join(f"  • {p}" for p in parts)
    )


def build_correlation_answer(
    metric_x: str,
    metric_y: str,
    correlation: Optional[float],
    row_count: int,
    window_label: str,
    lab_name: Optional[str],
) -> str:
    scope = lab_name or "selected scope"
    if row_count < 3 or correlation is None:
        return (
            f"I couldn't compute a reliable correlation between {metric_x} and {metric_y} "
            f"for {scope} in {window_label}. At least 3 paired datapoints are required."
        )
    abs_corr = abs(correlation)
    strength = "very weak"
    if abs_corr >= 0.8:
        strength = "strong"
    elif abs_corr >= 0.5:
        strength = "moderate"
    elif abs_corr >= 0.3:
        strength = "weak"
    direction = "positive" if correlation >= 0 else "negative"
    return (
        f"Computed Pearson correlation between {metric_x} and {metric_y} for {scope} in {window_label}: "
        f"{correlation:.3f} ({strength} {direction} relationship) from {row_count} paired readings."
    )


def ensure_think_prefix(text: str) -> str:
    normalized = str(text or "").strip()
    if not normalized:
        return "<think></think>"
    if normalized.startswith("<think>"):
        return normalized
    return f"<think></think>{normalized}"


def build_db_payload(
    intent: Any,
    metric_alias: str,
    window_label: str,
    rows: List[Dict],
    window_start: Optional[str] = None,
    window_end: Optional[str] = None,
    display_start: Optional[str] = None,
    display_end: Optional[str] = None,
    knowledge_cards: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    intent_value = getattr(intent, "value", str(intent))
    payload: Dict[str, Any] = {
        "intent": intent_value,
        "metric": metric_alias,
        "window": window_label,
        "rows": [serialize_timestamp_value(row) for row in rows[:120]],
    }
    if window_start:
        payload["window_start"] = window_start
    if window_end:
        payload["window_end"] = window_end
    if display_start:
        payload["display_start"] = display_start
    if display_end:
        payload["display_end"] = display_end
    if knowledge_cards:
        payload["knowledge_cards"] = knowledge_cards[:5]
    return payload


def metric_unit(metric_alias: str) -> str:
    return metric_registry.metric_unit(metric_alias) or "value"


def wants_time_series(question: str) -> bool:
    return db_time_windows.wants_time_series(question)


def wants_forecast(question: str) -> bool:
    return db_time_windows.wants_forecast(question)


def topic_matches_card(card: Dict[str, Any], requested_topics: List[str]) -> bool:
    if not requested_topics:
        return True
    card_type = str(card.get("card_type") or "").strip().lower()
    topic = str(card.get("topic") or "").strip().lower()
    title = str(card.get("title") or "").strip().lower()
    topic_targets = {
        "definitions": {"explanation"},
        "metric_explanations": {"interpretation", "rule", "explanation"},
        "ieq_subindex_explanations": {"iaq_subindex", "thermal_subindex", "acoustic_subindex"},
        "recommendations": {"rule"},
        "caveats": {"caveat"},
    }
    for requested in requested_topics:
        if requested == "definitions" and card_type in topic_targets["definitions"]:
            return True
        if requested == "metric_explanations" and card_type in topic_targets["metric_explanations"]:
            return True
        if requested == "ieq_subindex_explanations" and topic in topic_targets["ieq_subindex_explanations"]:
            return True
        if requested == "recommendations" and card_type in topic_targets["recommendations"]:
            return True
        if requested == "caveats" and card_type in topic_targets["caveats"]:
            return True
        if requested.replace("_", " ") in title:
            return True
    return False


def fetch_knowledge_cards(
    *,
    question: str,
    search_fn: Any,
    limit: int = 4,
    card_topics: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    requested_topics = list(card_topics or [])
    try:
        cards = search_fn(question=question, k=max(6, min(limit * 3, 20)))
    except Exception:
        return []
    compact_cards = [
        {
            "card_type": card.get("card_type"),
            "topic": card.get("topic"),
            "title": card.get("title"),
            "summary": card.get("summary"),
            "content": str(card.get("content") or "")[:800],
            "severity_level": card.get("severity_level"),
            "source_label": card.get("source_label"),
        }
        for card in (cards or [])
    ]
    filtered = [card for card in compact_cards if topic_matches_card(card, requested_topics)]
    if filtered:
        return filtered[:limit]
    return compact_cards[:limit]


def split_knowledge_cards(cards: Optional[List[Dict[str, Any]]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    knowledge = []
    guardrails = []
    for card in cards or []:
        if card.get("card_type") == "caveat":
            guardrails.append(card)
        else:
            knowledge.append(card)
    return knowledge, guardrails


def build_multi_metric_comparison_answer(metric_aliases: List[str], rows: List[Dict[str, Any]], window_label: str) -> str:
    if len(rows) < 2:
        return (
            "I couldn't compare the selected air-quality metrics because fewer than two spaces "
            f"had data in {window_label}."
        )
    left, right = rows[0], rows[1]
    parts: List[str] = []
    for metric in metric_aliases:
        left_v = left.get(metric)
        right_v = right.get(metric)
        if left_v is None or right_v is None:
            continue
        delta = float(left_v) - float(right_v)
        direction = "higher" if delta >= 0 else "lower"
        parts.append(f"{metric}: {left.get('lab_space')} is {abs(delta):.2f} {direction} than {right.get('lab_space')}")
    if not parts:
        return f"I found both spaces but couldn't compute comparable metric values in {window_label}."
    return f"Air-quality comparison over {window_label}: " + "; ".join(parts) + "."


def build_multi_metric_aggregation_answer(metric_aliases: List[str], row: Dict[str, Any], window_label: str) -> str:
    if not row:
        return f"I couldn't find metric data for {window_label}."
    lab_raw = row.get("lab_space")
    lab = str(lab_raw).strip() if lab_raw is not None else ""
    if not lab:
        lab = "all_labs"
    parts: List[str] = []
    for metric in metric_aliases:
        value = row.get(metric)
        if value is None:
            continue
        parts.append(f"{metric}={float(value):.2f}")
    if not parts:
        return f"I found data for {lab}, but no metric values were available in {window_label}."
    return f"Air-quality analysis for {lab} over {window_label}: " + ", ".join(parts) + "."


def build_db_sources(
    *,
    operation_type: str,
    metric_alias: str,
    window_label: str,
    window_start: datetime,
    window_end: datetime,
    resolved_lab_name: Optional[str],
    compared_spaces: List[str],
    rows: List[Dict[str, Any]],
    metric_pair: Optional[List[str]] = None,
    correlation: Optional[float] = None,
    metrics_used: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    source: Dict[str, Any] = {
        "source_kind": "db_query",
        "table": "lab_ieq_final",
        "operation_type": operation_type,
        "metric": metric_alias,
        "window_label": window_label,
        "window_start": serialize_datetime_iso(window_start),
        "window_end": serialize_datetime_iso(window_end),
        "lab_scope": resolved_lab_name,
        "compared_spaces": compared_spaces[:2],
        "row_count": len(rows),
    }
    if metric_pair:
        source["metric_pair"] = metric_pair
    if correlation is not None:
        source["correlation"] = correlation
    if metrics_used:
        source["metrics_used"] = metrics_used
    return [source]
