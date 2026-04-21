"""Reusable DB-executor response and payload helper functions."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
try:
    from executors.db_support import charts as db_charts
    from executors.db_support import time_windows as db_time_windows
except ImportError:
    from . import charts as db_charts
    from . import time_windows as db_time_windows

try:
    from prophet import Prophet
except ImportError:  # pragma: no cover
    Prophet = None


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

DB_TOOL_RESPONSE_DIRECTIVE = """
You are answering from a structured DB query result.
- First, answer the exact user question directly before additional detail.
- Keep the tone warm and personable: write like a helpful IEQ teammate, not a strict compliance report.
- For air-quality assessment/summary queries, include:
  1) overall status,
  2) metric-by-metric interpretation,
  3) explicit analysis window using the provided time bounds ("from ... to ..."),
  4) stability/trend summary and notable peaks/dips when those stats are available,
  5) missing-metric coverage note (especially TVOC, PM2.5, CO2, humidity),
  6) confidence qualifier tied to metric coverage.
- For risk-focused questions, lead with the main risk level and concrete risk drivers first.
- Provide recommendations only when asked, when conditions are concerning, or when user asks for next steps.
- If recommendations are not needed, do not add a "Recommendations" section.
- When `display_start` and `display_end` are present in measured room facts, copy those values verbatim
    when mentioning the analysis window. Do not rewrite or infer date/month values.
- If a metric was requested but not available, explicitly state it as "not available in this window".
- If recommendations are included, keep them actionable and grounded in provided measurements/guidelines.
""".strip()
DB_TOOL_RESPONSE_DIRECTIVE_POINT_LOOKUP = """
You are answering a point lookup from a structured DB query result.
- Lead with the current/latest value requested.
- Use a friendly, reassuring tone where appropriate so the message feels supportive, not robotic.
- Give a short plain-language interpretation with unit and citation-style classification if available.
- Keep it concise (one short paragraph + up to 2 bullets).
- If value is missing, say it clearly and suggest the nearest useful fallback window.
""".strip()
DB_TOOL_RESPONSE_DIRECTIVE_AIR_QUALITY_POINT_LOOKUP = """
You are answering a current air-quality point lookup from a structured DB query result.
- First, directly answer the exact question asked.
- Use a friendly, reassuring tone where appropriate so the message feels supportive, not robotic.
- Provide an overall current air-quality status in plain language.
- Include concise metric-by-metric interpretation for available core metrics (CO2, PM2.5, TVOC, humidity, and IEQ when present).
- Explain what occupants would likely notice/feel.
- Add recommendations only when the user asks for them or risk/quality concerns justify action.
- If conditions are stable/good and no action is requested, end with assessment only (no recommendation bullets).
- If any core metric is missing, call it out clearly and lower confidence in the overall assessment.
- If the question is risk-focused, start with the risk level and the top risk drivers (or say no major risk is evident).
- Do not collapse the answer to a single sentence.
""".strip()
DB_TOOL_RESPONSE_DIRECTIVE_COMPARISON = """
You are answering a comparison from a structured DB query result.
- Highlight which space is better/worse for each available metric and by how much.
- Use a friendly, reassuring tone where appropriate so the message feels supportive, not robotic.
- Use `metric_coverage.available_metrics` and `metric_coverage.missing_metrics` from context as source of truth.
- Call out missing metrics explicitly (especially TVOC for air-quality comparisons).
- Never claim a metric is missing if it appears in `available_metrics` or has numeric values in rows.
- Include practical actions only if the user asks for actions or the weaker metric is materially concerning.
""".strip()
DB_TOOL_RESPONSE_DIRECTIVE_FORECAST = """
You are answering a forecast from a structured DB query result.
- Report forecast horizon, trend direction, and confidence in plain language.
- Mention assumptions/limits and avoid deterministic claims beyond provided forecast output.
- Provide operational recommendations only when requested or when confidence/risk warrants cautionary action.
""".strip()
DB_TOOL_RESPONSE_DIRECTIVE_ANOMALY = """
You are answering an anomaly analysis from a structured DB query result.
- State whether anomalies were detected, when they occurred, and likely occupant impact.
- If no anomalies are detected, say so explicitly.
- Provide troubleshooting/monitoring actions when the user asks for next steps or anomalies are material.
""".strip()
DB_TOOL_RESPONSE_DIRECTIVE_DIAGNOSTIC = """
You are answering a root-cause diagnostic question about IEQ.
- Lead with the specific metric(s) most likely driving the IEQ drop, with evidence.
- State the Pearson correlation in plain language
  (e.g. "CO2 rises strongly when IEQ drops, r=-0.74").
- Describe WHEN the dips occurred and what values the culprit metric had then.
- Rank multiple culprits by correlation strength if present.
- If correlation is weak for all metrics (abs(r) < 0.3), say so and note
  possible data gaps or external factors.
- Do NOT say data is unavailable if correlation_analysis is present in context.
- Do NOT say "I cannot identify" if rows were returned - analyze what is there.
- End with 2-3 targeted actions specific to the identified driver(s).
""".strip()

CITATION_FORMAT_INSTRUCTION = """
CITATION REQUIREMENT — FOLLOW EXACTLY:
When classifying a measured value against a threshold, insert a citation marker
immediately after the claim.

Use this format: [N]
Where N is the source index from the "## Citation Sources" context section.

Examples of correct citation:
  "CO2 at 1,450 ppm exceeds RESET Air Grade A (1,000 ppm) [1]."
  "PM2.5 exceeds the EPA daily threshold [2]."
  "Research suggests cognitive decline above 1,000 ppm [3]."
  "The IEQ score indicates medium quality [4]."

Rules:
1. ONLY use citation indices that appear in the
   "## Citation Sources" section. Never invent an index.
2. Place the marker directly after the specific claim,
   before punctuation where possible.
3. If the same source supports multiple claims, reuse the same index.
4. Do NOT add a References or Footnotes section at the end.
   The system handles reference rendering automatically.
5. If no guideline records are in context, do not add
   any citation markers.
6. For metric-by-metric air-quality assessments:
   - every metric claim with a numeric value (CO2, PM2.5, TVOC, humidity, IEQ)
     MUST include at least one [N] if that metric has a source in Citation Sources.
7. Never cite ASHRAE 62.1 as a CO2 ppm threshold source.
   For CO2 ppm limits/classification, cite RESET/research/internal sources only.
8. For IEQ index classifications, cite the internal IEQ source [N] when available.
""".strip()

FRIENDLY_TONE_INSTRUCTION = """
TONE AND READABILITY:
- Keep wording friendly, supportive, and human while staying evidence-grounded.
- Prefer natural conversational phrasing over rigid policy/report language.
- When risk is low, allow brief reassuring phrasing; when risk is elevated, stay calm and constructive.
- You may use light emoji usage (1-3 relevant emojis per response) when it genuinely improves readability
  (for example: ✅, ⚠️, 🌡️, 💧, 🌬️).
- Do not overuse emojis, and never use emojis in place of concrete evidence or recommendations.
""".strip()

DB_TOOL_RESPONSE_DIRECTIVE = f"""
{DB_TOOL_RESPONSE_DIRECTIVE}

{FRIENDLY_TONE_INSTRUCTION}

{CITATION_FORMAT_INSTRUCTION}
""".strip()
DB_TOOL_RESPONSE_DIRECTIVE_POINT_LOOKUP = f"""
{DB_TOOL_RESPONSE_DIRECTIVE_POINT_LOOKUP}

{FRIENDLY_TONE_INSTRUCTION}

{CITATION_FORMAT_INSTRUCTION}
""".strip()
DB_TOOL_RESPONSE_DIRECTIVE_AIR_QUALITY_POINT_LOOKUP = f"""
{DB_TOOL_RESPONSE_DIRECTIVE_AIR_QUALITY_POINT_LOOKUP}

{FRIENDLY_TONE_INSTRUCTION}

{CITATION_FORMAT_INSTRUCTION}
""".strip()
DB_TOOL_RESPONSE_DIRECTIVE_COMPARISON = f"""
{DB_TOOL_RESPONSE_DIRECTIVE_COMPARISON}

{FRIENDLY_TONE_INSTRUCTION}

{CITATION_FORMAT_INSTRUCTION}
""".strip()
DB_TOOL_RESPONSE_DIRECTIVE_FORECAST = f"""
{DB_TOOL_RESPONSE_DIRECTIVE_FORECAST}

{FRIENDLY_TONE_INSTRUCTION}

{CITATION_FORMAT_INSTRUCTION}
""".strip()
DB_TOOL_RESPONSE_DIRECTIVE_ANOMALY = f"""
{DB_TOOL_RESPONSE_DIRECTIVE_ANOMALY}

{FRIENDLY_TONE_INSTRUCTION}

{CITATION_FORMAT_INSTRUCTION}
""".strip()
DB_TOOL_RESPONSE_DIRECTIVE_DIAGNOSTIC = f"""
{DB_TOOL_RESPONSE_DIRECTIVE_DIAGNOSTIC}

{FRIENDLY_TONE_INSTRUCTION}

{CITATION_FORMAT_INSTRUCTION}
""".strip()


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
    if intent_value == "forecast_db":
        return DB_TOOL_RESPONSE_DIRECTIVE_FORECAST
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
                    period[column] = db_charts._serialize_timestamp_value(value)
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


def build_diagnostic_chart(
    rows: List[Dict[str, Any]],
    culprit_metrics: List[str],
    window_label: str,
    lab_name: Optional[str],
    max_lookback: int,
) -> Dict[str, Any]:
    """
    Multi-series line chart: IEQ index + top culprit metrics.
    """
    recent_rows = db_charts._clip_rows(rows, max_lookback)
    series: List[Dict[str, Any]] = []
    ieq_points = [
        {"x": db_charts._serialize_timestamp_value(row.get("bucket")), "y": float(row.get("ieq") or 0.0)}
        for row in recent_rows
        if row.get("bucket") is not None and row.get("ieq") is not None
    ]
    series.append({"name": "IEQ Index", "points": ieq_points})
    for metric in culprit_metrics:
        if metric == "ieq":
            continue
        points = [
            {"x": db_charts._serialize_timestamp_value(row.get("bucket")), "y": float(row.get(metric) or 0.0)}
            for row in recent_rows
            if row.get("bucket") is not None and row.get(metric) is not None
        ]
        if not points:
            continue
        series.append({"name": metric.upper(), "points": points})
    title_scope = lab_name or "selected scope"
    return {
        "visualization_type": "line",
        "chart": {
            "title": f"IEQ diagnostic drivers ({title_scope}, {window_label})",
            "x_label": "time",
            "y_label": "value",
            "series": series,
            "note": "IEQ and correlated metrics overlaid to identify drivers",
        },
    }


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


def build_forecast_answer(
    metric_alias: str,
    forecast: Optional[Dict[str, Any]],
    window_label: str,
    horizon_label: str,
) -> str:
    if not forecast:
        if Prophet is None:
            return (
                f"I couldn't compute a {metric_alias} forecast because Meta Prophet is not available "
                f"in the runtime environment."
            )
        return (
            f"I couldn't compute a {metric_alias} forecast because there is not enough historical data "
            f"in {window_label}."
        )
    points = forecast.get("forecast_points", [])
    if not points:
        return f"I couldn't compute a {metric_alias} forecast because forecast points were not generated."
    start_point = points[0]
    end_point = points[-1]
    requested_horizon = int(forecast.get("requested_horizon_hours") or forecast.get("horizon_hours") or 0)
    used_horizon = int(forecast.get("horizon_hours") or 0)
    downgrade_note = ""
    if requested_horizon > 0 and used_horizon > 0 and used_horizon < requested_horizon:
        downgrade_note = (
            f" Horizon was automatically reduced from {requested_horizon}h to {used_horizon}h "
            f"to satisfy the history-to-horizon rule."
        )
    return (
        f"Computed a deterministic {metric_alias} forecast for {horizon_label} using a "
        f"{forecast.get('model', 'basic')} model over {forecast.get('history_points_used', 0)} "
        f"historical points from {window_label}. Predicted value moves from "
        f"{start_point.get('value')} at {start_point.get('bucket')} to "
        f"{end_point.get('value')} at {end_point.get('bucket')} "
        f"(confidence: {forecast.get('confidence', 'low')}, score: {forecast.get('confidence_score', 0.0):.2f})."
        f"{downgrade_note}"
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
    forecast: Optional[Dict[str, Any]] = None,
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
    if forecast:
        payload["forecast"] = {
            "model": forecast.get("model"),
            "confidence": forecast.get("confidence"),
            "confidence_score": forecast.get("confidence_score"),
            "history_points_used": forecast.get("history_points_used"),
            "history_start": serialize_timestamp_value(forecast.get("history_start")),
            "history_end": serialize_timestamp_value(forecast.get("history_end")),
            "requested_horizon_hours": forecast.get("requested_horizon_hours"),
            "horizon_hours": forecast.get("horizon_hours"),
            "horizon_downgraded": forecast.get("horizon_downgraded", False),
            "assumptions": forecast.get("assumptions", []),
            "forecast_points": [
                {
                    "bucket": serialize_timestamp_value(row.get("bucket")),
                    "value": row.get("value"),
                    "lower": row.get("lower"),
                    "upper": row.get("upper"),
                }
                for row in forecast.get("forecast_points", [])[:336]
            ],
        }
    if knowledge_cards:
        payload["knowledge_cards"] = knowledge_cards[:5]
    return payload


def metric_unit(metric_alias: str) -> str:
    return METRIC_UNIT_MAP.get(metric_alias, "value")


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


def extract_forecast_horizon_hours(question: str) -> Tuple[int, str]:
    return db_time_windows.extract_forecast_horizon_hours(question)


def forecast_history_window(
    question: str,
    horizon_hours: int,
    default_start: datetime,
    default_end: datetime,
    default_label: str,
) -> Tuple[datetime, datetime, str]:
    return db_time_windows.forecast_history_window(
        question=question,
        horizon_hours=horizon_hours,
        default_start=default_start,
        default_end=default_end,
        default_label=default_label,
    )


def build_forecast_from_rows(series_rows: List[Dict[str, Any]], horizon_hours: int) -> Optional[Dict[str, Any]]:
    cleaned_rows = [row for row in series_rows if row.get("bucket") is not None and row.get("value") is not None]
    if len(cleaned_rows) < 24 or Prophet is None:
        return None
    history = cleaned_rows[-1000:]
    history_df = pd.DataFrame(history).rename(columns={"bucket": "ds", "value": "y"})
    history_df = history_df[["ds", "y"]].copy()
    history_df["ds"] = pd.to_datetime(history_df["ds"], utc=True, errors="coerce")
    history_df["y"] = pd.to_numeric(history_df["y"], errors="coerce")
    history_df = history_df.dropna(subset=["ds", "y"]).sort_values("ds")
    history_df = history_df.drop_duplicates(subset=["ds"], keep="last")
    if len(history_df) < 24:
        return None
    requested_horizon_hours = max(1, int(horizon_hours))
    max_allowed_horizon = max(1, len(history_df) // 5)
    effective_horizon_hours = min(requested_horizon_hours, max_allowed_horizon)
    history_df["ds"] = history_df["ds"].dt.tz_localize(None)
    y_min = float(history_df["y"].quantile(0.01))
    y_max = float(history_df["y"].quantile(0.99))
    floor = max(0.0, y_min * 0.9)
    cap = max(floor, y_max * 1.1)
    history_df["y"] = history_df["y"].clip(lower=floor, upper=cap)
    history_df["floor"] = floor
    history_df["cap"] = cap
    try:
        model = Prophet(
            growth="logistic",
            interval_width=0.9,
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False,
            changepoint_prior_scale=0.01,
            n_changepoints=10,
            seasonality_mode="additive",
        )
        model.fit(history_df)
        future = model.make_future_dataframe(periods=effective_horizon_hours, freq="h", include_history=True)
        future["floor"] = floor
        future["cap"] = cap
        full_forecast = model.predict(future)
    except Exception:
        return None
    forecast_rows = full_forecast.tail(effective_horizon_hours).copy()
    if forecast_rows.empty:
        return None
    forecast_rows["yhat"] = forecast_rows["yhat"].clip(lower=floor, upper=cap)
    forecast_rows["yhat_lower"] = forecast_rows["yhat_lower"].clip(lower=floor, upper=cap)
    forecast_rows["yhat_upper"] = forecast_rows["yhat_upper"].clip(lower=floor, upper=cap)
    mean_pred = max(float(forecast_rows["yhat"].abs().mean()), 1e-6)
    avg_band = float((forecast_rows["yhat_upper"] - forecast_rows["yhat_lower"]).mean())
    interval_ratio = avg_band / mean_pred
    history_factor = min(1.0, len(history_df) / 240.0)
    horizon_penalty = min(1.0, effective_horizon_hours / max(len(history_df), 1))
    confidence_score = max(
        0.0,
        min(1.0, (0.65 * history_factor) + (0.35 / (1.0 + interval_ratio)) - (0.2 * horizon_penalty)),
    )
    confidence = "high" if confidence_score >= 0.75 else "medium" if confidence_score >= 0.45 else "low"
    forecast_points: List[Dict[str, Any]] = []
    for _, row in forecast_rows.iterrows():
        bucket_dt = row["ds"].to_pydatetime().replace(tzinfo=timezone.utc).astimezone(db_time_windows.TARGET_TZ)
        forecast_points.append(
            {
                "bucket": bucket_dt,
                "value": round(float(row["yhat"]), 4),
                "lower": round(float(row["yhat_lower"]), 4),
                "upper": round(float(row["yhat_upper"]), 4),
            }
        )
    history_start = (
        history_df["ds"].iloc[0].to_pydatetime().replace(tzinfo=timezone.utc).astimezone(db_time_windows.TARGET_TZ)
    )
    history_end = (
        history_df["ds"].iloc[-1].to_pydatetime().replace(tzinfo=timezone.utc).astimezone(db_time_windows.TARGET_TZ)
    )
    return {
        "model": "meta_prophet",
        "confidence": confidence,
        "confidence_score": round(confidence_score, 4),
        "history_points_used": int(len(history_df)),
        "history_start": history_start,
        "history_end": history_end,
        "requested_horizon_hours": requested_horizon_hours,
        "horizon_hours": effective_horizon_hours,
        "horizon_downgraded": effective_horizon_hours < requested_horizon_hours,
        "forecast_points": forecast_points,
        "assumptions": [
            "Forecast produced by Prophet trend and seasonality components.",
            "No major occupancy or ventilation regime shift.",
        ],
    }


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


def build_multi_metric_bar_chart(
    metric_aliases: List[str], unit_by_metric: Dict[str, str], window_label: str, rows: List[Dict[str, Any]]
) -> Dict[str, Any]:
    series = []
    for metric in metric_aliases:
        points = []
        for row in rows:
            value = row.get(metric)
            if value is None:
                continue
            points.append({"x": str(row.get("lab_space", "unknown")), "y": float(value)})
        if points:
            series.append({"name": f"{metric} ({unit_by_metric.get(metric, 'value')})", "points": points})
    return {
        "visualization_type": "bar",
        "chart": {
            "title": f"Air-quality metric comparison ({window_label})",
            "x_label": "space",
            "y_label": "value",
            "series": series,
        },
    }


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


def build_multi_metric_snapshot_chart(
    metric_aliases: List[str], unit_by_metric: Dict[str, str], window_label: str, row: Dict[str, Any]
) -> Dict[str, Any]:
    points = []
    for metric in metric_aliases:
        value = row.get(metric)
        if value is None:
            continue
        points.append({"x": metric, "y": float(value), "unit": unit_by_metric.get(metric, "value")})
    lab_raw = row.get("lab_space")
    series_name = str(lab_raw).strip() if lab_raw is not None else ""
    if not series_name:
        series_name = "all_labs"
    return {
        "visualization_type": "bar",
        "chart": {
            "title": f"Air-quality metric snapshot ({window_label})",
            "x_label": "metric",
            "y_label": "value",
            "series": [{"name": series_name, "points": points}],
        },
    }


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
    forecast: Optional[Dict[str, Any]],
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
    if forecast:
        source["forecast"] = {
            "model": forecast.get("model"),
            "confidence": forecast.get("confidence"),
            "confidence_score": forecast.get("confidence_score"),
            "history_points_used": forecast.get("history_points_used"),
            "history_start": serialize_timestamp_value(forecast.get("history_start")),
            "history_end": serialize_timestamp_value(forecast.get("history_end")),
            "requested_horizon_hours": forecast.get("requested_horizon_hours"),
            "horizon_hours": forecast.get("horizon_hours"),
        }
    return [source]
