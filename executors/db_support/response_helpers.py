"""Reusable DB-executor response and payload helper functions."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

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

MAX_CHART_LOOKBACK_POINTS = 72  # 3 days at hourly resolution
_TARGET_TZ = timezone(timedelta(hours=4))

DB_TOOL_RESPONSE_DIRECTIVE = """
You are answering from a structured DB query result.
- For air-quality assessment queries, include:
  1) overall status,
  2) metric-by-metric interpretation,
  3) explicit analysis window using the provided time bounds ("from ... to ..."),
  4) stability/trend summary and notable peaks/dips when those stats are available,
  5) missing-metric coverage note (especially TVOC, PM2.5, CO2, humidity),
  6) confidence qualifier tied to metric coverage,
  7) 2-4 practical recommendations.
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
- Provide an overall current air-quality status in plain language.
- Include concise metric-by-metric interpretation for available core metrics (CO2, PM2.5, TVOC, humidity, and IEQ when present).
- Explain what occupants would likely notice/feel.
- Add 2-4 practical recommendations (maintenance actions are allowed when conditions are good).
- If any core metric is missing, call it out clearly and lower confidence in the overall assessment.
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


def to_target_timezone(dt: datetime) -> datetime:
    normalized = dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    return normalized.astimezone(_TARGET_TZ)


def serialize_datetime_iso(dt: datetime) -> str:
    return to_target_timezone(dt).isoformat()


def serialize_timestamp_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return serialize_datetime_iso(value)
    if isinstance(value, list):
        return [serialize_timestamp_value(item) for item in value]
    if isinstance(value, dict):
        return {k: serialize_timestamp_value(v) for k, v in value.items()}
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return value
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return value
        return serialize_datetime_iso(parsed)
    return value


def is_air_quality_query_text(question: str) -> bool:
    q = (question or "").lower()
    if any(hint in q for hint in ("air quality", "indoor air quality", "ieq")):
        return True
    if re.search(r"\bhow\s+(?:is|was)\s+the\s+air\b", q):
        return True
    if re.search(r"\bthe\s+air\b", q) and ("_lab" in q or re.search(r"\b[a-z0-9]+\s+lab\b", q) is not None):
        return True
    return False


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


def db_response_directive(intent: Any, question: str = "") -> str:
    intent_value = getattr(intent, "value", str(intent))
    if intent_value in {"point_lookup_db", "current_status_db"}:
        if is_air_quality_query_text(question) or is_comfort_assessment_query_text(question):
            return DB_TOOL_RESPONSE_DIRECTIVE_AIR_QUALITY_POINT_LOOKUP
        return DB_TOOL_RESPONSE_DIRECTIVE_POINT_LOOKUP
    if intent_value == "comparison_db":
        return DB_TOOL_RESPONSE_DIRECTIVE_COMPARISON
    if intent_value == "forecast_db":
        return DB_TOOL_RESPONSE_DIRECTIVE_FORECAST
    if intent_value == "anomaly_analysis_db":
        return DB_TOOL_RESPONSE_DIRECTIVE_ANOMALY
    return DB_TOOL_RESPONSE_DIRECTIVE


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
    q = (question or "").lower()
    hints = (
        "values",
        "readings",
        "data points",
        "per hour",
        "hourly",
        "over time",
        "trend",
        "this week",
        "last week",
        "this month",
        "last month",
        "last ",
        "past ",
    )
    return any(hint in q for hint in hints)


def wants_forecast(question: str) -> bool:
    q = (question or "").lower()
    hints = (
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
    return any(hint in q for hint in hints)


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
    q = (question or "").lower()
    week_match = re.search(r"\bnext\s+(\d+)\s+weeks?\b", q)
    if week_match:
        weeks = max(1, min(int(week_match.group(1)), 4))
        return weeks * 24 * 7, f"next {weeks} week(s)"
    if "next week" in q:
        return 24 * 7, "next week"
    day_match = re.search(r"\bnext\s+(\d+)\s+days?\b", q)
    if day_match:
        days = max(1, min(int(day_match.group(1)), 31))
        return days * 24, f"next {days} day(s)"
    hour_match = re.search(r"\bnext\s+(\d+)\s+hours?\b", q)
    if hour_match:
        hours = max(1, min(int(hour_match.group(1)), 24 * 14))
        return hours, f"next {hours} hour(s)"
    if "next month" in q:
        return 24 * 30, "next month"
    if "tomorrow" in q:
        return 24, "next 24 hour(s)"
    return 12, "next 12 hour(s)"


def forecast_history_window(
    question: str,
    horizon_hours: int,
    default_start: datetime,
    default_end: datetime,
    default_label: str,
) -> Tuple[datetime, datetime, str]:
    if default_label != "last 24 hours":
        return default_start, default_end, default_label
    now = datetime.now(_TARGET_TZ)
    if horizon_hours <= 12:
        history_hours = 24 * 30
    elif horizon_hours <= 24:
        history_hours = 24 * 60
    elif horizon_hours <= 168:
        history_hours = 24 * 90
    else:
        history_hours = 24 * 120
    start = now - timedelta(hours=history_hours)
    return start, now, f"last {history_hours} hours (auto history for forecast)"


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
        bucket_dt = row["ds"].to_pydatetime().replace(tzinfo=timezone.utc).astimezone(_TARGET_TZ)
        forecast_points.append(
            {
                "bucket": bucket_dt,
                "value": round(float(row["yhat"]), 4),
                "lower": round(float(row["yhat_lower"]), 4),
                "upper": round(float(row["yhat_upper"]), 4),
            }
        )
    history_start = history_df["ds"].iloc[0].to_pydatetime().replace(tzinfo=timezone.utc).astimezone(_TARGET_TZ)
    history_end = history_df["ds"].iloc[-1].to_pydatetime().replace(tzinfo=timezone.utc).astimezone(_TARGET_TZ)
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
    lab = row.get("lab_space", "selected scope")
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
    return {
        "visualization_type": "bar",
        "chart": {
            "title": f"Air-quality metric snapshot ({window_label})",
            "x_label": "metric",
            "y_label": "value",
            "series": [{"name": str(row.get("lab_space", "selected_scope")), "points": points}],
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
