"""Shared time-window and timestamp helpers for DB support modules."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import re
from typing import Any, Optional, Tuple


TARGET_TZ = timezone(timedelta(hours=4))
TARGET_TZ_LABEL = "GMT+4"


def to_target_timezone(dt: datetime) -> datetime:
    normalized = dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    return normalized.astimezone(TARGET_TZ)


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


def format_display_datetime(dt: datetime) -> str:
    rendered = dt.astimezone(TARGET_TZ).strftime(f"%b %d, %Y, %I:%M %p {TARGET_TZ_LABEL}")
    rendered = re.sub(r"^([A-Za-z]{3}) 0(\d),", r"\1 \2,", rendered)
    rendered = re.sub(r", 0(\d):", r", \1:", rendered)
    return rendered


def format_display_window_bounds(window_start: datetime, window_end: datetime) -> Tuple[str, str]:
    return format_display_datetime(window_start), format_display_datetime(window_end)


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
    _ = question
    if default_label != "last 24 hours":
        return default_start, default_end, default_label
    now = datetime.now(TARGET_TZ)
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
