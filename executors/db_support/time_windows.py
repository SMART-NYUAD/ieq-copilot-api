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


