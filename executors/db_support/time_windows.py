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


def granularity_hours_for_window(window_start: datetime, window_end: datetime) -> int:
    """Derive the aggregation granularity (hours per bucket) from the window span.

    Mirrors the frontend's range-based granularity rule for the
    ``indoor-data`` / ``agg-summary`` endpoints:

    - a month or more  → 12 hours
    - a week up to a month → 6 hours
    - less than a week → 1 hour
    """
    try:
        span_hours = max(0.0, (window_end - window_start).total_seconds() / 3600.0)
    except Exception:
        span_hours = 0.0
    span_days = span_hours / 24.0
    if span_days >= 28.0:
        return 12
    if span_days >= 7.0:
        return 6
    return 1


def widen_window_to_min_span(
    window_start: datetime, window_end: datetime, min_hours: float
) -> Tuple[datetime, datetime]:
    """Extend ``window_start`` backward so the span is at least ``min_hours``.

    Used where downstream analysis (trend/anomaly baselines) needs a minimum
    number of buckets regardless of the user's stated window.
    """
    try:
        span_hours = (window_end - window_start).total_seconds() / 3600.0
    except Exception:
        return window_start, window_end
    if span_hours < float(min_hours):
        return window_end - timedelta(hours=float(min_hours)), window_end
    return window_start, window_end


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


