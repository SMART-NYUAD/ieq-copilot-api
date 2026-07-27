"""Shared time-window and timestamp helpers for DB support modules."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import re
from typing import Any, Optional, Tuple

from core_settings import display_timezone, display_utc_offset_hours


def target_tz() -> timezone:
    """The timezone every user-facing timestamp is expressed in.

    Delegates to :func:`core_settings.display_timezone` (``DISPLAY_UTC_OFFSET_HOURS``) so
    parsing, window bounds, and display labels all move together. This used to be a
    module-level constant hardcoded to +4 while the serializers read the setting, which
    meant changing the offset shifted some timestamps and not others.
    """
    return display_timezone()


def target_tz_label() -> str:
    """Human-readable name of the display timezone, e.g. ``GMT+4``."""
    offset = display_utc_offset_hours()
    if offset == 0:
        return "UTC"
    return f"GMT{offset:+d}"


def to_target_timezone(dt: datetime) -> datetime:
    normalized = dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    return normalized.astimezone(target_tz())


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


def parse_bucket_utc(bucket: Any) -> Optional[datetime]:
    """Parse an API bucket timestamp into an aware UTC datetime, or None.

    Bucket strings and window bounds do not share an offset representation — bounds are
    built in the display timezone, buckets come back from the API in its own. Comparing
    them as strings therefore mis-partitions rows near a period boundary; compare the
    parsed instants instead.
    """
    try:
        parsed = datetime.fromisoformat(str(bucket).replace("Z", "+00:00"))
    except (ValueError, TypeError, AttributeError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def format_display_datetime(dt: datetime) -> str:
    rendered = dt.astimezone(target_tz()).strftime(f"%b %d, %Y, %I:%M %p {target_tz_label()}")
    rendered = re.sub(r"^([A-Za-z]{3}) 0(\d),", r"\1 \2,", rendered)
    rendered = re.sub(r", 0(\d):", r", \1:", rendered)
    return rendered


def format_display_window_bounds(window_start: datetime, window_end: datetime) -> Tuple[str, str]:
    return format_display_datetime(window_start), format_display_datetime(window_end)


def granularity_hours_for_window(window_start: datetime, window_end: datetime) -> int:
    """Aggregation granularity (hours per bucket) for the ``indoor-data`` /
    ``agg-summary`` endpoints.

    Always **1 hour**, regardless of the window span. We deliberately do NOT
    coarsen to 6h/12h buckets for wide ranges: every data aggregation in this
    app is hourly so the numbers stay consistent across questions. The window
    arguments are retained for signature compatibility with the call sites.
    """
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


