"""Query parsing helpers for DB executor.

This module contains parsing/resolution utilities extracted from the monolithic
DB executor to keep query preparation logic cohesive and reusable.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import calendar
import re
import time
from typing import Any, Dict, List, Optional, Tuple

try:
    from query_routing.intent_classifier import IntentType
    from storage.postgres_client import get_cursor
except ImportError:
    from ...query_routing.intent_classifier import IntentType
    from ...storage.postgres_client import get_cursor


METRIC_COLUMN_MAP = {
    "pm25": "pm25_avg",
    "pm2.5": "pm25_avg",
    "pm 2.5": "pm25_avg",
    "pm 2 5": "pm25_avg",
    "co2": "co2_avg",
    "tvoc": "voc_avg",
    "voc": "voc_avg",
    "temperature": "temp_avg",
    "temp": "temp_avg",
    "humidity": "humidity_avg",
    "light": "light_avg",
    "lux": "light_avg",
    "sound": "sound_avg",
    "noise": "sound_avg",
    "ieq": "index_value",
    "index": "index_value",
}

CANONICAL_METRIC_COLUMN_MAP = {
    "air_contribution": "contri_air",
    "pm25": "pm25_avg",
    "co2": "co2_avg",
    "tvoc": "voc_avg",
    "temperature": "temp_avg",
    "humidity": "humidity_avg",
    "light": "light_avg",
    "sound": "sound_avg",
    "ieq": "index_value",
}

_SPACE_TOKEN_RE = re.compile(r"\b([a-z0-9]+_lab)\b")
_CORRELATION_HINTS = (
    "correlation",
    "correlate",
    "relationship",
    "associated",
    "association",
    "related",
)

_TARGET_TZ = timezone(timedelta(hours=4))
_TARGET_TZ_LABEL = "GMT+4"
_CONVERSATION_CONTEXT_MARKER = "\n\nprevious conversation context"
_TIME_HINT_RE = re.compile(
    r"\b("
    r"today|yesterday|this week|last week|this month|last month|"
    r"last\s+\d+\s+(hour|hours|day|days|week|weeks|month|months)|"
    r"past\s+\d+\s+(hour|hours|day|days|week|weeks|month|months)|"
    r"\d{4}-\d{1,2}-\d{1,2}|"
    r"(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|"
    r"january|february|march|april|june|july|august|september|october|november|december)"
    r")\b"
)

_LAB_NAMES_CACHE: Tuple[str, ...] = tuple()
_LAB_NAMES_CACHE_TS: float = 0.0
_LAB_NAMES_CACHE_TTL_SECONDS = 300.0


def get_known_lab_names() -> Tuple[str, ...]:
    """Fetch known lab names from app_lab with short-lived cache."""
    global _LAB_NAMES_CACHE
    global _LAB_NAMES_CACHE_TS
    now = time.time()
    if _LAB_NAMES_CACHE and (now - _LAB_NAMES_CACHE_TS) < _LAB_NAMES_CACHE_TTL_SECONDS:
        return _LAB_NAMES_CACHE
    try:
        with get_cursor(real_dict=True) as cur:
            cur.execute("SELECT name FROM app_lab WHERE name IS NOT NULL")
            names = []
            for row in cur.fetchall():
                name = str(row.get("name") or "").strip().lower()
                if name:
                    names.append(name)
            resolved = tuple(sorted(set(names)))
            if resolved:
                _LAB_NAMES_CACHE = resolved
                _LAB_NAMES_CACHE_TS = now
            return resolved
    except Exception:
        return _LAB_NAMES_CACHE


def build_lab_alias_map() -> Dict[str, str]:
    alias_map: Dict[str, str] = {}
    for canonical in get_known_lab_names():
        alias_map[canonical] = canonical
        alias_map[canonical.replace("_", " ")] = canonical
        if canonical.endswith("_lab"):
            base = canonical[: -len("_lab")].strip("_")
            if base:
                alias_map[base] = canonical
                alias_map[base.replace("_", " ")] = canonical
    return alias_map


def resolve_lab_alias(raw_lab: Optional[str]) -> Optional[str]:
    raw = str(raw_lab or "").strip().lower()
    if not raw:
        return None
    token = re.sub(r"[^a-z0-9_\s]", "", raw)
    token = re.sub(r"\s+", " ", token).strip()
    if not token:
        return None
    alias_map = build_lab_alias_map()
    if not alias_map:
        return token.replace(" ", "_")

    candidates: List[str] = []

    def _push(value: str) -> None:
        normalized = str(value or "").strip().lower()
        if normalized and normalized not in candidates:
            candidates.append(normalized)

    _push(token)
    _push(token.replace(" ", "_"))
    _push(token.replace("_", " "))

    if token.endswith(" lab"):
        base = token[: -len(" lab")].strip()
        _push(base)
        _push(f"{base}_lab")
    if token.endswith("_lab"):
        base = token[: -len("_lab")].strip("_")
        _push(base)
        _push(base.replace("_", " "))
    if " " not in token and "_" not in token:
        _push(f"{token}_lab")

    for candidate in candidates:
        canonical = alias_map.get(candidate)
        if canonical:
            return canonical
    return token.replace(" ", "_")


def resolve_labs_from_question(question: str) -> List[str]:
    q = (question or "").lower()
    alias_map = build_lab_alias_map()
    if not alias_map:
        return []
    hits: List[Tuple[int, str]] = []
    for alias, canonical in alias_map.items():
        pattern = rf"(?<![a-z0-9_]){re.escape(alias)}(?![a-z0-9_])"
        for match in re.finditer(pattern, q):
            hits.append((match.start(), canonical))
    hits.sort(key=lambda item: item[0])
    ordered_unique: List[str] = []
    for _, canonical in hits:
        if canonical not in ordered_unique:
            ordered_unique.append(canonical)
    return ordered_unique


def extract_compared_spaces(question: str) -> List[str]:
    q = question.lower()
    resolved_labs = resolve_labs_from_question(q)
    if len(resolved_labs) >= 2:
        return resolved_labs[:2]

    explicit_space_tokens = re.findall(r"\b([a-z0-9]+_lab)\b", q)
    if len(explicit_space_tokens) >= 2:
        deduped: List[str] = []
        for token in explicit_space_tokens:
            if token not in deduped:
                deduped.append(token)
            if len(deduped) == 2:
                break
        if len(deduped) == 2:
            return deduped

    patterns = [
        r"\b([a-z0-9_ ]+?)\s+(?:vs|versus)\s+([a-z0-9_ ]+?)\b",
        r"\bcompare\s+([a-z0-9_ ]+?)\s+(?:and|to|with)\s+([a-z0-9_ ]+?)\b",
        r"\bin(?:\s+the)?\s+([a-z0-9_ ]+?)\s+(?:and|to|with)\s+(?:the\s+)?([a-z0-9_ ]+?)\b",
    ]

    def _normalize_space_token(raw: str) -> str:
        token = (raw or "").strip().strip("?.!,;:")
        token = re.sub(r"\s+", " ", token)
        if token.endswith("_lab"):
            return token
        if token.endswith(" lab"):
            return token.replace(" ", "_")
        if " " in token:
            return token.replace(" ", "_")
        return token

    for pattern in patterns:
        match = re.search(pattern, q)
        if match:
            left = _normalize_space_token(match.group(1))
            right = _normalize_space_token(match.group(2))
            if left and not left.endswith("_lab") and len(left) >= 3:
                left = f"{left}_lab"
            if right and not right.endswith("_lab") and len(right) >= 3:
                right = f"{right}_lab"
            if left and right and left != right:
                left_resolved = resolve_lab_alias(left)
                right_resolved = resolve_lab_alias(right)
                if left_resolved and right_resolved and left_resolved != right_resolved:
                    return [left_resolved, right_resolved]
                return [left, right]
    return []


def extract_space_from_question(question: str) -> Optional[str]:
    q = question.lower()
    resolved_labs = resolve_labs_from_question(q)
    if resolved_labs:
        return resolved_labs[0]

    m = _SPACE_TOKEN_RE.search(q)
    if m:
        return resolve_lab_alias(m.group(1))

    phrase_match = re.search(r"\b([a-z0-9]+)\s+lab\b", q)
    if phrase_match:
        return resolve_lab_alias(f"{phrase_match.group(1)}_lab")

    fallback = re.search(r"\b(?:in|for|at)\s+([a-z0-9_]+)\b", q)
    if fallback:
        candidate = fallback.group(1).strip("?.!,;:")
        if "_" in candidate and len(candidate) >= 4:
            return resolve_lab_alias(candidate)
    return None


def pick_metric(question: str) -> Tuple[str, str]:
    q = question.lower()
    q = re.sub(r"pm\s*2\.?\s*5", "pm2.5", q)
    for alias, column in METRIC_COLUMN_MAP.items():
        if re.search(rf"\b{re.escape(alias)}\b", q):
            return alias, column
    return "ieq", "index_value"


def extract_metric_aliases(question: str) -> List[str]:
    q = (question or "").lower()
    q = re.sub(r"pm\s*2\.?\s*5", "pm2.5", q)
    canonical_by_column = {
        "contri_air": "air_contribution",
        "pm25_avg": "pm25",
        "co2_avg": "co2",
        "voc_avg": "tvoc",
        "temp_avg": "temperature",
        "humidity_avg": "humidity",
        "light_avg": "light",
        "sound_avg": "sound",
        "index_value": "ieq",
    }
    ordered: List[str] = []
    for alias, column in METRIC_COLUMN_MAP.items():
        if re.search(rf"\b{re.escape(alias)}\b", q):
            canonical = canonical_by_column.get(column, alias)
            if canonical not in ordered:
                ordered.append(canonical)
    return ordered


def is_generic_air_quality_scope_query(question: str) -> bool:
    """Detect broad IEQ/air-quality asks where metric/time defaults are acceptable."""
    q = str(question or "").strip().lower()
    if not q:
        return False
    air_quality_hints = (
        "air quality",
        "indoor air quality",
        "ieq",
        "environment quality",
        "indoor environment",
    )
    return any(hint in q for hint in air_quality_hints)


def normalize_metric_alias(metric: str) -> Optional[str]:
    m = str(metric or "").strip().lower().replace(" ", "_")
    if m == "air":
        m = "air_contribution"
    if m in CANONICAL_METRIC_COLUMN_MAP:
        return m
    return None


def planner_metrics(planner_hints: Optional[Dict[str, Any]]) -> List[str]:
    if not isinstance(planner_hints, dict):
        return []
    raw = planner_hints.get("metrics_priority")
    if not isinstance(raw, list):
        return []
    out: List[str] = []
    for item in raw:
        normalized = normalize_metric_alias(item)
        if normalized and normalized not in out:
            out.append(normalized)
    return out


def validate_db_execution_invariants(
    *,
    question: str,
    intent: IntentType,
    selected_metric: str,
    resolved_lab_name: Optional[str],
    request_lab_name: Optional[str],
    explicit_metrics: List[str],
    hinted_metrics: List[str],
    planner_hints: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Verify DB execution inputs are justified by current question/planner output."""
    q = str(question or "").strip().lower()
    signals = (planner_hints or {}).get("query_signals") if isinstance(planner_hints, dict) else {}
    signals = signals if isinstance(signals, dict) else {}
    selected = normalize_metric_alias(selected_metric) or selected_metric
    generic_air_quality_query = is_generic_air_quality_scope_query(q)

    has_time_hint = bool(signals.get("has_time_window_hint")) or _TIME_HINT_RE.search(q) is not None
    has_currentness_hint = any(token in q for token in ("current", "now", "latest", "right now", "at this moment"))
    has_lab_hint = bool(signals.get("has_lab_reference")) or bool(request_lab_name)
    has_db_scope = bool(signals.get("asks_for_db_facts")) or bool(signals.get("has_db_scope_phrase"))
    has_metric_hint = bool(signals.get("has_metric_reference")) or (selected in explicit_metrics)
    metric_explicit_in_planner = selected in hinted_metrics
    analytical_intent = intent in {
        IntentType.AGGREGATION_DB,
        IntentType.COMPARISON_DB,
        IntentType.ANOMALY_ANALYSIS_DB,
        IntentType.FORECAST_DB,
    }

    metric_justified = has_metric_hint or metric_explicit_in_planner
    if not metric_justified and selected in {"ieq", "co2", "pm25", "tvoc", "humidity"}:
        # Permit core IEQ defaults when user asks generic air-quality status.
        metric_justified = generic_air_quality_query or any(token in q for token in ("comfortable", "comfort"))
    if not metric_justified and analytical_intent and has_db_scope:
        # For comparison/anomaly/forecast style intents, fallback metric defaults are valid.
        metric_justified = True

    time_justified = has_time_hint or has_currentness_hint or intent in {
        IntentType.CURRENT_STATUS_DB,
        IntentType.POINT_LOOKUP_DB,
    }
    if not time_justified and analytical_intent and has_db_scope:
        # Aggregation-like intents can safely use deterministic default windows.
        time_justified = True
    resolved_lab_token = str(resolved_lab_name or "").strip().lower()
    has_prepositional_lab_scope = False
    if resolved_lab_token:
        scope_patterns = (
            rf"\bin\s+{re.escape(resolved_lab_token)}\b",
            rf"\bfor\s+{re.escape(resolved_lab_token)}\b",
            rf"\bat\s+{re.escape(resolved_lab_token)}\b",
            rf"\bin\s+{re.escape(resolved_lab_token.replace('_', ' '))}\b",
            rf"\bfor\s+{re.escape(resolved_lab_token.replace('_', ' '))}\b",
            rf"\bat\s+{re.escape(resolved_lab_token.replace('_', ' '))}\b",
        )
        has_prepositional_lab_scope = any(re.search(pattern, q) is not None for pattern in scope_patterns)
    lab_justified = (
        resolved_lab_name is None
        or bool(has_lab_hint)
        or bool(request_lab_name)
        or has_prepositional_lab_scope
        or (generic_air_quality_query and resolved_lab_name is not None)
    )
    if not has_db_scope and generic_air_quality_query:
        has_db_scope = True
    db_scope_justified = has_db_scope or has_time_hint or has_lab_hint or has_metric_hint or generic_air_quality_query

    violations: List[str] = []
    if not metric_justified:
        violations.append("metric_not_justified")
    if not time_justified:
        violations.append("time_window_not_justified")
    if not lab_justified:
        violations.append("lab_scope_not_justified")
    if not db_scope_justified:
        violations.append("db_scope_not_justified")

    allowed = len(violations) == 0
    return {
        "allowed": allowed,
        "violations": violations,
        "justification": {
            "selected_metric": selected,
            "resolved_lab_name": resolved_lab_name,
            "request_lab_name": request_lab_name,
            "has_time_hint": has_time_hint,
            "has_lab_hint": has_lab_hint,
            "has_metric_hint": has_metric_hint,
            "metric_explicit_in_planner": metric_explicit_in_planner,
            "has_db_scope": has_db_scope,
        },
    }


def planner_card_controls(planner_hints: Optional[Dict[str, Any]]) -> Tuple[bool, List[str], int]:
    if not isinstance(planner_hints, dict):
        return False, [], 2
    raw_needs_cards = planner_hints.get("needs_cards")
    raw_topics = planner_hints.get("card_topics")
    raw_max_cards = planner_hints.get("max_cards")
    needs_cards = bool(raw_needs_cards) if isinstance(raw_needs_cards, bool) else False
    card_topics: List[str] = []
    if isinstance(raw_topics, list):
        for item in raw_topics:
            topic = str(item or "").strip().lower().replace(" ", "_")
            if topic and topic not in card_topics:
                card_topics.append(topic)
    try:
        max_cards = int(raw_max_cards)
    except (TypeError, ValueError):
        max_cards = 2
    max_cards = max(1, min(4, max_cards))
    return needs_cards, card_topics, max_cards


def wants_correlation(question: str) -> bool:
    q = (question or "").lower()
    return any(hint in q for hint in _CORRELATION_HINTS)


def _resolve_year_for_month(month: int, explicit_year: Optional[int], now: datetime) -> int:
    if explicit_year:
        return explicit_year
    return now.year if month <= now.month else now.year - 1


def strip_conversation_context(question: str) -> str:
    """Strip appended conversation transcript from effective question text."""
    text = str(question or "")
    marker_idx = text.lower().find(_CONVERSATION_CONTEXT_MARKER.lower())
    if marker_idx >= 0:
        text = text[:marker_idx]
    return text.strip()


def _latest_user_question(question: str) -> str:
    """Strip appended conversation transcript from effective question text."""
    return strip_conversation_context(question)


def _cap_window_end_at_now(start: datetime, end: datetime, now: datetime) -> datetime:
    """Avoid reporting/using future bounds when the requested window reaches into the future."""
    if start <= now < end:
        return now
    return end


def extract_time_window(question: str, default_hours: int = 24) -> Tuple[datetime, datetime, str]:
    q = _latest_user_question(question).lower()
    now = datetime.now(_TARGET_TZ)
    month_names = [m.lower() for m in calendar.month_name if m]
    month_abbr = [m.lower() for m in calendar.month_abbr if m]
    month_lookup = {name: idx + 1 for idx, name in enumerate(month_names)}
    month_lookup.update({name: idx + 1 for idx, name in enumerate(month_abbr)})
    year_match = re.search(r"\b(20\d{2}|19\d{2})\b", q)
    explicit_year = int(year_match.group(1)) if year_match else None

    day_month_re = r"\b(\d{1,2})(?:st|nd|rd|th)?(?:\s+of)?\s+(" + "|".join(month_lookup.keys()) + r")\b"
    day_month_match = re.search(day_month_re, q)
    if day_month_match:
        day = int(day_month_match.group(1))
        month = month_lookup[day_month_match.group(2)]
        year = _resolve_year_for_month(month, explicit_year, now)
        try:
            start = datetime(year, month, day, tzinfo=_TARGET_TZ)
            end = _cap_window_end_at_now(start, start + timedelta(days=1), now)
            return start, end, start.strftime("%B %d, %Y")
        except ValueError:
            pass

    month_day_re = r"\b(" + "|".join(month_lookup.keys()) + r")\s+(\d{1,2})(?:st|nd|rd|th)?\b"
    month_day_match = re.search(month_day_re, q)
    if month_day_match:
        month = month_lookup[month_day_match.group(1)]
        day = int(month_day_match.group(2))
        year = _resolve_year_for_month(month, explicit_year, now)
        try:
            start = datetime(year, month, day, tzinfo=_TARGET_TZ)
            end = _cap_window_end_at_now(start, start + timedelta(days=1), now)
            return start, end, start.strftime("%B %d, %Y")
        except ValueError:
            pass

    iso_match = re.search(r"\b(20\d{2}|19\d{2})-(\d{1,2})-(\d{1,2})\b", q)
    if iso_match:
        year = int(iso_match.group(1))
        month = int(iso_match.group(2))
        day = int(iso_match.group(3))
        try:
            start = datetime(year, month, day, tzinfo=_TARGET_TZ)
            end = _cap_window_end_at_now(start, start + timedelta(days=1), now)
            return start, end, start.strftime("%B %d, %Y")
        except ValueError:
            pass

    month_pattern = r"\b(" + "|".join(month_lookup.keys()) + r")\b"
    month_match = re.search(month_pattern, q)
    if month_match:
        month_token = month_match.group(1)
        month = month_lookup[month_token]
        year = _resolve_year_for_month(month, explicit_year, now)
        start = datetime(year, month, 1, tzinfo=_TARGET_TZ)
        month_end = datetime(year + 1, 1, 1, tzinfo=_TARGET_TZ) if month == 12 else datetime(
            year, month + 1, 1, tzinfo=_TARGET_TZ
        )
        end = _cap_window_end_at_now(start, month_end, now)
        return start, end, start.strftime("%B %Y")

    if re.search(r"\b(last|past)\s+hour\b", q):
        return now - timedelta(hours=1), now, "last 1 hour"

    hour_match = re.search(r"\b(last|past)\s+(\d+)\s+hours?\b", q)
    if hour_match:
        hours = max(1, min(int(hour_match.group(2)), 24 * 31))
        return now - timedelta(hours=hours), now, f"last {hours} hours"

    day_match = re.search(r"\b(last|past)\s+(\d+)\s+days?\b", q)
    if day_match:
        days = max(1, min(int(day_match.group(2)), 366))
        return now - timedelta(days=days), now, f"last {days} days"

    if "last week" in q:
        start_of_week = (now - timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
        prev_week_start = start_of_week - timedelta(days=7)
        return prev_week_start, start_of_week, "last week"
    if "this week" in q:
        start_of_week = (now - timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
        return start_of_week, now, "this week"
    if "yesterday" in q:
        start = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return start, start + timedelta(days=1), "yesterday"
    if "today" in q:
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return start, now, "today"

    weekdays = list(calendar.day_name)
    weekdays_lower = [d.lower() for d in weekdays]
    weekday_pattern = r"\b(" + "|".join(weekdays_lower) + r")\b"
    weekday_match = re.search(weekday_pattern, q)
    if weekday_match:
        wd = weekdays_lower.index(weekday_match.group(1))
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        delta = (today_start.weekday() - wd) % 7
        if delta == 0:
            delta = 7
        if "last " in q:
            delta += 7
        start = today_start - timedelta(days=delta)
        end = start + timedelta(days=1)
        return start, end, f"{weekdays[wd]} ({start.date().isoformat()})"

    fallback_hours = max(1, min(int(default_hours or 24), 24 * 31))
    return now - timedelta(hours=fallback_hours), now, f"last {fallback_hours} hours"


def default_window_hours_for_intent(intent: IntentType) -> int:
    if intent in {IntentType.CURRENT_STATUS_DB, IntentType.POINT_LOOKUP_DB}:
        return 1
    if intent in {IntentType.AGGREGATION_DB, IntentType.COMPARISON_DB, IntentType.ANOMALY_ANALYSIS_DB}:
        return 24
    return 24


def format_display_datetime(dt: datetime) -> str:
    rendered = dt.astimezone(_TARGET_TZ).strftime(f"%b %d, %Y, %I:%M %p {_TARGET_TZ_LABEL}")
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

