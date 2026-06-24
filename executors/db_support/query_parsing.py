"""Query parsing helpers for DB executor.

This module contains parsing/resolution utilities extracted from the monolithic
DB executor to keep query preparation logic cohesive and reusable.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import calendar
import re
from typing import Any, Dict, List, Optional, Tuple

from query_routing.intent_classifier import IntentType
from executors.db_support import time_windows as db_time_windows
from executors.metric_registry import METRIC_COLUMN_MAP, CANONICAL_METRIC_COLUMN_MAP

_SPACE_TOKEN_RE = re.compile(r"\b([a-z0-9]+_lab)\b")
_CORRELATION_HINTS = (
    "correlation",
    "correlate",
    "relationship",
    "associated",
    "association",
    "related",
)
_BASELINE_REFERENCE_HINTS = (
    "baseline",
    "normal",
    "usual",
    "typical",
    "expected",
    "reference",
    "deviation",
)
_TEMPORAL_CURRENT_PATTERNS = (
    r"\btoday\b",
    r"\bthis\s+week\b",
    r"\bthis\s+month\b",
)
_TEMPORAL_REFERENCE_PATTERNS = (
    r"\byesterday\b",
    r"\blast\s+week\b",
    r"\blast\s+month\b",
    r"\blast\s+\d+\s+days?\b",
    r"\blast\s+\d+\s+hours?\b",
)
_DEICTIC_SCOPE_HINTS = (
    " over there",
    " in there",
    " right there",
    " that room",
    " this room",
    " same room",
    " same lab",
    " same space",
    " in the room",
    " in this room",
    " the room",
    " the lab",
    " in the lab",
    " in here",
)
_GLOBAL_SCOPE_HINTS = (
    "all labs",
    "all lab spaces",
    "across all labs",
    "across labs",
    "all spaces",
    "across all spaces",
    "across spaces",
    "all rooms",
    "across all rooms",
    "every lab",
    "every space",
    "all_labs",
)

_CONVERSATION_CONTEXT_MARKER = "\n\nprevious conversation context"
_TIME_HINT_RE = re.compile(
    r"\b("
    r"today|yesterday|tomorrow|this week|last week|this month|last month|"
    r"last\s+\d+\s+(hour|hours|day|days|week|weeks|month|months)|"
    r"past\s+\d+\s+(hour|hours|day|days|week|weeks|month|months)|"
    r"next\s+\d+\s+(hour|hours|day|days|week|weeks|month|months)|"
    r"last\s+(hour|day|week|month|hours|days|weeks|months)|"
    r"past\s+(hour|day|week|month|hours|days|weeks|months)|"
    r"next\s+(hour|day|week|month)|"
    r"this\s+(hour|morning|afternoon|evening)|"
    r"recent\s+(hour|hours|day|days)|"
    r"\d{4}-\d{1,2}-\d{1,2}|"
    r"(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|"
    r"january|february|march|april|june|july|august|september|october|november|december)"
    r")\b"
)

# Generic relative windows introduced by a determiner ("for the week",
# "over the past month", "the day"). These are *not* the calendar-anchored
# phrases above ("this week"/"last week"); they read as plain rolling windows.
# Without this, a follow-up like "show me for the week" matches no time phrase
# and silently collapses to the default 24h window — looking like the prior
# turn's range was carried over.
_GENERIC_RELATIVE_WINDOW_RE = re.compile(
    r"\b(?:the|a|an|this|that|over|for|past|current|recent|whole|entire|each|every|per)\s+"
    r"(?:the\s+|a\s+|past\s+|last\s+|current\s+|whole\s+|entire\s+|few\s+)?"
    r"(?P<unit>hour|day|week|month|year)s?\b"
)
_GENERIC_RELATIVE_WINDOW_SPEC = {
    "hour": (timedelta(hours=1), "last 1 hour"),
    "day": (timedelta(hours=24), "last 24 hours"),
    "week": (timedelta(days=7), "last 7 days"),
    "month": (timedelta(days=30), "last 30 days"),
    "year": (timedelta(days=365), "last 365 days"),
}

def resolve_lab_alias(raw_lab: Optional[str]) -> Optional[str]:
    """Normalize a client-provided lab slug without remote validation."""
    raw = str(raw_lab or "").strip().lower()
    if not raw:
        return None
    token = re.sub(r"[^a-z0-9_\s]", "", raw)
    token = re.sub(r"\s+", " ", token).strip()
    if not token:
        return None
    return token.replace(" ", "_")


def resolve_labs_from_question(question: str) -> List[str]:
    q = (question or "").lower()
    ordered_unique: List[str] = []

    def _append(lab: Optional[str]) -> None:
        if lab and lab not in ordered_unique:
            ordered_unique.append(lab)

    for match in _SPACE_TOKEN_RE.finditer(q):
        _append(resolve_lab_alias(match.group(1)))

    for match in re.finditer(r"\b([a-z0-9]+)\s+lab\b", q):
        _append(resolve_lab_alias(f"{match.group(1)}_lab"))

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
    first_match: Optional[Tuple[int, str, str]] = None
    for alias, column in METRIC_COLUMN_MAP.items():
        match = re.search(rf"\b{re.escape(alias)}\b", q)
        if match and (first_match is None or match.start() < first_match[0]):
            first_match = (match.start(), alias, column)
    if first_match is not None:
        return first_match[1], first_match[2]
    return "ieq", "index_value"


def extract_metric_aliases(question: str) -> List[str]:
    q = (question or "").lower()
    q = re.sub(r"pm\s*2\.?\s*5", "pm2.5", q)
    canonical_by_column = {
        "contri_air": "air_contribution",
        "pm25_avg": "pm25",
        "co2_avg": "co2",
        "voc_avg": "voc",
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


def is_baseline_reference_query(question: str) -> bool:
    q = str(question or "").strip().lower()
    if not q:
        return False
    return any(token in q for token in _BASELINE_REFERENCE_HINTS)


def is_temporal_period_comparison(question: str) -> bool:
    """Return True when the question compares within the same lab across two explicit time periods."""
    q = str(question or "").strip().lower()
    has_current = any(re.search(pat, q) is not None for pat in _TEMPORAL_CURRENT_PATTERNS)
    has_reference = any(re.search(pat, q) is not None for pat in _TEMPORAL_REFERENCE_PATTERNS)
    return has_current and has_reference


def extract_temporal_comparison_windows(
    question: str,
) -> Optional[Tuple[datetime, datetime, str, datetime, datetime, str]]:
    """
    Extract both time windows for a temporal period comparison query.
    Returns (current_start, current_end, current_label, ref_start, ref_end, ref_label)
    or None if both periods cannot be resolved.
    """
    q = str(question or "").strip().lower()
    now = datetime.now(db_time_windows.TARGET_TZ)

    current_start: Optional[datetime] = None
    current_end: Optional[datetime] = None
    current_label: str = ""

    if re.search(r"\btoday\b", q):
        current_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        current_end = now
        current_label = "today"
    elif re.search(r"\bthis\s+week\b", q):
        current_start = (now - timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
        current_end = now
        current_label = "this week"
    elif re.search(r"\bthis\s+month\b", q):
        current_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        current_end = now
        current_label = "this month"

    if current_start is None:
        return None

    ref_start: Optional[datetime] = None
    ref_end: Optional[datetime] = None
    ref_label: str = ""

    if re.search(r"\byesterday\b", q):
        ref_start = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        ref_end = ref_start + timedelta(days=1)
        ref_label = "yesterday"
    elif re.search(r"\blast\s+week\b", q):
        start_of_week = (now - timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
        ref_start = start_of_week - timedelta(days=7)
        ref_end = start_of_week
        ref_label = "last week"
    elif re.search(r"\blast\s+month\b", q):
        this_month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if now.month == 1:
            ref_start = this_month_start.replace(year=now.year - 1, month=12)
        else:
            ref_start = this_month_start.replace(month=now.month - 1)
        ref_end = this_month_start
        ref_label = "last month"
    else:
        days_m = re.search(r"\blast\s+(\d+)\s+days?\b", q)
        if days_m:
            days = max(1, min(int(days_m.group(1)), 366))
            ref_start = now - timedelta(days=days)
            ref_end = now
            ref_label = f"last {days} days"
        else:
            hours_m = re.search(r"\blast\s+(\d+)\s+hours?\b", q)
            if hours_m:
                hours = max(1, min(int(hours_m.group(1)), 24 * 31))
                ref_start = now - timedelta(hours=hours)
                ref_end = now
                ref_label = f"last {hours} hours"

    if ref_start is None:
        return None

    return (current_start, current_end, current_label, ref_start, ref_end, ref_label)


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
    selected = normalize_metric_alias(selected_metric) or selected_metric
    generic_air_quality_query = is_generic_air_quality_scope_query(q)

    has_time_hint = _TIME_HINT_RE.search(q) is not None
    has_currentness_hint = any(token in q for token in ("current", "now", "latest", "right now", "at this moment"))
    extracted_lab_scope = extract_space_from_question(q)
    has_explicit_lab_scope = bool(extracted_lab_scope)
    has_db_scope = False
    has_metric_hint = selected in explicit_metrics
    has_deictic_scope_hint = any(token in f" {q} " for token in _DEICTIC_SCOPE_HINTS)
    is_baseline_query = is_baseline_reference_query(q)
    compared_spaces = extract_compared_spaces(q)
    has_explicit_second_space = len(compared_spaces) >= 2
    if has_explicit_second_space:
        has_explicit_lab_scope = True
    has_lab_hint = bool(request_lab_name) or has_explicit_lab_scope
    metric_explicit_in_planner = selected in hinted_metrics
    analytical_intent = intent in {
        IntentType.AGGREGATION_DB,
        IntentType.COMPARISON_DB,
        IntentType.ANOMALY_ANALYSIS_DB,
        IntentType.FORECAST_DB,
    }

    metric_justified = has_metric_hint or metric_explicit_in_planner
    if not metric_justified and selected in {"ieq", "co2", "pm25", "voc", "humidity"}:
        # Permit core IEQ defaults when user asks generic air-quality status.
        metric_justified = generic_air_quality_query or any(token in q for token in ("comfortable", "comfort"))
    if not metric_justified and analytical_intent and has_db_scope:
        # For comparison/anomaly/forecast style intents, fallback metric defaults are valid.
        metric_justified = True
    if not metric_justified and analytical_intent and has_lab_hint:
        # When the user names a lab for an analytical query, the default metric is valid.
        metric_justified = True

    time_justified = has_time_hint or has_currentness_hint or intent in {
        IntentType.CURRENT_STATUS_DB,
        IntentType.POINT_LOOKUP_DB,
        IntentType.FORECAST_DB,  # always "next 6 hours" — no user time phrase needed
    }
    if not time_justified and analytical_intent and has_db_scope:
        # Aggregation-like intents can safely use deterministic default windows.
        time_justified = True
    if not time_justified and analytical_intent and has_lab_hint:
        # When the user explicitly names a lab for an analytical intent, the default
        # window is always well-defined — don't gate on a missing time phrase.
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
    has_global_scope_hint = any(token in q for token in _GLOBAL_SCOPE_HINTS)
    lab_justified = (
        bool(resolved_lab_name)
        or bool(has_lab_hint)
        or bool(request_lab_name)
        or has_prepositional_lab_scope
        or has_explicit_second_space
        or has_global_scope_hint
    )
    if has_deictic_scope_hint and not has_explicit_lab_scope and not request_lab_name:
        # Vague room references ("in the room", "here", etc.) without an explicit lab
        # name in the question should not silently use a lab inferred from conversation
        # context — ask for clarification instead. This applies to all DB intents.
        lab_justified = False
    if (
        not has_global_scope_hint
        and resolved_lab_name is None
        and not has_lab_hint
        and not request_lab_name
    ):
        # For measured DB queries, require explicit lab scope unless user asks
        # for an explicit global scope (e.g. "across all labs").
        lab_justified = False
    if (
        intent in {IntentType.CURRENT_STATUS_DB, IntentType.POINT_LOOKUP_DB}
        and resolved_lab_name is None
        and not has_lab_hint
        and not request_lab_name
    ):
        # For current/point asks, avoid silently selecting an arbitrary latest lab.
        lab_justified = False
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
            "is_baseline_reference_query": is_baseline_query,
            "has_explicit_second_space": has_explicit_second_space,
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


def has_explicit_time_hint(question: str) -> bool:
    """Return True if the question contains an explicit time window reference."""
    q = str(question or "").lower()
    return (
        _TIME_HINT_RE.search(q) is not None
        or _GENERIC_RELATIVE_WINDOW_RE.search(q) is not None
    )


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
    """Return the current turn question text only."""
    return str(question or "").strip()


def _cap_window_end_at_now(start: datetime, end: datetime, now: datetime) -> datetime:
    """Avoid reporting/using future bounds when the requested window reaches into the future."""
    if start <= now < end:
        return now
    return end


def _month_start(year: int, month: int) -> datetime:
    return datetime(year, month, 1, tzinfo=db_time_windows.TARGET_TZ)


def _shift_month(year: int, month: int, delta_months: int) -> Tuple[int, int]:
    absolute = (year * 12 + (month - 1)) + delta_months
    shifted_year = absolute // 12
    shifted_month = (absolute % 12) + 1
    return shifted_year, shifted_month


def extract_time_window(question: str, default_hours: int = 24) -> Tuple[datetime, datetime, str]:
    q = _latest_user_question(question).lower()
    now = datetime.now(db_time_windows.TARGET_TZ)
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
            start = datetime(year, month, day, tzinfo=db_time_windows.TARGET_TZ)
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
            start = datetime(year, month, day, tzinfo=db_time_windows.TARGET_TZ)
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
            start = datetime(year, month, day, tzinfo=db_time_windows.TARGET_TZ)
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
        start = datetime(year, month, 1, tzinfo=db_time_windows.TARGET_TZ)
        month_end = datetime(year + 1, 1, 1, tzinfo=db_time_windows.TARGET_TZ) if month == 12 else datetime(
            year, month + 1, 1, tzinfo=db_time_windows.TARGET_TZ
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

    if "last month" in q:
        this_month_start = _month_start(now.year, now.month)
        prev_year, prev_month = _shift_month(now.year, now.month, -1)
        prev_month_start = _month_start(prev_year, prev_month)
        return prev_month_start, this_month_start, "last month"
    if "this month" in q:
        this_month_start = _month_start(now.year, now.month)
        return this_month_start, now, "this month"
    months_match = re.search(r"\b(last|past)\s+(\d+)\s+months?\b", q)
    if months_match:
        months = max(1, min(int(months_match.group(2)), 12))
        start_year, start_month = _shift_month(now.year, now.month, -months)
        start = _month_start(start_year, start_month)
        return start, now, f"last {months} months"

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

    # Generic determiner-introduced windows ("for the week", "over the past
    # month", "the day") resolve to plain rolling windows. Calendar-anchored
    # phrases ("this week"/"last week"/"this month"/...) are handled above and
    # return before reaching here, so this only catches the leftovers.
    generic_match = _GENERIC_RELATIVE_WINDOW_RE.search(q)
    if generic_match:
        span, label = _GENERIC_RELATIVE_WINDOW_SPEC[generic_match.group("unit")]
        return now - span, now, label

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
    return db_time_windows.format_display_datetime(dt)


def format_display_window_bounds(window_start: datetime, window_end: datetime) -> Tuple[str, str]:
    return db_time_windows.format_display_window_bounds(window_start, window_end)


def wants_time_series(question: str) -> bool:
    return db_time_windows.wants_time_series(question)


