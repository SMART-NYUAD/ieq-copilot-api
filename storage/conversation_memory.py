"""Structured routing memory extracted from compact conversation context."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict, Optional, Tuple

try:
    from executors.db_support.query_parsing import extract_space_from_question
except ImportError:
    from ..executors.db_support.query_parsing import extract_space_from_question


_METRIC_KEYWORDS = (
    "ieq",
    "co2",
    "pm2.5",
    "pm25",
    "tvoc",
    "temperature",
    "humidity",
    "light",
    "sound",
)
_TIME_PHRASES = (
    "today",
    "yesterday",
    "this week",
    "last week",
    "this month",
    "last month",
)
_FOLLOW_UP_HINTS = (
    "what about",
    "and what about",
    "also",
    "how about",
    "same",
    "there",
    "that room",
    "same room",
    "same lab",
    "same space",
)
_CLARIFICATION_SELECTION_HINTS = (
    "data-backed answer",
    "exact values",
    "from the database",
    "high-level conceptual explanation",
    "conceptual explanation",
)
_TOPIC_PHRASES = (
    "air quality",
    "indoor air quality",
    "ieq",
    "comfort",
    "comfortable",
)


@dataclass(frozen=True)
class RoutingMemory:
    """Compact structured memory for routing carry-over."""

    lab_name: Optional[str] = None
    metric: Optional[str] = None
    time_phrase: Optional[str] = None
    topic_phrase: Optional[str] = None
    previous_user: str = ""


def _extract_requested_metrics(question: str) -> list[str]:
    q = str(question or "").lower()
    out: list[str] = []
    for token in _METRIC_KEYWORDS:
        if token in q and token not in out:
            out.append(token)
    return out


def _requested_time_phrase(question: str) -> Optional[str]:
    q = str(question or "").lower()
    for token in ("right now", "at this moment", "latest", "current", "now"):
        if token in q:
            return token
    for phrase in _TIME_PHRASES:
        if phrase in q:
            return phrase
    m = re.search(r"\b(last|past)\s+\d+\s+(hour|hours|day|days|week|weeks|month|months)\b", q)
    if m:
        return m.group(0)
    return None


def _context_user_lines(conversation_context: str) -> list[str]:
    lines: list[str] = []
    for line in str(conversation_context or "").splitlines():
        if line.strip().lower().startswith("user:"):
            lines.append(line.split(":", 1)[1].strip())
    return lines


def _latest_value_from_user_lines(user_lines: list[str], extractor) -> Any:
    for candidate in reversed(user_lines):
        value = extractor(candidate)
        if value:
            return value
    return None


def _extract_topic_phrase(question: str) -> Optional[str]:
    q = str(question or "").lower()
    for phrase in _TOPIC_PHRASES:
        if phrase in q:
            return phrase
    return None


def _should_apply_memory(question: str, current_signals: Dict[str, Any]) -> bool:
    q = str(question or "").strip().lower()
    if not q:
        return False
    token_count = len([item for item in re.split(r"\s+", q) if item])
    has_followup_hint = any(hint in q for hint in _FOLLOW_UP_HINTS)
    has_clarification_selection_hint = any(hint in q for hint in _CLARIFICATION_SELECTION_HINTS)
    has_lab_reference = bool(current_signals.get("has_lab_reference"))
    has_time_window_hint = bool(current_signals.get("has_time_window_hint"))
    has_metric_reference = bool(current_signals.get("has_metric_reference"))
    if has_followup_hint:
        return True
    # Clarification-choice replies (e.g. "data-backed answer...") should carry
    # prior lab/time scope even when they are verbose.
    if has_clarification_selection_hint:
        return True
    # Lab-only clarification replies (e.g. "smart_lab") should inherit missing
    # metric/time context from the immediate prior user turn.
    if token_count <= 3 and has_lab_reference and not has_time_window_hint and not has_metric_reference:
        return True
    if token_count <= 6 and (has_time_window_hint or has_metric_reference) and not has_lab_reference:
        return True
    if token_count <= 4 and not (has_lab_reference or has_time_window_hint or has_metric_reference):
        return True
    return False


def extract_routing_memory(conversation_context: str, current_signals: Dict[str, Any]) -> RoutingMemory:
    """Extract only lab/metric/time from prior user turns."""
    _ = current_signals  # reserved for future weighting logic
    user_lines = _context_user_lines(conversation_context)
    if not user_lines:
        return RoutingMemory()
    previous_user = str(user_lines[-1] or "").strip()
    previous_lab = _latest_value_from_user_lines(user_lines, extract_space_from_question)
    previous_time = _latest_value_from_user_lines(user_lines, _requested_time_phrase)
    previous_metric = _latest_value_from_user_lines(
        user_lines, lambda text: (_extract_requested_metrics(text) or [None])[0]
    )
    previous_topic = _latest_value_from_user_lines(user_lines, _extract_topic_phrase)
    return RoutingMemory(
        lab_name=previous_lab,
        metric=previous_metric,
        time_phrase=previous_time,
        topic_phrase=previous_topic,
        previous_user=previous_user,
    )


def apply_routing_memory(
    question: str,
    lab_name: Optional[str],
    memory: RoutingMemory,
    current_signals: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Optional[str], Dict[str, Any]]:
    """Apply structured carry-over without injecting raw conversation text."""
    base_question = str(question or "").strip()
    signals = dict(current_signals or {})
    if not base_question:
        return base_question, (lab_name or "").strip() or None, {"applied": False}
    if not _should_apply_memory(base_question, signals):
        return base_question, (lab_name or "").strip() or None, {"applied": False}

    has_lab_reference = bool(signals.get("has_lab_reference"))
    has_time_window_hint = bool(signals.get("has_time_window_hint"))
    has_metric_reference = bool(signals.get("has_metric_reference"))

    effective_question = base_question
    effective_lab = (lab_name or "").strip() or None
    carried_lab = None
    carried_time = None
    carried_metric = None

    # If the current short follow-up already names a lab (e.g. "smart_lab"),
    # promote it to explicit lab scope for downstream DB executor calls.
    if not effective_lab and has_lab_reference:
        extracted_current_lab = extract_space_from_question(base_question)
        if extracted_current_lab:
            effective_lab = extracted_current_lab

    if not effective_lab and not has_lab_reference and memory.lab_name:
        effective_lab = memory.lab_name
        carried_lab = memory.lab_name

    if not has_time_window_hint and memory.time_phrase:
        effective_question = f"{effective_question} ({memory.time_phrase})"
        carried_time = memory.time_phrase

    if not has_metric_reference and memory.metric:
        effective_question = f"{effective_question} ({memory.metric})"
        carried_metric = memory.metric
    elif not has_metric_reference and memory.topic_phrase:
        effective_question = f"{effective_question} ({memory.topic_phrase})"
        carried_metric = memory.topic_phrase

    applied = bool(carried_lab or carried_time or carried_metric)
    return effective_question, effective_lab, {
        "applied": applied,
        "carried_lab_name": carried_lab,
        "carried_time_phrase": carried_time,
        "carried_metric": carried_metric,
        "previous_user": memory.previous_user if applied else "",
    }
