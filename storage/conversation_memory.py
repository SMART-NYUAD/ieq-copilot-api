"""Structured routing memory extracted from compact conversation context."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict, Optional, Tuple

from executors.db_support.query_parsing import extract_space_from_question


_METRIC_KEYWORDS = (
    "ieq",
    "co2",
    "pm2.5",
    "pm25",
    "voc",
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
_TOPIC_PHRASES = (
    "air quality",
    "indoor air quality",
    "ieq",
    "comfort",
    "comfortable",
)
_DEFINITIONAL_PHRASES = (
    "what does",
    "what is",
    "what are",
    "what do",
    "meaning of",
    "define",
    "definition",
    "explain",
    "how does",
    "how do",
    "what do you mean",
    "tell me about",
)
# Questions with an explicit analytical intent carry their own metric scope — don't
# inject a prior metric that would mislead the LLM router (e.g. appending "(ieq)" to
# an anomaly question causes it to route as a current-status IEQ query instead).
_ANALYTICAL_INTENT_RE = re.compile(
    r"\b(anomal|spike|outlier|unusual|abnormal|deviation|"
    r"trend|average|avg|compare|comparison|versus|vs)\b"
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
    # Generic determiner-introduced windows ("for the week", "over the past
    # month") count as the question naming its own time scope, so the prior
    # turn's window is not carried over it.
    g = re.search(
        r"\b(?:the|a|an|this|that|over|for|past|current|recent|whole|entire|each|every|per)\s+"
        r"(?:the\s+|a\s+|past\s+|last\s+|current\s+|whole\s+|entire\s+|few\s+)?"
        r"(?:hour|day|week|month|year)s?\b",
        q,
    )
    if g:
        return g.group(0)
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


def _latest_metric_and_topic(user_lines: list[str]) -> tuple[Optional[str], Optional[str]]:
    """Return (metric, topic) from the single most-recent turn that mentions either.

    Scanning independently would let an older explicit metric (e.g. "temperature"
    from turn 2) shadow a more recently discussed topic phrase (e.g. "air quality"
    from turn 4).  By scanning once in reverse we always surface the newest context.
    """
    for candidate in reversed(user_lines):
        metrics = _extract_requested_metrics(candidate)
        topic = _extract_topic_phrase(candidate)
        if metrics or topic:
            return (metrics[0] if metrics else None), topic
    return None, None


def _is_definitional_question(question: str) -> bool:
    q = str(question or "").strip().lower()
    return any(phrase in q for phrase in _DEFINITIONAL_PHRASES)


def compute_question_signals(question: str) -> Dict[str, Any]:
    """Return whether the current question already names a lab, metric, or time window."""
    try:
        from executors.db_support.query_parsing import extract_space_from_question, extract_metric_aliases
    except ImportError:
        return {}
    q = str(question or "")
    return {
        "has_lab_reference": bool(extract_space_from_question(q)),
        "has_metric_reference": bool(_extract_requested_metrics(q)),
        "has_time_window_hint": bool(_requested_time_phrase(q)),
        "has_topic_reference": bool(_extract_topic_phrase(q)),
    }


def extract_routing_memory(conversation_context: str, current_signals: Dict[str, Any]) -> RoutingMemory:
    """Extract only lab/metric/time from prior user turns."""
    _ = current_signals  # reserved for future weighting logic
    user_lines = _context_user_lines(conversation_context)
    if not user_lines:
        return RoutingMemory()
    previous_user = str(user_lines[-1] or "").strip()
    previous_lab = _latest_value_from_user_lines(user_lines, extract_space_from_question)
    previous_time = _latest_value_from_user_lines(user_lines, _requested_time_phrase)
    previous_metric, previous_topic = _latest_metric_and_topic(user_lines)
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
    """Fill missing lab/metric/time from prior turns when the question omits them.

    Returns the clean (unmutated) question alongside resolved lab and carry-over
    slots.  Callers must NOT append the carry-over text to the question string —
    they read ``carried_metric`` and ``carried_time_phrase`` from the returned
    dict and pass them as structured data to downstream components.
    """
    base_question = str(question or "").strip()
    signals = dict(current_signals or {})
    if not base_question:
        return base_question, (lab_name or "").strip() or None, {"applied": False}

    has_lab_reference = bool(signals.get("has_lab_reference"))
    has_time_window_hint = bool(signals.get("has_time_window_hint"))
    has_metric_reference = bool(signals.get("has_metric_reference"))
    has_topic_reference = bool(signals.get("has_topic_reference"))

    # effective_question stays clean — no appended carry-over text
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

    # Skip time carry when:
    # - definitional question (semantics don't depend on a window)
    # - user named a broad topic (air quality, IEQ, comfort) — topic change → fresh window
    skip_time_phrase = (
        _is_definitional_question(base_question)
        or has_topic_reference
    )
    # Do not inject a prior single-metric scope when the user asked a broad topic
    # (e.g. air quality) or an analytical question with its own scope.
    skip_metric_carry = (
        _is_definitional_question(base_question)
        or has_topic_reference
        or bool(_ANALYTICAL_INTENT_RE.search(base_question.lower()))
    )

    if not skip_time_phrase and not has_time_window_hint and memory.time_phrase:
        carried_time = memory.time_phrase

    if not skip_metric_carry and not has_metric_reference and memory.metric:
        carried_metric = memory.metric
    elif not skip_metric_carry and not has_metric_reference and memory.topic_phrase:
        carried_metric = memory.topic_phrase

    applied = bool(carried_lab or carried_time or carried_metric)
    return effective_question, effective_lab, {
        "applied": applied,
        "carried_lab_name": carried_lab,
        "carried_time_phrase": carried_time,
        "carried_metric": carried_metric,
        "previous_user": memory.previous_user if applied else "",
    }
