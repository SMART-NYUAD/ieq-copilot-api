"""Deterministic routing signals extracted from natural language questions."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

try:
    from query_routing.router_types import QueryScopeClass
except ImportError:
    from .router_types import QueryScopeClass


_METRIC_TOKEN_RE = re.compile(
    r"\b(pm\s*2\.?\s*5|pm2\.?5|pm25|co2|tvoc|voc|temperature|temp|humidity|light|lux|sound|noise|ieq)\b"
)
_TIME_WINDOW_HINT_RE = re.compile(
    r"\b("
    r"this week|last week|this month|last month|today|yesterday|"
    r"last\s+\d+\s+(hour|hours|day|days|week|weeks|month|months)|"
    r"past\s+\d+\s+(hour|hours|day|days|week|weeks|month|months)|"
    r"january|february|march|april|may|june|july|august|september|october|november|december|"
    r"jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec|"
    r"\d{4}-\d{1,2}-\d{1,2}"
    r")\b"
)
_GENERAL_QA_HINTS = (
    "what is",
    "who are you",
    "what are you",
    "what can you do",
    "introduce yourself",
    "tell me about yourself",
    "your name",
    "what does",
    "meaning of",
    "explain",
    "definition",
    "interpret",
    "how should i interpret",
    "how do i interpret",
    "how to interpret",
    "guidance",
    "guideline",
    "recommendation",
    "threshold",
    "measure",
    "how to improve",
    "how can i improve",
)
_KNOWLEDGE_DOMAIN_HINTS = (
    "ieq",
    "air quality",
    "indoor air quality",
    "co2",
    "pm2.5",
    "pm25",
    "tvoc",
    "humidity",
    "ventilation",
)
_IEQ_CONCEPTUAL_HINTS = (
    "warning trend",
    "anomaly",
    "anomalies",
    "drift",
    "spike",
    "threshold breach",
)
_EXPLICIT_LAB_TOKEN_RE = re.compile(r"\b([a-z0-9]+_lab)\b")
_NATURAL_LAB_TOKEN_RE = re.compile(r"\b([a-z0-9]+)\s+lab\b")
_COMPARE_ONE_WORD_RE = re.compile(
    r"\b(?:compare\s+)?([a-z0-9]{3,})\s+(?:vs|versus|and|with)\s+([a-z0-9]{3,})\b"
)
_COMPARISON_HINTS = ("compare", "comparison", "vs", "versus", "higher than", "lower than")
_COMFORT_ASSESSMENT_HINTS = (
    "comfortable",
    "comfort",
    "too hot",
    "too cold",
    "stuffy",
    "fresh air",
    "dry",
    "humid",
)
_DB_SCOPE_HINTS = (
    "average",
    "avg",
    "trend",
    "compare",
    "vs",
    "versus",
    "during",
    "between",
    "over",
    "last ",
    "past ",
)
_CONVERSATION_CONTEXT_MARKER = "\n\nprevious conversation context"
_SOCIAL_IDENTITY_RE = re.compile(
    r"\b(" 
    r"who are you|what are you|what can you do|"
    r"introduce yourself|tell me about yourself|"
    r"your name|are you (?:an? )?(?:ai|assistant|bot)"
    r")\b"
)
_GREETING_ONLY_RE = re.compile(
    r"^\s*(?:hi|hello|hey|yo|good\s+morning|good\s+afternoon|good\s+evening)\W*$"
)
_GENERAL_CONVERSATION_QUESTION_RE = re.compile(
    r"^\s*(?:who|what|when|where|why|how|can|could|would|is|are|do|does|did)\b"
)


def _is_air_quality_query_text(question: str) -> bool:
    q = (question or "").lower()
    issue_hints = ("issue", "issues", "problem", "problems", "anything wrong", "wrong")
    currentness_hints = ("right now", "now", "current", "currently", "latest", "today", "at this moment")
    if any(hint in q for hint in issue_hints) and (
        any(hint in q for hint in currentness_hints)
        or "_lab" in q
        or re.search(r"\b[a-z0-9]+\s+lab\b", q) is not None
    ):
        return True
    if any(hint in q for hint in ("air quality", "indoor air quality", "ieq")):
        return True
    if re.search(r"\bhow\s+(?:is|was)\s+the\s+air\b", q):
        return True
    if re.search(r"\bthe\s+air\b", q) and (
        "_lab" in q or re.search(r"\b[a-z0-9]+\s+lab\b", q) is not None
    ):
        return True
    return False


def _latest_user_question(question: str) -> str:
    """Strip appended conversation transcript from effective question text."""
    text = str(question or "")
    lower_text = text.lower()
    marker_idx = lower_text.find(_CONVERSATION_CONTEXT_MARKER)
    if marker_idx >= 0:
        text = text[:marker_idx]
    return text.strip()


def _is_social_identity_query(question: str) -> bool:
    text = _latest_user_question(question).lower().strip()
    if not text:
        return False
    if _SOCIAL_IDENTITY_RE.search(text):
        return True
    return bool(_GREETING_ONLY_RE.match(text))


def _extract_lab_candidates(question: str, lab_name: Optional[str]) -> list[str]:
    q = _latest_user_question(question).lower()
    candidates: list[str] = []

    def _push(value: str) -> None:
        token = str(value or "").strip().lower()
        token = re.sub(r"[^a-z0-9_\s]", "", token)
        token = re.sub(r"\s+", " ", token).strip()
        if not token:
            return
        for variant in (token, token.replace(" ", "_"), token.replace("_", " ")):
            if variant and variant not in candidates:
                candidates.append(variant)

    if lab_name and str(lab_name).strip():
        _push(str(lab_name))

    for match in _EXPLICIT_LAB_TOKEN_RE.finditer(q):
        _push(match.group(1))
    for match in _NATURAL_LAB_TOKEN_RE.finditer(q):
        _push(f"{match.group(1)}_lab")
        _push(match.group(1))
    for match in _COMPARE_ONE_WORD_RE.finditer(q):
        left = match.group(1)
        right = match.group(2)
        if left in {"this", "that", "with"} or right in {"this", "that", "with"}:
            continue
        _push(left)
        _push(f"{left}_lab")
        _push(right)
        _push(f"{right}_lab")

    return candidates[:8]


def extract_query_signals(question: str, lab_name: Optional[str] = None) -> Dict[str, Any]:
    latest_question = _latest_user_question(question)
    q = latest_question.lower()
    lab_candidates = _extract_lab_candidates(question=question, lab_name=lab_name)
    has_lab_reference = len(lab_candidates) > 0 or (
        "_lab" in q or re.search(r"\b[a-z0-9]+\s+lab\b", q) is not None
    )
    has_time_window_hint = bool(_TIME_WINDOW_HINT_RE.search(q))
    has_metric_reference = bool(_METRIC_TOKEN_RE.search(q))
    has_general_qa_phrase = any(hint in q for hint in _GENERAL_QA_HINTS)
    is_social_identity_query = _is_social_identity_query(latest_question)
    has_conceptual_ieq_term = any(hint in q for hint in _IEQ_CONCEPTUAL_HINTS)
    has_knowledge_domain = any(hint in q for hint in _KNOWLEDGE_DOMAIN_HINTS) or (
        has_conceptual_ieq_term and has_general_qa_phrase
    )
    is_air_assessment_phrase = _is_air_quality_query_text(latest_question)
    is_general_knowledge_question = (
        has_knowledge_domain and has_general_qa_phrase and not has_time_window_hint
    )
    is_comfort_assessment_phrase = any(hint in q for hint in _COMFORT_ASSESSMENT_HINTS)
    has_db_scope_phrase = any(hint in q for hint in _DB_SCOPE_HINTS) or any(
        hint in q for hint in _COMPARISON_HINTS
    )
    has_domain_anchor = (
        has_metric_reference
        or has_lab_reference
        or has_knowledge_domain
        or has_conceptual_ieq_term
        or is_air_assessment_phrase
        or is_comfort_assessment_phrase
    )
    is_general_conversation_question = (
        not has_domain_anchor and bool(_GENERAL_CONVERSATION_QUESTION_RE.search(q))
    )
    if not has_domain_anchor:
        if (
            has_time_window_hint
            or has_general_qa_phrase
            or is_social_identity_query
            or is_general_conversation_question
        ):
            query_scope_class = QueryScopeClass.NON_DOMAIN.value
        else:
            query_scope_class = QueryScopeClass.AMBIGUOUS.value
    else:
        has_db_scope_evidence = (
            has_metric_reference
            or has_lab_reference
            or is_air_assessment_phrase
            or is_comfort_assessment_phrase
            or (has_time_window_hint and has_domain_anchor)
            or has_db_scope_phrase
        )
        if has_db_scope_evidence and not is_general_knowledge_question:
            query_scope_class = QueryScopeClass.DOMAIN.value
        else:
            query_scope_class = QueryScopeClass.AMBIGUOUS.value

    asks_for_db_facts = query_scope_class == QueryScopeClass.DOMAIN.value
    explicit_scope_intent = (
        has_time_window_hint
        or any(hint in q for hint in _COMPARISON_HINTS)
        or re.search(r"\bin\s+[a-z0-9_ ]+\b", q) is not None
    )
    if query_scope_class == QueryScopeClass.AMBIGUOUS.value and explicit_scope_intent and has_domain_anchor:
        asks_for_db_facts = True
    return {
        "has_lab_reference": has_lab_reference,
        "has_time_window_hint": has_time_window_hint,
        "has_metric_reference": has_metric_reference,
        "has_db_scope_phrase": has_db_scope_phrase,
        "is_air_assessment_phrase": is_air_assessment_phrase,
        "is_general_knowledge_question": is_general_knowledge_question,
        "is_social_identity_query": is_social_identity_query,
        "is_general_conversation_question": is_general_conversation_question,
        "is_comfort_assessment_phrase": is_comfort_assessment_phrase,
        "asks_for_db_facts": asks_for_db_facts,
        "query_scope_class": query_scope_class,
        "has_domain_anchor": has_domain_anchor,
        "lab_candidates": lab_candidates,
    }
