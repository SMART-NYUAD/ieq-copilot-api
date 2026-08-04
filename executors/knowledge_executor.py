"""Knowledge card QA executor: semantic search + Ollama LLM answering."""

import json
from datetime import datetime, timezone
import logging
import os
from threading import Lock
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import httpx

from core_settings import (
    display_timezone,
    ollama_base_url,
    ollama_model,
    ollama_temperature,
    ollama_thinking,
    ollama_timeout_seconds,
)
from ollama_helpers import (
    build_prompt_text_from_messages,
    extract_generate_chunk,
    generate_ollama_text,
)
from executors import sse
from storage.embeddings import embed_query
from storage.postgres_client import get_cursor
from storage.sql_queries import ENV_KNOWLEDGE_QUERY_SEMANTIC_SQL
from executors.db_support import threshold_assessment
from prompting.db_prompts import THRESHOLD_VERDICTS
from prompting.roles import ROLE_DEFAULT, role_wants_compliance_detail
from storage.guideline_store import get_thresholds_for_metrics
from prompting.shared_prompts import build_grounded_context_sections, get_shared_prompt_template
from evidence.citation_processor import build_numbered_sources_block, process_answer_citations
from storage.guideline_store import search_guideline_records, wants_guideline_detail


_log = logging.getLogger(__name__)

_NO_LLM_KNOWLEDGE_ANSWER = (
    "I can't reach the language model right now, and I don't have a knowledge card "
    "that covers this question. Please try again in a moment."
)
_DETERMINISTIC_ANSWER_MAX_CHARS = 600

CARD_TOOL_RESPONSE_DIRECTIVE = f"""
You are answering from card-based retrieval context.
- Follow the shared presentation style from the system prompt.
- Use the retrieved cards as grounding for key evidence.
- If the question is about risks, lead with risk level and main drivers.
- Whether to volunteer recommendations that were not asked for is decided by the audience
  block under `Domain style:` in the system prompt. Do not restate a rule about
  recommendations back to the reader.
- A definition answer explains the concept first. When a live reading for that metric is in
  context, you may add what it currently shows — but the verdict on that reading comes from
  the Threshold Assessment below, never from your own judgement of whether the number looks
  normal.

{THRESHOLD_VERDICTS}
""".strip()

_KNOWLEDGE_CONTEXT_CACHE_LOCK = Lock()
_KNOWLEDGE_CONTEXT_CACHE: Dict[Tuple[str, int, Optional[str]], Tuple[float, Dict[str, Any]]] = {}


def _knowledge_context_cache_ttl_seconds() -> float:
    raw = str(os.getenv("KNOWLEDGE_CONTEXT_CACHE_TTL_SECONDS", "30")).strip()
    try:
        value = float(raw)
    except ValueError:
        value = 30.0
    return max(0.0, value)


def _knowledge_context_cache_max_entries() -> int:
    raw = str(os.getenv("KNOWLEDGE_CONTEXT_CACHE_MAX_ENTRIES", "256")).strip()
    try:
        value = int(raw)
    except ValueError:
        value = 256
    return max(16, value)


def _serialize_timestamp_gmt4(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return text
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(display_timezone()).isoformat()


def _is_explanation_query(question: str) -> bool:
    """Whether the user is asking what something IS, rather than how it is right now.

    The list is deliberately broader than it looks: it previously missed "explain" and
    "what are", so *"explain PM2.5"* fell through to the status branch, where explanation
    cards are penalised and interpretation cards rewarded — the reranker actively pushed
    away the one card type that answers a definition question. See tests/retrieval_eval.py
    group `definition`.
    """
    q = (question or "").lower()
    hints = (
        "what is", "what are", "what does", "what do you mean", "define", "definition",
        "mean by", "explain", "tell me about", "meaning of", "stand for", "stands for",
    )
    return any(hint in q for hint in hints)


def _is_guardrail_query(question: str) -> bool:
    """Whether the user is asking about health risk OR the limits of what a metric shows.

    Caveat cards answer both, and both are penalised by the default branch. The second
    kind was missing: "does CO2 tell me everything about air quality?" is exactly what the
    CO2-is-a-ventilation-proxy caveat exists to answer, but with only health-risk hints it
    fell through to the status branch and the caveat took a penalty instead of a boost.
    """
    q = (question or "").lower()
    hints = (
        # health / safety framing
        "health risk", "safe", "dangerous", "medical", "diagnosis", "harmful", "unhealthy",
        # "what does this metric NOT tell me" framing
        "tell me everything", "everything about", "full picture", "complete picture",
        "enough to", "only indicator", "limitation", "caveat", "does that mean",
        "reliable", "accurate enough", "the whole story",
    )
    return any(hint in q for hint in hints)


# Card-type nudges, added to a cosine similarity. They MUST stay small relative to the
# spread of real similarities, which on this corpus runs roughly 0.45-0.65 — a span of
# about 0.2. The original weights went up to +0.7, several times that span, so card_type
# decided the ranking outright and the semantic score was decoration: a barely-related
# interpretation card outranked an exactly-on-topic explanation card by a fixed +0.35.
# These are sized to break near-ties only. If you raise one above ~0.10, re-run
# tests/retrieval_eval.py — you are probably overriding the embedding again.
_MAX_PRIORITY_NUDGE = 0.10


def _knowledge_card_priority(question: str, card_type: str) -> float:
    if _is_guardrail_query(question):
        if card_type == "caveat":
            return 0.10
        if card_type == "rule":
            return 0.04
        return -0.02
    if _is_explanation_query(question):
        if card_type == "explanation":
            return 0.10
        if card_type == "ieq_subindex":
            return 0.06
        if card_type == "caveat":
            return -0.04
        return 0.0
    if card_type == "caveat":
        return -0.05
    return 0.06 if card_type in {"interpretation", "rule"} else 0.0


def search_knowledge_cards(question: str, k: int = 4) -> List[Dict[str, Any]]:
    """Search static knowledge cards using semantic similarity with light type-aware reranking."""
    query_embedding = embed_query(question)
    if not query_embedding:
        return []

    # Fetch generously and let the reranker choose: the corpus is small enough that the
    # scan is exact and cheap, and a narrow candidate set was part of the original bug —
    # no reranking weight can recover a card that was never fetched.
    fetch_k = max(12, min(40, k * 6))
    try:
        with get_cursor(real_dict=True) as cur:
            # No ivfflat probe tuning: migration 004 removed the approximate index, so this
            # is an exact nearest-neighbour scan. Reintroducing an index means reintroducing
            # a probe setting, and the recall trap that came with it.
            cur.execute(ENV_KNOWLEDGE_QUERY_SEMANTIC_SQL, (query_embedding, query_embedding, fetch_k))
            rows = [dict(row) for row in cur.fetchall()]
    except Exception:
        # An unreachable card store means ungrounded answers, not an error response —
        # log it so that degradation is visible.
        _log.exception("knowledge card search failed for question=%r", (question or "")[:120])
        return []

    reranked = []
    for row in rows:
        semantic_score = float(row.get("distance") or 0.0)
        priority = _knowledge_card_priority(question, str(row.get("card_type") or ""))
        reranked.append((semantic_score + priority, row))
    reranked.sort(key=lambda item: item[0], reverse=True)
    return [row for _, row in reranked[:k]]


def _knowledge_context_cache_key(user_question: str, k: int, space: Optional[str]) -> Tuple[str, int, Optional[str]]:
    normalized_question = str(user_question or "").strip()
    normalized_space = (space or "").strip().lower() or None
    effective_k = max(3, min(5, int(k or 5)))
    return normalized_question, effective_k, normalized_space


def _prune_knowledge_context_cache(now: float) -> None:
    expired = [key for key, (expires_at, _) in _KNOWLEDGE_CONTEXT_CACHE.items() if expires_at <= now]
    for key in expired:
        _KNOWLEDGE_CONTEXT_CACHE.pop(key, None)
    max_entries = _knowledge_context_cache_max_entries()
    while len(_KNOWLEDGE_CONTEXT_CACHE) > max_entries:
        oldest_key = min(_KNOWLEDGE_CONTEXT_CACHE, key=lambda key: _KNOWLEDGE_CONTEXT_CACHE[key][0])
        _KNOWLEDGE_CONTEXT_CACHE.pop(oldest_key, None)


def _split_knowledge_cards(cards: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    guardrails = []
    knowledge = []
    for card in cards:
        compact = {
            "card_type": card.get("card_type"),
            "topic": card.get("topic"),
            "title": card.get("title"),
            "summary": card.get("summary"),
            "content": card.get("content"),
            "severity_level": card.get("severity_level"),
            "source_label": card.get("source_label"),
            "source_url_key": card.get("source_url_key"),
        }
        if card.get("card_type") == "caveat":
            guardrails.append(compact)
        else:
            knowledge.append(compact)
    return knowledge, guardrails


def build_card_grounded_context(
    cards: List[Dict[str, Any]],
    knowledge_cards: List[Dict[str, Any]],
    allow_general_knowledge: bool = False,
    guideline_records: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Format measured room facts and knowledge guidance into labeled sections."""
    measured_room_facts = []
    for card in cards:
        measured_room_facts.append(
            {
                "space": card.get("space"),
                "window_start": _serialize_timestamp_gmt4(card.get("window_start")),
                "window_end": _serialize_timestamp_gmt4(card.get("window_end")),
                "overall_air_label": card.get("overall_air_label"),
                "summary_text": card.get("summary_text"),
                "distance": card.get("distance"),
            }
        )
    interpretation_cards, guardrails = _split_knowledge_cards(knowledge_cards)
    return build_grounded_context_sections(
        measured_room_facts=measured_room_facts,
        backend_semantic_state=None,
        knowledge_cards=interpretation_cards,
        communication_guardrails=guardrails,
        guideline_records=guideline_records,
        allow_general_knowledge=allow_general_knowledge,
    )


def _build_knowledge_context_uncached(
    user_question: str,
    k: int = 5,
    space: Optional[str] = None,
    guideline_records: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    knowledge_cards = search_knowledge_cards(user_question, k=max(3, min(5, k)))
    grounded_context = build_card_grounded_context(
        [],
        knowledge_cards,
        allow_general_knowledge=True,
        guideline_records=guideline_records,
    )
    return {
        "knowledge_cards": knowledge_cards,
        "grounded_context": grounded_context,
    }


def _build_knowledge_context(
    user_question: str,
    k: int = 5,
    space: Optional[str] = None,
    guideline_records: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    key = _knowledge_context_cache_key(user_question=user_question, k=k, space=space)
    ttl_seconds = _knowledge_context_cache_ttl_seconds()
    if ttl_seconds <= 0:
        return _build_knowledge_context_uncached(
            user_question=user_question, k=k, space=space, guideline_records=guideline_records
        )

    now = time.monotonic()
    with _KNOWLEDGE_CONTEXT_CACHE_LOCK:
        cached = _KNOWLEDGE_CONTEXT_CACHE.get(key)
        if cached and cached[0] > now:
            return cached[1]

    context = _build_knowledge_context_uncached(
        user_question=user_question, k=k, space=space, guideline_records=guideline_records
    )
    expires_at = now + ttl_seconds
    with _KNOWLEDGE_CONTEXT_CACHE_LOCK:
        _KNOWLEDGE_CONTEXT_CACHE[key] = (expires_at, context)
        _prune_knowledge_context_cache(now=now)
    return context


def _deterministic_knowledge_answer(knowledge_cards: List[Dict[str, Any]]) -> str:
    """Answer built from retrieved cards alone, for when the LLM is unreachable.

    Mirrors the deterministic fallbacks in the DB, IFC, and sensor executors: the
    text is quoted from the highest-ranked retrieved card, never generated, so an
    LLM outage degrades to a grounded answer instead of an empty response.
    """
    for card in knowledge_cards or []:
        body = str(card.get("summary") or card.get("content") or "").strip()
        if not body:
            continue
        title = str(card.get("title") or "").strip()
        answer = f"**{title}** — {body}" if title else body
        if len(answer) > _DETERMINISTIC_ANSWER_MAX_CHARS:
            answer = answer[:_DETERMINISTIC_ANSWER_MAX_CHARS].rstrip() + "…"
        return answer
    return _NO_LLM_KNOWLEDGE_ANSWER


def _readings_from_live_data(live_sensor_data: Any) -> Dict[str, Any]:
    """Metric -> latest value from the live payload handed over by the orchestrator.

    Row-shape normalisation lives in threshold_assessment.readings_from_rows so the DB and
    knowledge paths cannot disagree about what a reading row means.
    """
    if not isinstance(live_sensor_data, dict):
        return {}
    return threshold_assessment.readings_from_rows(
        live_sensor_data.get("rows") or [], live_sensor_data.get("metric")
    )


def _merge_guideline_records(
    primary: List[Dict[str, Any]], extra: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Union two record lists, keeping the first occurrence of each id."""
    merged: List[Dict[str, Any]] = []
    seen: set = set()
    for record in list(primary or []) + list(extra or []):
        key = record.get("id") or (record.get("source_key"), record.get("metric"))
        if key in seen:
            continue
        seen.add(key)
        merged.append(record)
    return merged


def build_knowledge_grounding(
    *,
    user_question: str,
    knowledge_cards: List[Dict[str, Any]],
    live_sensor_data: Any,
    guideline_records: List[Dict[str, Any]],
    role: str = ROLE_DEFAULT,
) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Assemble the grounded context for a knowledge answer.

    Returns ``(context_text, indexed_sources, effective_guideline_records)``.

    Built once and shared by the sync and streaming paths, which previously assembled it
    separately and had already drifted — the sync path passed the searched guideline records
    into the card context while the stream passed an empty list.

    The threshold work this restores: when a live reading is in context, the knowledge path
    now fetches the guideline records for exactly the metrics on screen and renders the same
    computed ``## Threshold Assessment`` section the DB path uses. Without it the model was
    handed a number and no limit to compare it against, and judged the reading itself —
    producing "TVOC is 0.08 ppm, within typical indoor ranges and considered acceptable" one
    turn after the DB path had correctly reported the same 0.08 ppm as exceeding the WHO
    guideline of 0.061 ppm. Same conversation, same reading, opposite verdicts.
    """
    readings = _readings_from_live_data(live_sensor_data)
    effective_records = list(guideline_records or [])
    if readings:
        # Deterministic lookup by metric name — the same call the DB path makes. Semantic
        # guideline search only fires when the question is *about* standards, so without
        # this a definition question had no citation sources at all and therefore nothing
        # for the assessment to compare against.
        effective_records = _merge_guideline_records(
            effective_records, get_thresholds_for_metrics(sorted(readings))
        )

    numbered_sources_block, indexed_sources = build_numbered_sources_block(effective_records)
    assessment = (
        threshold_assessment.build_assessment_section(
            readings, indexed_sources, compliance_detail=role_wants_compliance_detail(role)
        )
        if readings
        else ""
    )
    grounded = build_grounded_context_sections(
        measured_room_facts=live_sensor_data if live_sensor_data is not None else [],
        backend_semantic_state=None,
        knowledge_cards=knowledge_cards,
        numbered_sources_block=numbered_sources_block,
        allow_general_knowledge=True,
        threshold_assessment=assessment,
    )
    return grounded, indexed_sources, effective_records


def answer_env_question_with_metadata(
    user_question: str,
    k: int = 5,
    space: Optional[str] = None,
    guideline_records: Optional[List[Dict[str, Any]]] = None,
    live_sensor_data: Optional[Any] = None,
    role: str = ROLE_DEFAULT,
) -> Dict[str, Any]:
    effective_guideline_records = list(guideline_records or [])
    if wants_guideline_detail(user_question):
        searched_guidelines = search_guideline_records(question=user_question, k=3)
        if searched_guidelines:
            effective_guideline_records = searched_guidelines

    context = _build_knowledge_context(
        user_question=user_question,
        k=k,
        space=space,
        guideline_records=effective_guideline_records,
    )
    grounded_context, indexed_sources, effective_guideline_records = build_knowledge_grounding(
        user_question=user_question,
        knowledge_cards=context.get("knowledge_cards", []),
        live_sensor_data=live_sensor_data,
        guideline_records=effective_guideline_records,
        role=role,
    )
    context_label = (
        "Live sensor readings with knowledge grounding"
        if live_sensor_data is not None
        else "Measured room facts with knowledge grounding"
    )
    prompt_template = get_shared_prompt_template(
        response_directive=CARD_TOOL_RESPONSE_DIRECTIVE, role=role
    )
    messages = prompt_template.format_messages(
        question=user_question,
        context_label=context_label,
        context_data=grounded_context,
    )
    prompt_text = build_prompt_text_from_messages(messages)
    knowledge_cards = context.get("knowledge_cards") or []
    try:
        answer = generate_ollama_text(prompt_text, temperature=ollama_temperature())
    except Exception:
        answer = ""
    llm_used = bool(answer.strip())
    if not llm_used:
        # An unreachable answer model must not surface as a 500 — fall back to the
        # retrieved cards, the same way the DB path falls back to its row summary.
        answer = _deterministic_knowledge_answer(knowledge_cards)
    resolved_answer, footnotes = process_answer_citations(
        answer_text=answer,
        guideline_records=effective_guideline_records,
        indexed_sources=indexed_sources,
    )
    return {
        "answer": resolved_answer,
        "footnotes": footnotes,
        "indexed_sources": indexed_sources,
        "cards_retrieved": int(len(knowledge_cards)),
        "knowledge_cards_retrieved": int(len(knowledge_cards)),
        "guideline_records": effective_guideline_records,
        "llm_used": llm_used,
    }


async def stream_knowledge_tokens(
    user_question: str,
    k: int = 5,
    space: Optional[str] = None,
    guideline_records: Optional[List[Dict[str, Any]]] = None,
    live_sensor_data: Optional[Any] = None,
    role: str = ROLE_DEFAULT,
) -> AsyncIterator[str]:
    effective_guideline_records = list(guideline_records or [])
    if wants_guideline_detail(user_question):
        searched_guidelines = search_guideline_records(question=user_question, k=3)
        if searched_guidelines:
            effective_guideline_records = searched_guidelines
    context = _build_knowledge_context(
        user_question=user_question,
        k=k,
        space=space,
        guideline_records=effective_guideline_records,
    )
    grounded_context, indexed_sources, effective_guideline_records = build_knowledge_grounding(
        user_question=user_question,
        knowledge_cards=context.get("knowledge_cards", []),
        live_sensor_data=live_sensor_data,
        guideline_records=effective_guideline_records,
        role=role,
    )
    context_label = (
        "Live sensor readings with knowledge grounding"
        if live_sensor_data is not None
        else "Measured room facts with knowledge grounding"
    )
    prompt_template = get_shared_prompt_template(
        response_directive=CARD_TOOL_RESPONSE_DIRECTIVE, role=role
    )
    messages = prompt_template.format_messages(
        question=user_question,
        context_label=context_label,
        context_data=grounded_context,
    )
    prompt_text = build_prompt_text_from_messages(messages)

    ollama_payload = {
        "model": ollama_model(),
        "prompt": prompt_text,
        "stream": True,
        "think": ollama_thinking(),
        "temperature": ollama_temperature(),
    }

    emitted: List[str] = []
    try:
        async with httpx.AsyncClient(timeout=ollama_timeout_seconds()) as client:
            async with client.stream("POST", f"{ollama_base_url()}/api/generate", json=ollama_payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    response_text = extract_generate_chunk(event)
                    if response_text:
                        emitted.append(response_text)
                        yield sse.token_event(response_text)
    except Exception:
        pass

    if not emitted:
        # LLM unreachable — emit the card-grounded answer so the stream is never empty.
        yield sse.token_event(_deterministic_knowledge_answer(context.get("knowledge_cards") or []))

    yield sse.sources_event_for_answer(emitted, effective_guideline_records, indexed_sources)
    yield sse.done_event()
