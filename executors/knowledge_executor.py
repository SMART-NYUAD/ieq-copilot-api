"""Knowledge card QA executor: semantic search + Ollama LLM answering."""

import json
from datetime import datetime, timedelta, timezone
import os
from threading import Lock
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import httpx

from core_settings import ollama_base_url, ollama_model, ollama_temperature, ollama_timeout_seconds
from storage.embeddings import embed_texts
from storage.postgres_client import get_cursor
from storage.sql_queries import ENV_KNOWLEDGE_QUERY_SEMANTIC_SQL
from prompting.shared_prompts import build_grounded_context_sections, get_shared_prompt_template
from http_schemas import validate_tool_evidence
from evidence.citation_processor import build_numbered_sources_block, process_answer_citations
from storage.guideline_store import search_guideline_records, wants_guideline_detail


CARD_TOOL_RESPONSE_DIRECTIVE = """
You are answering from card-based retrieval context.
- Start with exactly one short verdict sentence answering the question directly.
- Then provide at most 3 short bullets with key grounded evidence.
- If the question is about risks, lead with risk level and main drivers.
- Do not provide recommendations unless the user explicitly asks for recommendations or next steps.
- Do not include long background/context unless the user explicitly asks "why", "details", or "full report".
""".strip()

_TARGET_TZ = timezone(timedelta(hours=4))
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
    return parsed.astimezone(_TARGET_TZ).isoformat()


def _is_explanation_query(question: str) -> bool:
    q = (question or "").lower()
    hints = ("what is", "what does", "what do you mean", "define", "definition", "mean by")
    return any(hint in q for hint in hints)


def _is_guardrail_query(question: str) -> bool:
    q = (question or "").lower()
    hints = ("health risk", "safe", "dangerous", "medical", "diagnosis", "harmful", "unhealthy")
    return any(hint in q for hint in hints)


def _knowledge_card_priority(question: str, card_type: str) -> float:
    if _is_guardrail_query(question):
        if card_type == "caveat":
            return 0.7
        if card_type == "rule":
            return 0.2
        return -0.05
    if card_type == "caveat":
        return -0.25
    if _is_explanation_query(question):
        return 0.5 if card_type == "explanation" else 0.15 if card_type == "rule" else 0.0
    return 0.45 if card_type in {"interpretation", "rule"} else 0.1


def search_knowledge_cards(question: str, k: int = 4) -> List[Dict[str, Any]]:
    """Search static knowledge cards using semantic similarity with light type-aware reranking."""
    embeddings = embed_texts([question])
    if not embeddings:
        return []

    query_embedding = embeddings[0]
    fetch_k = max(6, min(20, k * 3))
    try:
        with get_cursor(real_dict=True) as cur:
            cur.execute("SET LOCAL ivfflat.probes = %s", (max(10, fetch_k),))
            cur.execute(ENV_KNOWLEDGE_QUERY_SEMANTIC_SQL, (query_embedding, query_embedding, fetch_k))
            rows = [dict(row) for row in cur.fetchall()]
    except Exception as e:
        print(f"[ERROR] Knowledge card search failed: {e}")
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


def _coerce_chunk_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "".join(parts)
    if isinstance(value, dict):
        return str(value.get("text", ""))
    return str(value)


def _build_prompt_text_from_messages(messages: List[Any]) -> str:
    """Convert LangChain-style message objects to a plain prompt string."""
    prompt_parts = []
    for message in messages:
        role = getattr(message, "type", "user").upper()
        content = _coerce_chunk_text(getattr(message, "content", ""))
        prompt_parts.append(f"{role}:\n{content}")
    return "\n\n".join(prompt_parts)


def _generate_ollama_text(prompt_text: str, *, temperature: float) -> str:
    payload: Dict[str, Any] = {
        "model": ollama_model(),
        "prompt": prompt_text,
        "stream": False,
        "temperature": temperature,
    }

    with httpx.Client(timeout=ollama_timeout_seconds()) as client:
        response = client.post(f"{ollama_base_url()}/api/generate", json=payload)
        response.raise_for_status()
        event = response.json()

    return _coerce_chunk_text(event.get("response"))


def get_knowledge_context_stats(user_question: str, k: int = 5, space: Optional[str] = None) -> Dict[str, Any]:
    context = _build_knowledge_context(user_question=user_question, k=k, space=space)
    knowledge_cards = context.get("knowledge_cards") or []
    return {
        "cards_retrieved": int(len(knowledge_cards)),
        "knowledge_cards_retrieved": int(len(knowledge_cards)),
    }


def get_guideline_records_for_question(user_question: str, k: int = 3) -> List[Dict[str, Any]]:
    if not wants_guideline_detail(user_question):
        return []
    return search_guideline_records(question=user_question, k=max(1, int(k or 3)))


def answer_env_question_with_metadata(
    user_question: str,
    k: int = 5,
    space: Optional[str] = None,
    guideline_records: Optional[List[Dict[str, Any]]] = None,
    live_sensor_data: Optional[Any] = None,
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
    numbered_sources_block, indexed_sources = build_numbered_sources_block(effective_guideline_records)
    grounded_context = build_grounded_context_sections(
        measured_room_facts=live_sensor_data if live_sensor_data is not None else [],
        backend_semantic_state=None,
        knowledge_cards=context.get("knowledge_cards", []),
        numbered_sources_block=numbered_sources_block,
        allow_general_knowledge=True,
    )
    context_label = (
        "Live sensor readings with knowledge grounding"
        if live_sensor_data is not None
        else "Measured room facts with knowledge grounding"
    )
    prompt_template = get_shared_prompt_template(response_directive=CARD_TOOL_RESPONSE_DIRECTIVE)
    messages = prompt_template.format_messages(
        question=user_question,
        context_label=context_label,
        context_data=grounded_context,
    )
    prompt_text = _build_prompt_text_from_messages(messages)
    answer = _generate_ollama_text(prompt_text, temperature=ollama_temperature())
    resolved_answer, footnotes = process_answer_citations(
        answer_text=answer,
        guideline_records=effective_guideline_records,
        indexed_sources=indexed_sources,
    )
    knowledge_cards = context.get("knowledge_cards") or []
    evidence_sources = [
        {
            "source_kind": "knowledge_card",
            "table": "env_knowledge_cards",
            "operation": "semantic_retrieval",
            "metric": None,
            "source_label": card.get("source_label"),
            "topic": card.get("topic"),
            "title": card.get("title"),
            "details": {"card_type": card.get("card_type")},
        }
        for card in knowledge_cards
    ]
    evidence = validate_tool_evidence(
        {
            "evidence_kind": "knowledge_qa",
            "intent": "definition_explanation",
            "strategy": "direct",
            "metric_aliases": [],
            "resolved_scope": space,
            "resolved_time_window": None,
            "provenance_sources": evidence_sources,
            "confidence_notes": [],
            "recommendation_allowed": True,
        }
    )
    return {
        "answer": resolved_answer,
        "footnotes": footnotes,
        "indexed_sources": indexed_sources,
        "cards_retrieved": int(len(knowledge_cards)),
        "knowledge_cards_retrieved": int(len(knowledge_cards)),
        "guideline_records": effective_guideline_records,
        "evidence": evidence,
    }


async def stream_knowledge_tokens(
    user_question: str,
    k: int = 5,
    space: Optional[str] = None,
    guideline_records: Optional[List[Dict[str, Any]]] = None,
    live_sensor_data: Optional[Any] = None,
) -> AsyncIterator[str]:
    effective_guideline_records = list(guideline_records or [])
    if wants_guideline_detail(user_question):
        searched_guidelines = search_guideline_records(question=user_question, k=3)
        if searched_guidelines:
            effective_guideline_records = searched_guidelines
    numbered_sources_block, _ = build_numbered_sources_block(effective_guideline_records)

    context = _build_knowledge_context(
        user_question=user_question,
        k=k,
        space=space,
        guideline_records=[],
    )
    grounded_context = build_grounded_context_sections(
        measured_room_facts=live_sensor_data if live_sensor_data is not None else [],
        backend_semantic_state=None,
        knowledge_cards=context.get("knowledge_cards", []),
        numbered_sources_block=numbered_sources_block,
        allow_general_knowledge=True,
    )
    context_label = (
        "Live sensor readings with knowledge grounding"
        if live_sensor_data is not None
        else "Measured room facts with knowledge grounding"
    )
    prompt_template = get_shared_prompt_template(response_directive=CARD_TOOL_RESPONSE_DIRECTIVE)
    messages = prompt_template.format_messages(
        question=user_question,
        context_label=context_label,
        context_data=grounded_context,
    )
    prompt_text = _build_prompt_text_from_messages(messages)

    ollama_payload = {
        "model": ollama_model(),
        "prompt": prompt_text,
        "stream": True,
        "temperature": ollama_temperature(),
    }

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

                    response_text = _coerce_chunk_text(event.get("response"))
                    if response_text:
                        yield f"data: {json.dumps({'event': 'token', 'text': response_text})}\n\n"
    except Exception:
        pass

    yield f"data: {json.dumps({'event': 'done'})}\n\n"


async def stream_answer_env_question(
    user_question: str,
    k: int = 5,
    space: Optional[str] = None,
    guideline_records: Optional[List[Dict[str, Any]]] = None,
    indexed_sources_out: Optional[List[Dict[str, Any]]] = None,
) -> AsyncIterator[str]:
    effective_guideline_records = list(guideline_records or [])
    if wants_guideline_detail(user_question):
        searched_guidelines = search_guideline_records(question=user_question, k=3)
        if searched_guidelines:
            effective_guideline_records = searched_guidelines
    numbered_sources_block, indexed_sources = build_numbered_sources_block(effective_guideline_records)
    if indexed_sources_out is not None:
        indexed_sources_out.extend(indexed_sources)

    context = _build_knowledge_context(
        user_question=user_question,
        k=k,
        space=space,
        guideline_records=[],
    )
    grounded_context = build_grounded_context_sections(
        measured_room_facts=[],
        backend_semantic_state=None,
        knowledge_cards=context.get("knowledge_cards", []),
        numbered_sources_block=numbered_sources_block,
        allow_general_knowledge=True,
    )
    prompt_template = get_shared_prompt_template(response_directive=CARD_TOOL_RESPONSE_DIRECTIVE)
    messages = prompt_template.format_messages(
        question=user_question,
        context_label="Measured room facts with knowledge grounding",
        context_data=grounded_context,
    )
    prompt_text = _build_prompt_text_from_messages(messages)

    payload = {
        "model": ollama_model(),
        "prompt": prompt_text,
        "stream": True,
        "temperature": ollama_temperature(),
    }

    async with httpx.AsyncClient(timeout=ollama_timeout_seconds()) as client:
        async with client.stream("POST", f"{ollama_base_url()}/api/generate", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                response_text = _coerce_chunk_text(event.get("response"))
                if response_text:
                    yield response_text
