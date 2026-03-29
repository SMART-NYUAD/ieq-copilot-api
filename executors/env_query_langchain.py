"""
LangChain-based query module for knowledge-card grounded QA.

This module provides semantic search and Q&A capabilities over static
knowledge cards stored in PostgreSQL with pgvector embeddings.
"""

import json
from datetime import datetime, timedelta, timezone
import os
from threading import Lock
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

try:
    from storage.embeddings import embed_texts
    from storage.postgres_client import get_cursor
    from storage.sql_queries import ENV_KNOWLEDGE_QUERY_SEMANTIC_SQL
except ImportError:
    from ..storage.embeddings import embed_texts
    from ..storage.postgres_client import get_cursor
    from ..storage.sql_queries import ENV_KNOWLEDGE_QUERY_SEMANTIC_SQL
try:
    from prompting.shared_prompts import (
        build_grounded_context_sections,
        get_shared_prompt_template,
    )
except ImportError:
    from ..prompting.shared_prompts import (
        build_grounded_context_sections,
        get_shared_prompt_template,
    )

try:
    from http_schemas import validate_tool_evidence
except ImportError:
    from ..http_schemas import validate_tool_evidence
try:
    from query_routing.router_signals import extract_query_signals
except ImportError:
    from ..query_routing.router_signals import extract_query_signals

CARD_TOOL_RESPONSE_DIRECTIVE = """
You are answering from card-based retrieval context.
- First, answer the exact user question in 1-2 lines.
- For air-quality assessment/summary queries, include:
  1) overall status in plain language,
  2) metric interpretation with citations when metrics are present,
  3) confidence qualifier when key metrics are missing,
  4) 2-4 practical recommendations.
- If the question is specifically about risks, lead with the main risk level and top risk drivers (or explicitly state that no major risk is evident from the data) before any broader summary.
- Recommendations must be grounded in provided context only.
- Keep wording practical (what occupants should do next), not just descriptive.
""".strip()
GENERAL_CHAT_RESPONSE_DIRECTIVE = """
You are a helpful conversational assistant in an IEQ application.
- For social or general questions, respond naturally and directly.
- Do not invent measured IEQ values, trends, or recommendations unless explicitly asked for measured analysis.
- Keep responses concise and human.
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


# --------------------------------------------------------------------
# LLM CLIENT INITIALIZATION
# --------------------------------------------------------------------

def get_llm_client():
    """Initialize and return ChatOllama client."""
    # Use direct local Ollama by default; override with OLLAMA_BASE_URL when needed.
    import os
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    model = os.getenv("OLLAMA_MODEL", "qwen3:30b-a3b-instruct-2507-q4_K_M")
    
    return ChatOllama(
        base_url=base_url,
        model=model,
        temperature=0.1,
    )


def _is_explanation_query(question: str) -> bool:
    q = (question or "").lower()
    hints = (
        "what is",
        "what does",
        "what do you mean",
        "define",
        "definition",
        "mean by",
    )
    return any(hint in q for hint in hints)


def _is_guardrail_query(question: str) -> bool:
    q = (question or "").lower()
    hints = (
        "health risk",
        "safe",
        "dangerous",
        "medical",
        "diagnosis",
        "harmful",
        "unhealthy",
    )
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
            # Increase probes for small/medium collections to avoid ANN false negatives.
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


def _build_knowledge_context_uncached(user_question: str, k: int = 5, space: Optional[str] = None) -> Dict[str, Any]:
    knowledge_cards = search_knowledge_cards(user_question, k=max(3, min(5, k)))
    grounded_context = build_card_grounded_context(
        [], knowledge_cards, allow_general_knowledge=True
    )
    return {
        "knowledge_cards": knowledge_cards,
        "grounded_context": grounded_context,
    }


def _build_knowledge_context(user_question: str, k: int = 5, space: Optional[str] = None) -> Dict[str, Any]:
    key = _knowledge_context_cache_key(user_question=user_question, k=k, space=space)
    ttl_seconds = _knowledge_context_cache_ttl_seconds()
    if ttl_seconds <= 0:
        return _build_knowledge_context_uncached(user_question=user_question, k=k, space=space)

    now = time.monotonic()
    with _KNOWLEDGE_CONTEXT_CACHE_LOCK:
        cached = _KNOWLEDGE_CONTEXT_CACHE.get(key)
        if cached and cached[0] > now:
            return cached[1]

    context = _build_knowledge_context_uncached(user_question=user_question, k=k, space=space)
    expires_at = now + ttl_seconds
    with _KNOWLEDGE_CONTEXT_CACHE_LOCK:
        _KNOWLEDGE_CONTEXT_CACHE[key] = (expires_at, context)
        _prune_knowledge_context_cache(now=now)
    return context


def get_knowledge_context_stats(user_question: str, k: int = 5, space: Optional[str] = None) -> Dict[str, Any]:
    if _is_non_domain_question(user_question):
        return {"cards_retrieved": 0, "knowledge_cards_retrieved": 0}
    context = _build_knowledge_context(user_question=user_question, k=k, space=space)
    knowledge_cards = context.get("knowledge_cards") or []
    return {
        "cards_retrieved": int(len(knowledge_cards)),
        "knowledge_cards_retrieved": int(len(knowledge_cards)),
    }


def answer_env_question_with_metadata(
    user_question: str, k: int = 5, space: Optional[str] = None
) -> Dict[str, Any]:
    if _is_non_domain_question(user_question):
        answer = get_general_chat_chain().invoke({"question": user_question})
        evidence = validate_tool_evidence(
            {
                "evidence_kind": "knowledge_qa",
                "intent": "definition_explanation",
                "strategy": "direct",
                "metric_aliases": [],
                "resolved_scope": space,
                "resolved_time_window": None,
                "provenance_sources": [],
                "confidence_notes": ["general_non_domain_response"],
                "recommendation_allowed": False,
            }
        )
        return {
            "answer": answer,
            "cards_retrieved": 0,
            "knowledge_cards_retrieved": 0,
            "evidence": evidence,
        }

    context = _build_knowledge_context(user_question=user_question, k=k, space=space)
    qa_chain = get_qa_chain()
    answer = qa_chain.invoke(
        {
            "question": user_question,
            "context_label": "Measured room facts with knowledge grounding",
            "context_data": context["grounded_context"],
        }
    )
    knowledge_cards = context.get("knowledge_cards") or []
    evidence_sources = []
    for card in knowledge_cards:
        evidence_sources.append(
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
        )
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
        "answer": answer,
        "cards_retrieved": int(len(knowledge_cards)),
        "knowledge_cards_retrieved": int(len(knowledge_cards)),
        "evidence": evidence,
    }


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
        allow_general_knowledge=allow_general_knowledge,
    )


# --------------------------------------------------------------------
# QA CHAIN
# --------------------------------------------------------------------

def get_qa_prompt() -> ChatPromptTemplate:
    """Build and return the QA prompt template."""
    return get_shared_prompt_template(response_directive=CARD_TOOL_RESPONSE_DIRECTIVE)


def get_qa_chain():
    """Build and return the QA chain."""
    prompt = get_qa_prompt()
    return prompt | get_llm_client() | StrOutputParser()


def get_general_chat_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", GENERAL_CHAT_RESPONSE_DIRECTIVE),
            ("user", "{question}"),
        ]
    )


def get_general_chat_chain():
    return get_general_chat_prompt() | get_llm_client() | StrOutputParser()


def _is_non_domain_question(user_question: str) -> bool:
    try:
        signals = extract_query_signals(user_question)
    except Exception:
        return False
    return str(signals.get("query_scope_class") or "").strip().lower() == "non_domain"


def _coerce_chunk_text(value) -> str:
    """Convert chunk payloads to plain text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        # LangChain content blocks can be lists of dicts/strings.
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


def _extract_thinking_and_response(chunk) -> Tuple[str, str]:
    """
    Extract thinking and response text from a streamed LLM chunk.

    Ollama may stream reasoning separately in a `thinking` field while normal
    answer tokens arrive in `content`/`response`.
    """
    thinking_text = ""
    response_text = ""

    if isinstance(chunk, str):
        return "", chunk

    # Most LangChain message chunks expose generated text in `content`.
    response_text += _coerce_chunk_text(getattr(chunk, "content", None))

    # Additional kwargs frequently carry provider-specific fields like thinking.
    additional_kwargs = getattr(chunk, "additional_kwargs", None)
    if isinstance(additional_kwargs, dict):
        thinking_text += _coerce_chunk_text(additional_kwargs.get("thinking"))

    # Some adapters expose raw provider payload in response metadata.
    response_metadata = getattr(chunk, "response_metadata", None)
    if isinstance(response_metadata, dict):
        thinking_text += _coerce_chunk_text(response_metadata.get("thinking"))
        message_obj = response_metadata.get("message")
        if isinstance(message_obj, dict):
            thinking_text += _coerce_chunk_text(message_obj.get("thinking"))
            if not response_text:
                response_text += _coerce_chunk_text(message_obj.get("content"))

    # Handle plain dict chunks for compatibility.
    if isinstance(chunk, dict):
        thinking_text += _coerce_chunk_text(chunk.get("thinking"))
        response_text += _coerce_chunk_text(chunk.get("response"))
        response_text += _coerce_chunk_text(chunk.get("content"))
        message_obj = chunk.get("message")
        if isinstance(message_obj, dict):
            thinking_text += _coerce_chunk_text(message_obj.get("thinking"))
            response_text += _coerce_chunk_text(message_obj.get("content"))

    return thinking_text, response_text


# --------------------------------------------------------------------
# TOP-LEVEL QUERY FUNCTION
# --------------------------------------------------------------------

def answer_env_question(user_question: str, k: int = 5, space: Optional[str] = None) -> str:
    """
    Answer a user question about indoor air quality using knowledge cards.
    
    Args:
        user_question: The user's question
        k: Number of cards to retrieve for context
        space: Optional space/lab name filter
    
    Returns:
        Answer string from the LLM
    """
    result = answer_env_question_with_metadata(user_question=user_question, k=k, space=space)
    return str(result.get("answer") or "")


async def stream_answer_env_question(
    user_question: str,
    k: int = 5,
    space: Optional[str] = None,
    think: Optional[bool] = None,
) -> AsyncIterator[str]:
    """
    Stream answer tokens for a user question.

    Args:
        user_question: The user's question
        k: Number of cards to retrieve for context
        space: Optional space/lab name filter

    Yields:
        Token/text chunks from the LLM as they are generated
    """
    if _is_non_domain_question(user_question):
        messages = get_general_chat_prompt().format_messages(question=user_question)
    else:
        context = _build_knowledge_context(
            user_question=user_question,
            k=k,
            space=space,
        )
        grounded_context = str(context.get("grounded_context") or "")
        qa_prompt = get_qa_prompt()
        messages = qa_prompt.format_messages(
            question=user_question,
            context_label="Measured room facts with knowledge grounding",
            context_data=grounded_context,
        )

    # Build a plain prompt for Ollama /api/generate streaming.
    # This lets us access the new `thinking` field directly and merge it
    # back into response text as <think>...</think> for frontend compatibility.
    prompt_parts = []
    for m in messages:
        role = getattr(m, "type", "user").upper()
        content = _coerce_chunk_text(getattr(m, "content", ""))
        prompt_parts.append(f"{role}:\n{content}")
    prompt_text = "\n\n".join(prompt_parts)

    import os
    import httpx

    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    model = os.getenv("OLLAMA_MODEL", "qwen3:30b-a3b-instruct-2507-q4_K_M")
    api_url = f"{base_url}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt_text,
        "stream": True,
        "temperature": 0.1,
    }
    # Some model/runtime combos emit more raw reasoning when think=False is passed
    # explicitly. We only pass think=True and use server-side filtering for think=False.
    if think is True:
        payload["think"] = True

    in_thinking_block = False
    include_thinking = think is not False
    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream("POST", api_url, json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    # Skip malformed chunks instead of breaking the stream.
                    continue

                thinking_text = _coerce_chunk_text(event.get("thinking"))
                response_text = _coerce_chunk_text(event.get("response"))

                if include_thinking and thinking_text:
                    if not in_thinking_block:
                        in_thinking_block = True
                        yield "<think>"
                    yield thinking_text

                if response_text:
                    if in_thinking_block:
                        in_thinking_block = False
                        yield "</think>"
                    yield response_text

    if include_thinking and in_thinking_block:
        yield "</think>"


# --------------------------------------------------------------------
# EXAMPLE USAGE
# --------------------------------------------------------------------

if __name__ == "__main__":
    # Example query
    question = "What was the air quality like in smart_lab yesterday afternoon?"
    answer = answer_env_question(question, k=5)
    print("\n" + "="*80)
    print("QUESTION:", question)
    print("="*80)
    print("\nANSWER:")
    print(answer)
    print("="*80)
