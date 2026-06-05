"""Canonical per-turn conversation context built once at the HTTP boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:
    from storage.conversation_store import build_compact_context
    from storage.conversation_memory import apply_routing_memory, compute_question_signals, extract_routing_memory
except ImportError:
    from .conversation_store import build_compact_context
    from .conversation_memory import apply_routing_memory, compute_question_signals, extract_routing_memory

_ROUTING_SNIPPET_LINES = 4   # lines fed to the router LLM
_LLM_HISTORY_MAX_CHARS = 800  # chars fed to the answer LLM


def _extract_content_lines(raw_block: str) -> list[str]:
    return [
        line for line in raw_block.strip().splitlines()
        if line.strip() and not line.startswith("Previous conversation context")
    ]


@dataclass(frozen=True)
class ConversationContext:
    """Single canonical conversation context created once per HTTP turn.

    All downstream components (router, DB executor, answer LLM) receive the
    same object and read the view they need — nothing reconstructs context
    from the raw string independently.
    """

    conversation_id: str
    original_question: str   # As typed by the user (for turn persistence)
    raw_block: str           # Full "Previous conversation context…" string
    effective_question: str  # After memory carry-over (metric/lab appended)
    effective_lab: Optional[str]  # Lab resolved from memory or request
    routing_snippet: str     # Compact last-N-lines for the router LLM
    llm_history: str         # Compact block injected into answer LLM context


def build_conversation_context(
    question: str,
    lab_name: Optional[str],
    conversation_id: Optional[str],
) -> ConversationContext:
    """Build the canonical context for one turn.

    Loads prior turns from the store, applies memory carry-over, and
    pre-computes every view needed downstream so no layer does its own
    extraction.
    """
    original_question = str(question or "").strip()
    cid, raw_block = build_compact_context(conversation_id)

    content_lines = _extract_content_lines(raw_block)
    routing_snippet = "\n".join(content_lines[-_ROUTING_SNIPPET_LINES:])
    llm_history = "\n".join(content_lines[-6:])[:_LLM_HISTORY_MAX_CHARS]

    if raw_block:
        signals = compute_question_signals(original_question)
        memory = extract_routing_memory(conversation_context=raw_block, current_signals=signals)
        effective_question, effective_lab, _ = apply_routing_memory(
            question=original_question,
            lab_name=lab_name,
            memory=memory,
            current_signals=signals,
        )
    else:
        effective_question = original_question
        effective_lab = (lab_name or "").strip() or None

    return ConversationContext(
        conversation_id=cid,
        original_question=original_question,
        raw_block=raw_block,
        effective_question=effective_question,
        effective_lab=effective_lab,
        routing_snippet=routing_snippet,
        llm_history=llm_history,
    )
