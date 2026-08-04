"""Shared helpers for HTTP route adapters."""

from __future__ import annotations

from typing import Any, Dict, Optional

from storage.conversation_context import ConversationContext, build_conversation_context
from storage.conversation_store import ANONYMOUS_OWNER, append_conversation_turn


SSE_HEADERS: Dict[str, str] = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def build_query_context(
    question: str,
    lab_name: Optional[str],
    conversation_id: Optional[str],
    owner: str = ANONYMOUS_OWNER,
    role: Optional[str] = None,
) -> ConversationContext:
    """Build the canonical ConversationContext for one HTTP turn."""
    return build_conversation_context(
        question=question,
        lab_name=lab_name,
        conversation_id=conversation_id,
        owner=owner,
        role=role,
    )


def persist_turn(
    conversation_id: Optional[str],
    question: str,
    answer: str,
    owner: str = ANONYMOUS_OWNER,
) -> Optional[int]:
    if not conversation_id:
        return None
    return append_conversation_turn(
        conversation_id=conversation_id,
        user_message=question,
        assistant_message=answer,
        owner=owner,
    )


def attach_conversation_metadata(
    metadata: Dict[str, Any],
    *,
    conversation_id: Optional[str],
    conversation_context_applied: bool,
    turn_index: Optional[int],
) -> Dict[str, Any]:
    meta = dict(metadata)
    meta["conversation_id"] = conversation_id
    meta["conversation_context_applied"] = conversation_context_applied
    meta["turn_index"] = turn_index
    return meta
