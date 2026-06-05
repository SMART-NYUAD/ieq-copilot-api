"""Shared helpers for HTTP route adapters."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from fastapi.concurrency import run_in_threadpool

try:
    from storage.conversation_context import ConversationContext, build_conversation_context
    from storage.conversation_store import append_conversation_turn
    from query_routing.query_orchestrator import execute_query as _default_execute_query
except ImportError:
    from ..storage.conversation_context import ConversationContext, build_conversation_context
    from ..storage.conversation_store import append_conversation_turn
    from ..query_routing.query_orchestrator import execute_query as _default_execute_query


SSE_HEADERS: Dict[str, str] = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def build_query_context(
    question: str,
    lab_name: Optional[str],
    conversation_id: Optional[str],
) -> ConversationContext:
    """Build the canonical ConversationContext for one HTTP turn."""
    return build_conversation_context(
        question=question,
        lab_name=lab_name,
        conversation_id=conversation_id,
    )


def persist_turn(conversation_id: Optional[str], question: str, answer: str) -> Optional[int]:
    if not conversation_id:
        return None
    return append_conversation_turn(
        conversation_id=conversation_id,
        user_message=question,
        assistant_message=answer,
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


def route_plan_metadata(route_plan: Any, **kwargs) -> Dict[str, Any]:
    if route_plan is None:
        return {}
    meta: Dict[str, Any] = {}
    for attr in ("intent", "confidence", "lab_name", "second_lab_name", "metrics", "time_phrase", "model", "fallback_used"):
        val = getattr(route_plan, attr, None)
        if val is not None:
            meta[attr] = val.value if hasattr(val, "value") else val
    return meta


def _collect_token_text(chunk: str, in_think: bool) -> tuple[str, bool]:
    """Parse one SSE chunk, return (text_to_keep, new_in_think_state)."""
    if not chunk.startswith("data:"):
        return "", in_think
    try:
        payload = json.loads(chunk[5:].strip())
    except (json.JSONDecodeError, AttributeError):
        return "", in_think
    if payload.get("event") != "token":
        return "", in_think
    text = payload.get("text", "")
    if text == "<think>":
        return "", True
    if text == "</think>":
        return "", False
    if in_think:
        return "", in_think
    return text, in_think


async def execute_non_stream_query(
    *,
    ctx: ConversationContext,
    k: int,
    allow_clarify: bool,
    endpoint_key: str = "query_sync",
    execute_query_fn: Any = None,
) -> Dict[str, Any]:
    fn = execute_query_fn if execute_query_fn is not None else _default_execute_query
    result = await run_in_threadpool(fn, ctx, k, allow_clarify, endpoint_key)
    turn_index = persist_turn(
        conversation_id=ctx.conversation_id,
        question=ctx.original_question,
        answer=str(result.get("answer") or ""),
    )
    metadata = attach_conversation_metadata(
        dict(result.get("metadata") or {}),
        conversation_id=ctx.conversation_id,
        conversation_context_applied=bool(ctx.raw_block),
        turn_index=turn_index,
    )
    return {"result": result, "turn_index": turn_index, "metadata": metadata}
