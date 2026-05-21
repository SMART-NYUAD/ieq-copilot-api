"""Shared helpers for HTTP route adapters."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from fastapi.concurrency import run_in_threadpool

try:
    from storage.conversation_store import append_conversation_turn, build_compact_context
    from query_routing.query_orchestrator import execute_query as _default_execute_query
except ImportError:
    from ..storage.conversation_store import append_conversation_turn, build_compact_context
    from ..query_routing.query_orchestrator import execute_query as _default_execute_query


SSE_HEADERS: Dict[str, str] = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def build_query_inputs(
    question: str,
    conversation_id: Optional[str],
) -> Tuple[str, Optional[str], str, bool]:
    normalized_conversation_id, context_block = build_compact_context(conversation_id)
    if context_block:
        return str(question or "").strip(), normalized_conversation_id, context_block, True
    return str(question or "").strip(), normalized_conversation_id, "", False


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


async def execute_non_stream_query(
    *,
    question: str,
    latest_user_question: str,
    conversation_context: str,
    k: int,
    lab_name: Optional[str],
    allow_clarify: bool,
    conversation_id: Optional[str],
    context_applied: bool,
    endpoint_key: str = "query_sync",
    execute_query_fn: Any = None,
) -> Dict[str, Any]:
    fn = execute_query_fn if execute_query_fn is not None else _default_execute_query
    result = await run_in_threadpool(
        fn,
        latest_user_question,
        k,
        lab_name,
        allow_clarify,
        endpoint_key,
        conversation_context,
    )
    turn_index = persist_turn(
        conversation_id=conversation_id,
        question=question,
        answer=str(result.get("answer") or ""),
    )
    metadata = attach_conversation_metadata(
        dict(result.get("metadata") or {}),
        conversation_id=conversation_id,
        conversation_context_applied=context_applied,
        turn_index=turn_index,
    )
    return {"result": result, "turn_index": turn_index, "metadata": metadata}
