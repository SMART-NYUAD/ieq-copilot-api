"""Shared query runtime helpers used by HTTP adapters."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from fastapi.concurrency import run_in_threadpool

try:
    from http_routes.route_helpers import (
        attach_conversation_metadata,
        build_query_inputs,
        persist_turn,
    )
    from query_routing.query_orchestrator import execute_query
except ImportError:
    from .route_helpers import (
        attach_conversation_metadata,
        build_query_inputs,
        persist_turn,
    )
    from ..query_routing.query_orchestrator import execute_query


def normalize_lab_name(lab_name: Optional[str]) -> Optional[str]:
    return (lab_name or "").strip() or None


def normalize_k(k: Optional[int], default: int = 5) -> int:
    return int(k or default)


def normalize_allow_clarify(flag: Optional[bool]) -> bool:
    return bool(flag if flag is not None else True)


def build_query_context(
    question: str,
    conversation_id: Optional[str],
) -> Tuple[str, Optional[str], str, bool]:
    return build_query_inputs(question=question, conversation_id=conversation_id)


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
    execute_query_fn: Any = execute_query,
) -> Dict[str, Any]:
    result = await run_in_threadpool(
        execute_query_fn,
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
