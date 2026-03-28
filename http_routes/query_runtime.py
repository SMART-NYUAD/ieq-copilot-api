"""Shared query runtime helpers used by HTTP adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from fastapi.concurrency import run_in_threadpool

try:
    from executors.db_query_executor import prepare_db_query
    from executors.env_query_langchain import get_knowledge_context_stats
    from http_routes.route_helpers import (
        attach_conversation_metadata,
        build_effective_question,
        persist_turn,
    )
    from query_routing.query_orchestrator import (
        execute_query,
        get_route_plan,
        query_scope_class,
        resolve_execution_intent,
        should_clarify,
        should_use_knowledge_executor,
    )
except ImportError:
    from ..executors.db_query_executor import prepare_db_query
    from ..executors.env_query_langchain import get_knowledge_context_stats
    from .route_helpers import (
        attach_conversation_metadata,
        build_effective_question,
        persist_turn,
    )
    from ..query_routing.query_orchestrator import (
        execute_query,
        get_route_plan,
        query_scope_class,
        resolve_execution_intent,
        should_clarify,
        should_use_knowledge_executor,
    )


@dataclass(frozen=True)
class StreamRouteRuntime:
    route_plan: Any
    decision: Any
    scope_class: str
    should_clarify_response: bool
    use_knowledge_executor: bool
    execution_intent: Any


def normalize_lab_name(lab_name: Optional[str]) -> Optional[str]:
    return (lab_name or "").strip() or None


def normalize_k(k: Optional[int], default: int = 5) -> int:
    return int(k or default)


def normalize_allow_clarify(flag: Optional[bool]) -> bool:
    return bool(flag if flag is not None else True)


def build_query_context(question: str, conversation_id: Optional[str]) -> Tuple[str, Optional[str], bool]:
    return build_effective_question(question=question, conversation_id=conversation_id)


async def execute_non_stream_query(
    *,
    question: str,
    effective_question: str,
    k: int,
    lab_name: Optional[str],
    allow_clarify: bool,
    conversation_id: Optional[str],
    context_applied: bool,
    execute_query_fn: Any = execute_query,
) -> Dict[str, Any]:
    result = await run_in_threadpool(
        execute_query_fn,
        effective_question,
        k,
        lab_name,
        allow_clarify,
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


async def resolve_stream_runtime(
    *,
    effective_question: str,
    lab_name: Optional[str],
    allow_clarify: bool,
    get_route_plan_fn: Any = get_route_plan,
    query_scope_class_fn: Any = query_scope_class,
    should_clarify_fn: Any = should_clarify,
    should_use_knowledge_executor_fn: Any = should_use_knowledge_executor,
    resolve_execution_intent_fn: Any = resolve_execution_intent,
) -> StreamRouteRuntime:
    route_plan = await run_in_threadpool(get_route_plan_fn, effective_question, lab_name)
    decision = route_plan.decision
    scope_class = query_scope_class_fn(route_plan)
    should_clarify_response = should_clarify_fn(route_plan=route_plan, allow_clarify=allow_clarify)
    if scope_class == "non_domain":
        # Non-domain questions should hit scope guardrail directly.
        should_clarify_response = False
    use_knowledge_executor = should_use_knowledge_executor_fn(route_plan)
    if scope_class == "non_domain" or should_clarify_response:
        use_knowledge_executor = False
    execution_intent = resolve_execution_intent_fn(decision.intent)
    return StreamRouteRuntime(
        route_plan=route_plan,
        decision=decision,
        scope_class=scope_class,
        should_clarify_response=should_clarify_response,
        use_knowledge_executor=use_knowledge_executor,
        execution_intent=execution_intent,
    )


async def fetch_knowledge_stats(
    effective_question: str,
    k: int,
    lab_name: Optional[str],
    stats_fn: Any = get_knowledge_context_stats,
) -> Dict[str, Any]:
    return await run_in_threadpool(
        stats_fn,
        effective_question,
        max(1, min(k, 8)),
        lab_name,
    )


async def fetch_db_context(
    *,
    effective_question: str,
    execution_intent: Any,
    lab_name: Optional[str],
    planner_parameters: Optional[Dict[str, Any]],
    prepare_db_query_fn: Any = prepare_db_query,
) -> Dict[str, Any]:
    return await run_in_threadpool(
        prepare_db_query_fn,
        effective_question,
        execution_intent,
        lab_name,
        planner_parameters,
    )
