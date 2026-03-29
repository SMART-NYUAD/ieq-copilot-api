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
        build_query_inputs,
        persist_turn,
    )
    from query_routing.query_orchestrator import (
        execute_query,
        get_route_decision_contract,
    )
    from query_routing.router_types import RouteDecisionContract, RouteExecutor
    from query_routing.observability import record_endpoint_executor
except ImportError:
    from ..executors.db_query_executor import prepare_db_query
    from ..executors.env_query_langchain import get_knowledge_context_stats
    from .route_helpers import (
        attach_conversation_metadata,
        build_query_inputs,
        persist_turn,
    )
    from ..query_routing.query_orchestrator import (
        execute_query,
        get_route_decision_contract,
    )
    from ..query_routing.router_types import RouteDecisionContract, RouteExecutor
    from ..query_routing.observability import record_endpoint_executor


@dataclass(frozen=True)
class StreamRouteRuntime:
    route_contract: RouteDecisionContract
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
    _ = conversation_context  # Reserved for future context-aware generation hooks.
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


async def resolve_stream_runtime(
    *,
    latest_user_question: str,
    lab_name: Optional[str],
    allow_clarify: bool,
    endpoint_key: str = "query_stream",
    get_route_decision_contract_fn: Any = get_route_decision_contract,
    get_route_plan_fn: Optional[Any] = None,
    query_scope_class_fn: Optional[Any] = None,
    should_clarify_fn: Optional[Any] = None,
    should_use_knowledge_executor_fn: Optional[Any] = None,
    resolve_execution_intent_fn: Optional[Any] = None,
) -> StreamRouteRuntime:
    route_plan_compat = get_route_plan_fn
    if route_plan_compat is not None:
        route_plan = await run_in_threadpool(route_plan_compat, latest_user_question, lab_name)
        # Compatibility mode accepts an externally-supplied planner while still
        # routing through the orchestrator contract selector (rollout/shadow).
        route_contract = await run_in_threadpool(
            get_route_decision_contract_fn,
            latest_user_question,
            lab_name,
            allow_clarify,
            route_plan,
        )
    else:
        route_contract = await run_in_threadpool(
            get_route_decision_contract_fn,
            latest_user_question,
            lab_name,
            allow_clarify,
        )
    route_plan = route_contract.route_plan
    decision = route_plan.decision
    scope_class = route_contract.query_scope_class
    should_clarify_response = route_contract.executor == RouteExecutor.CLARIFY_GATE
    use_knowledge_executor = route_contract.executor == RouteExecutor.KNOWLEDGE_QA
    execution_intent = route_contract.execution_intent
    record_endpoint_executor(
        latest_question_hash=route_contract.latest_question_hash,
        endpoint_key=endpoint_key,
        executor=route_contract.executor.value,
    )
    return StreamRouteRuntime(
        route_contract=route_contract,
        route_plan=route_plan,
        decision=decision,
        scope_class=scope_class,
        should_clarify_response=should_clarify_response,
        use_knowledge_executor=use_knowledge_executor,
        execution_intent=execution_intent,
    )


async def fetch_knowledge_stats(
    latest_user_question: str,
    k: int,
    lab_name: Optional[str],
    stats_fn: Any = get_knowledge_context_stats,
) -> Dict[str, Any]:
    return await run_in_threadpool(
        stats_fn,
        latest_user_question,
        max(1, min(k, 8)),
        lab_name,
    )


async def fetch_db_context(
    *,
    latest_user_question: str,
    execution_intent: Any,
    lab_name: Optional[str],
    planner_parameters: Optional[Dict[str, Any]],
    prepare_db_query_fn: Any = prepare_db_query,
) -> Dict[str, Any]:
    return await run_in_threadpool(
        prepare_db_query_fn,
        latest_user_question,
        execution_intent,
        lab_name,
        planner_parameters,
    )
