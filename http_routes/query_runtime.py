"""Shared query runtime helpers used by HTTP adapters."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Dict, List, Optional, Tuple

from fastapi.concurrency import run_in_threadpool

try:
    from core_settings import load_settings
    from query_routing.agent_tools import execute_agent_tool_call, summarize_tool_observation
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
    from query_routing.router_signals import extract_query_signals
    from query_routing.router_types import RouteDecisionContract, RouteExecutor
    from query_routing.observability import record_endpoint_executor
    from storage.conversation_memory import RoutingMemory, apply_routing_memory, extract_routing_memory
except ImportError:
    from ..core_settings import load_settings
    from ..query_routing.agent_tools import execute_agent_tool_call, summarize_tool_observation
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
    from ..query_routing.router_signals import extract_query_signals
    from ..query_routing.router_types import RouteDecisionContract, RouteExecutor
    from ..query_routing.observability import record_endpoint_executor
    from ..storage.conversation_memory import RoutingMemory, apply_routing_memory, extract_routing_memory


@dataclass(frozen=True)
class StreamRouteRuntime:
    route_contract: RouteDecisionContract
    route_plan: Any
    decision: Any
    scope_class: str
    should_clarify_response: bool
    use_knowledge_executor: bool
    execution_intent: Any


@dataclass(frozen=True)
class AgentStreamResolution:
    runtime: StreamRouteRuntime
    agent_step_trace: List[Dict[str, Any]]
    tools_called: List[str]
    finish_reason: str
    state: "AgentState"


@dataclass(frozen=True)
class AgentState:
    original_question: str
    conversation_context: str
    routing_memory: RoutingMemory
    effective_question: str
    effective_lab: Optional[str]


_VOLATILE_TOOL_ARGUMENT_KEYS = {
    "question",
    "user_question",
    "planning_question",
    "context",
    "conversation_context",
    "observation",
    "observations",
    "agent_observations",
    "agent_tool_observations",
    "previous_observations",
}


def _normalize_tool_arguments_for_signature(arguments: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Drop volatile narrative fields so dedupe keys reflect semantic tool calls."""

    def _normalize(value: Any) -> Any:
        if isinstance(value, dict):
            out: Dict[str, Any] = {}
            for k, v in value.items():
                key = str(k or "").strip().lower()
                if key in _VOLATILE_TOOL_ARGUMENT_KEYS:
                    continue
                out[str(k)] = _normalize(v)
            return out
        if isinstance(value, list):
            return [_normalize(item) for item in value]
        return value

    if not isinstance(arguments, dict):
        return {}
    normalized = _normalize(arguments)
    return normalized if isinstance(normalized, dict) else {}


def _tool_call_signature(
    *,
    tool_name: str,
    arguments: Optional[Dict[str, Any]],
    route_type: str,
    lab_name: Optional[str],
) -> str:
    normalized_arguments = _normalize_tool_arguments_for_signature(arguments)
    return json.dumps(
        {
            "tool_name": str(tool_name or "").strip().lower(),
            "arguments": normalized_arguments,
            "route_type": str(route_type or "").strip().lower(),
            "lab_name": str(lab_name or "").strip().lower() or None,
        },
        ensure_ascii=True,
        sort_keys=True,
    )


_TOOL_BUDGET_BY_INTENT: Dict[str, int] = {
    "comparison_db": 1,
    "aggregation_db": 1,
    "current_status_db": 1,
    "point_lookup_db": 1,
    "forecast_db": 1,
    "anomaly_analysis_db": 2,
}
_ALLOWED_AGENT_GOALS = {"compare", "explain", "recommend"}
_COMPARE_HINTS = ("compare", "comparison", "vs", "versus", "difference")
_EXPLAIN_HINTS = ("why", "explain", "reason", "because", "difference happened")
_RECOMMEND_HINTS = ("recommend", "action", "what to take", "what should", "next step", "improve")
_EXPLAIN_COVERAGE_MARKERS = ("because", "due to", "likely", "reason", "explains")
_RECOMMEND_COVERAGE_MARKERS = ("recommend", "should", "action", "improve", "consider")


def _tool_budget_for_intent(intent_value: str) -> int:
    return max(1, int(_TOOL_BUDGET_BY_INTENT.get(str(intent_value or "").strip().lower(), 1)))


def _derive_required_goals(question: str) -> set[str]:
    q = str(question or "").strip().lower()
    goals: set[str] = set()
    if any(token in q for token in _COMPARE_HINTS):
        goals.add("compare")
    if any(token in q for token in _EXPLAIN_HINTS):
        goals.add("explain")
    if any(token in q for token in _RECOMMEND_HINTS):
        goals.add("recommend")
    return goals


def _normalize_declared_goal_coverage(route_plan: Any) -> set[str]:
    raw = getattr(route_plan, "goal_coverage", ())
    if not isinstance(raw, (list, tuple)):
        return set()
    out: set[str] = set()
    for item in raw:
        goal = str(item or "").strip().lower()
        if goal in _ALLOWED_AGENT_GOALS:
            out.add(goal)
    return out


def _derive_goals_from_tool_result(*, route_type: str, tool_name: Optional[str], answer_text: str) -> set[str]:
    goals: set[str] = set()
    route = str(route_type or "").strip().lower()
    tool = str(tool_name or "").strip().lower()
    answer = str(answer_text or "").strip().lower()
    if route == "comparison_db" or tool == "compare_spaces":
        goals.add("compare")
    if any(marker in answer for marker in _EXPLAIN_COVERAGE_MARKERS):
        goals.add("explain")
    if any(marker in answer for marker in _RECOMMEND_COVERAGE_MARKERS):
        goals.add("recommend")
    return goals


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


async def resolve_agent_stream_runtime(
    *,
    latest_user_question: str,
    k: int,
    lab_name: Optional[str],
    allow_clarify: bool,
    conversation_context: str = "",
    endpoint_key: str,
    get_route_decision_contract_fn: Any = get_route_decision_contract,
    get_route_plan_fn: Optional[Any] = None,
    query_scope_class_fn: Optional[Any] = None,
    should_clarify_fn: Optional[Any] = None,
    should_use_knowledge_executor_fn: Optional[Any] = None,
    resolve_execution_intent_fn: Optional[Any] = None,
) -> AgentStreamResolution:
    original_question = str(latest_user_question or "").strip()
    query_signals = extract_query_signals(question=original_question, lab_name=lab_name)
    routing_memory = extract_routing_memory(
        conversation_context=conversation_context,
        current_signals=query_signals,
    )
    effective_question, effective_lab_name, _ = apply_routing_memory(
        question=original_question,
        lab_name=lab_name,
        memory=routing_memory,
        current_signals=query_signals,
    )
    state = AgentState(
        original_question=original_question,
        conversation_context=str(conversation_context or ""),
        routing_memory=routing_memory,
        effective_question=effective_question,
        effective_lab=effective_lab_name,
    )
    settings = load_settings()
    if not bool(getattr(settings, "agentic_mode", False)):
        runtime = await resolve_stream_runtime(
            latest_user_question=effective_question,
            lab_name=effective_lab_name,
            allow_clarify=allow_clarify,
            endpoint_key=endpoint_key,
            get_route_decision_contract_fn=get_route_decision_contract_fn,
            get_route_plan_fn=get_route_plan_fn,
            query_scope_class_fn=query_scope_class_fn,
            should_clarify_fn=should_clarify_fn,
            should_use_knowledge_executor_fn=should_use_knowledge_executor_fn,
            resolve_execution_intent_fn=resolve_execution_intent_fn,
        )
        return AgentStreamResolution(
            runtime=runtime,
            agent_step_trace=[],
            tools_called=[],
            finish_reason="disabled",
            state=state,
        )

    planning_question = effective_question
    planning_lab_name = effective_lab_name
    observations: List[str] = []
    executed_tool_signatures: set[str] = set()
    agent_step_trace: List[Dict[str, Any]] = []
    tools_called: List[str] = []
    finish_reason = "max_steps"
    final_runtime: Optional[StreamRouteRuntime] = None

    max_steps = max(1, int(getattr(settings, "agent_max_steps", 4)))
    max_failures = max(1, int(getattr(settings, "agent_max_consecutive_failures", 2)))
    stall_threshold = max(1, int(getattr(settings, "agent_stall_threshold", 2)))
    consecutive_failures = 0
    repeated_block_count = 0
    required_goals = _derive_required_goals(latest_user_question)
    covered_goals: set[str] = set()
    tool_calls_by_intent: Dict[str, int] = {}

    for step_index in range(1, max_steps + 1):
        runtime = await resolve_stream_runtime(
            latest_user_question=planning_question,
            lab_name=planning_lab_name,
            allow_clarify=allow_clarify,
            endpoint_key=endpoint_key,
            get_route_decision_contract_fn=get_route_decision_contract_fn,
            get_route_plan_fn=get_route_plan_fn,
            query_scope_class_fn=query_scope_class_fn,
            should_clarify_fn=should_clarify_fn,
            should_use_knowledge_executor_fn=should_use_knowledge_executor_fn,
            resolve_execution_intent_fn=resolve_execution_intent_fn,
        )
        final_runtime = runtime
        route_plan = runtime.route_plan
        action = str(getattr(getattr(route_plan, "agent_action", None), "value", "finalize"))
        tool_name = str(getattr(route_plan, "tool_name", "") or "").strip().lower() or None

        step_entry: Dict[str, Any] = {
            "step": step_index,
            "action": action,
            "tool_name": tool_name,
            "route_type": route_plan.decision.intent.value,
            "executor": runtime.route_contract.executor.value,
            "confidence": float(route_plan.decision.confidence),
            "required_goals": sorted(list(required_goals)),
            "covered_goals": sorted(list(covered_goals)),
        }
        agent_step_trace.append(step_entry)

        if runtime.should_clarify_response:
            finish_reason = "clarify"
            break

        if action == "tool_call" and tool_name:
            route_type = str(route_plan.decision.intent.value or "").strip().lower()
            intent_budget = _tool_budget_for_intent(route_type)
            calls_for_intent = int(tool_calls_by_intent.get(route_type, 0))
            if calls_for_intent >= intent_budget:
                step_entry["blocked_repeat"] = True
                step_entry["blocked_reason"] = "tool_budget_exhausted"
                step_entry["intent_tool_budget"] = intent_budget
                finish_reason = "tool_budget_exhausted"
                break
            signature = _tool_call_signature(
                tool_name=tool_name,
                arguments=getattr(route_plan, "tool_arguments", {}),
                route_type=route_type,
                lab_name=planning_lab_name,
            )
            if signature in executed_tool_signatures:
                repeated_block_count += 1
                step_entry["blocked_repeat"] = True
                step_entry["blocked_reason"] = "same_tool_signature_repeated"
                if repeated_block_count >= stall_threshold:
                    finish_reason = "tool_repeat_blocked"
                    break
                # Give planner another chance to finalize/alter tool arguments.
                planning_question = (
                    f"{planning_question}\n\nSystem hint: do not repeat the exact same tool call; "
                    "either finalize or change tool arguments."
                )
                continue

            executed_tool_signatures.add(signature)
            tool_calls_by_intent[route_type] = calls_for_intent + 1
            tools_called.append(tool_name)
            tool_output = await run_in_threadpool(
                execute_agent_tool_call,
                tool_name=tool_name,
                question=planning_question,
                k=k,
                lab_name=planning_lab_name,
                planner_hints=route_plan.planner_parameters,
                arguments=getattr(route_plan, "tool_arguments", {}),
            )
            tool_ok = bool(tool_output.get("ok"))
            step_entry["tool_ok"] = tool_ok
            step_entry["tool_intent"] = tool_output.get("intent")
            observation = summarize_tool_observation(tool_output)
            step_entry["observation"] = observation
            observations.append(observation)
            if tool_ok:
                consecutive_failures = 0
                result_payload = dict(tool_output.get("result") or {})
                answer_text = str(result_payload.get("answer") or observation)
                covered_goals.update(
                    _derive_goals_from_tool_result(
                        route_type=route_type,
                        tool_name=tool_name,
                        answer_text=answer_text,
                    )
                )
                covered_goals.update(_normalize_declared_goal_coverage(route_plan))
                step_entry["covered_goals"] = sorted(list(covered_goals))
            else:
                consecutive_failures += 1
            if consecutive_failures >= max_failures:
                finish_reason = "tool_failures"
                break
            all_goals_met = not required_goals or required_goals.issubset(covered_goals)
            enough_evidence = bool(getattr(route_plan, "enough_evidence", False))
            if enough_evidence and all_goals_met:
                finish_reason = "evidence_sufficient"
                break
            if required_goals and all_goals_met:
                finish_reason = "goal_coverage_complete"
                break
            remaining_goals = sorted(list(required_goals - covered_goals))
            planning_question = (
                f"{effective_question}\n\nAgent tool observations:\n"
                + "\n".join(f"- {item}" for item in observations[-4:])
                + (
                    f"\n\nCoverage status:\n- required_goals: {sorted(list(required_goals))}\n"
                    f"- covered_goals: {sorted(list(covered_goals))}\n"
                    f"- remaining_goals: {remaining_goals}\n"
                    "If remaining_goals is empty, choose action=finalize."
                    if required_goals
                    else ""
                )
            )
            continue

        finish_reason = "finalize"
        break

    if final_runtime is None:
        final_runtime = await resolve_stream_runtime(
            latest_user_question=effective_question,
            lab_name=effective_lab_name,
            allow_clarify=allow_clarify,
            endpoint_key=endpoint_key,
            get_route_decision_contract_fn=get_route_decision_contract_fn,
            get_route_plan_fn=get_route_plan_fn,
            query_scope_class_fn=query_scope_class_fn,
            should_clarify_fn=should_clarify_fn,
            should_use_knowledge_executor_fn=should_use_knowledge_executor_fn,
            resolve_execution_intent_fn=resolve_execution_intent_fn,
        )

    return AgentStreamResolution(
        runtime=final_runtime,
        agent_step_trace=agent_step_trace,
        tools_called=tools_called,
        finish_reason=finish_reason,
        state=state,
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
