"""Shared stream runtime preparation for HTTP adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

try:
    from core_settings import load_settings
    from query_routing.synthesizer import build_synthesis_context
except ImportError:
    from ..core_settings import load_settings
    from ..query_routing.synthesizer import build_synthesis_context


@dataclass(frozen=True)
class StreamExecutionContext:
    mode: str  # clarify | knowledge | db | db_clarify
    runtime: Any
    state: Any
    route_plan: Any
    decision: Any
    execution_intent: Any
    memory_meta: Dict[str, Any]
    meta: Dict[str, Any]
    knowledge_question: Optional[str] = None
    knowledge_lab_name: Optional[str] = None
    effective_question: Optional[str] = None
    effective_lab_name: Optional[str] = None
    db_context: Optional[Dict[str, Any]] = None
    agent_step_trace: Optional[list[Dict[str, Any]]] = None
    tools_called: Optional[list[str]] = None
    agent_finish_reason: str = "unknown"


def build_memory_meta(state: Any, requested_lab_name: Optional[str]) -> Dict[str, Any]:
    return {
        "applied": bool(state.effective_question != state.original_question or state.effective_lab != requested_lab_name),
        "carried_lab_name": state.routing_memory.lab_name if state.effective_lab and not requested_lab_name else None,
        "carried_time_phrase": (
            state.routing_memory.time_phrase if state.effective_question != state.original_question else None
        ),
        "carried_metric": state.routing_memory.metric if state.effective_question != state.original_question else None,
    }


def attach_agent_stream_metadata(
    meta: Dict[str, Any],
    route_plan: Any,
    *,
    include_expected_observation: bool = False,
) -> Dict[str, Any]:
    merged = dict(meta or {})
    settings = load_settings()
    merged["agent_mode"] = "enabled" if bool(getattr(settings, "agentic_mode", False)) else "disabled"
    merged["agent_action"] = str(getattr(getattr(route_plan, "agent_action", None), "value", "finalize"))
    merged["tool_name"] = getattr(route_plan, "tool_name", None)
    if include_expected_observation:
        merged["expected_observation"] = getattr(route_plan, "expected_observation", None)
    merged["agent_stream_step_events"] = bool(getattr(settings, "agent_stream_step_events", False))
    return merged


async def prepare_stream_execution_context(
    *,
    latest_user_question: str,
    k: int,
    lab_name: Optional[str],
    allow_clarify: bool,
    conversation_context: str,
    endpoint_key: str,
    resolve_agent_stream_runtime_fn: Any,
    get_route_plan_fn: Any,
    query_scope_class_fn: Any,
    should_clarify_fn: Any,
    should_use_knowledge_executor_fn: Any,
    resolve_execution_intent_fn: Any,
    fetch_knowledge_stats_fn: Any,
    fetch_db_context_fn: Any,
    prepare_db_query_fn: Any,
    build_stream_clarify_metadata_fn: Any,
    build_stream_knowledge_metadata_fn: Any,
    build_stream_db_metadata_fn: Any,
) -> StreamExecutionContext:
    agent_resolution = await resolve_agent_stream_runtime_fn(
        latest_user_question=latest_user_question,
        k=k,
        lab_name=lab_name,
        allow_clarify=allow_clarify,
        conversation_context=conversation_context,
        endpoint_key=endpoint_key,
        get_route_plan_fn=get_route_plan_fn,
        query_scope_class_fn=query_scope_class_fn,
        should_clarify_fn=should_clarify_fn,
        should_use_knowledge_executor_fn=should_use_knowledge_executor_fn,
        resolve_execution_intent_fn=resolve_execution_intent_fn,
    )
    runtime = agent_resolution.runtime
    state = agent_resolution.state
    route_plan = runtime.route_plan
    decision = runtime.decision
    execution_intent = runtime.execution_intent
    memory_meta = build_memory_meta(state=state, requested_lab_name=lab_name)
    agent_step_trace = list(agent_resolution.agent_step_trace or [])
    tools_called = list(agent_resolution.tools_called or [])
    agent_finish_reason = str(agent_resolution.finish_reason or "unknown")

    if runtime.should_clarify_response:
        meta = build_stream_clarify_metadata_fn(
            route_plan=route_plan,
            decision=decision,
            k=k,
            lab_name=lab_name,
            resolved_lab_name=state.effective_lab,
        )
        return StreamExecutionContext(
            mode="clarify",
            runtime=runtime,
            state=state,
            route_plan=route_plan,
            decision=decision,
            execution_intent=execution_intent,
            memory_meta=memory_meta,
            meta=meta,
            effective_question=state.effective_question,
            effective_lab_name=state.effective_lab,
            agent_step_trace=agent_step_trace,
            tools_called=tools_called,
            agent_finish_reason=agent_finish_reason,
        )

    stream_signals = (route_plan.planner_parameters or {}).get("query_signals") or {}
    suppress_live_scope = bool(stream_signals.get("is_hypothetical_conditional")) and not bool(
        stream_signals.get("requests_current_measured_data")
    )
    if runtime.use_knowledge_executor:
        knowledge_question = (
            state.effective_question
            if suppress_live_scope
            else build_synthesis_context(
                tool_results=None,
                conversation_context=conversation_context,
                question=state.effective_question,
            )
        )
        knowledge_lab_name = None if suppress_live_scope else state.effective_lab
        knowledge_stats = await fetch_knowledge_stats_fn(
            knowledge_question,
            k,
            knowledge_lab_name,
        )
        meta = build_stream_knowledge_metadata_fn(
            route_plan=route_plan,
            decision=decision,
            k=k,
            lab_name=lab_name,
            resolved_lab_name=knowledge_lab_name,
            cards_retrieved=int(knowledge_stats.get("cards_retrieved") or 0),
            knowledge_cards_retrieved=int(knowledge_stats.get("knowledge_cards_retrieved") or 0),
        )
        meta["memory_carryover_applied"] = bool(memory_meta.get("applied"))
        meta["memory_carried_lab_name"] = memory_meta.get("carried_lab_name")
        meta["memory_carried_time_phrase"] = memory_meta.get("carried_time_phrase")
        meta["memory_carried_metric"] = memory_meta.get("carried_metric")
        return StreamExecutionContext(
            mode="knowledge",
            runtime=runtime,
            state=state,
            route_plan=route_plan,
            decision=decision,
            execution_intent=execution_intent,
            memory_meta=memory_meta,
            meta=meta,
            knowledge_question=knowledge_question,
            knowledge_lab_name=knowledge_lab_name,
            agent_step_trace=agent_step_trace,
            tools_called=tools_called,
            agent_finish_reason=agent_finish_reason,
        )

    db_context = await fetch_db_context_fn(
        latest_user_question=state.effective_question,
        execution_intent=execution_intent,
        lab_name=state.effective_lab,
        planner_parameters=route_plan.planner_parameters,
        prepare_db_query_fn=prepare_db_query_fn,
    )
    if db_context.get("invariant_violation"):
        meta = build_stream_clarify_metadata_fn(
            route_plan=route_plan,
            decision=decision,
            k=k,
            lab_name=lab_name,
            resolved_lab_name=db_context.get("resolved_lab_name"),
            time_window=db_context.get("time_window"),
            invariant_violation=db_context.get("invariant_violation"),
        )
        meta["execution_intent"] = execution_intent.value
        meta["intent_rerouted_to_db"] = execution_intent != decision.intent
        meta["memory_carryover_applied"] = bool(memory_meta.get("applied"))
        meta["memory_carried_lab_name"] = memory_meta.get("carried_lab_name")
        meta["memory_carried_time_phrase"] = memory_meta.get("carried_time_phrase")
        meta["memory_carried_metric"] = memory_meta.get("carried_metric")
        return StreamExecutionContext(
            mode="db_clarify",
            runtime=runtime,
            state=state,
            route_plan=route_plan,
            decision=decision,
            execution_intent=execution_intent,
            memory_meta=memory_meta,
            meta=meta,
            db_context=db_context,
            effective_question=state.effective_question,
            effective_lab_name=state.effective_lab,
            agent_step_trace=agent_step_trace,
            tools_called=tools_called,
            agent_finish_reason=agent_finish_reason,
        )

    meta = build_stream_db_metadata_fn(
        route_plan=route_plan,
        decision=decision,
        execution_intent=execution_intent,
        k=k,
        lab_name=lab_name,
        db_context=db_context,
    )
    meta["memory_carryover_applied"] = bool(memory_meta.get("applied"))
    meta["memory_carried_lab_name"] = memory_meta.get("carried_lab_name")
    meta["memory_carried_time_phrase"] = memory_meta.get("carried_time_phrase")
    meta["memory_carried_metric"] = memory_meta.get("carried_metric")
    return StreamExecutionContext(
        mode="db",
        runtime=runtime,
        state=state,
        route_plan=route_plan,
        decision=decision,
        execution_intent=execution_intent,
        memory_meta=memory_meta,
        meta=meta,
        db_context=db_context,
        effective_question=state.effective_question,
        effective_lab_name=state.effective_lab,
        agent_step_trace=agent_step_trace,
        tools_called=tools_called,
        agent_finish_reason=agent_finish_reason,
    )
