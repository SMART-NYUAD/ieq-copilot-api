"""Routed query endpoints (sync + stream)."""

from typing import Dict
import json

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

try:
    from core_settings import load_settings
    from http_schemas import QueryRequest, QueryResponse
    from query_routing.query_orchestrator import (
        build_clarify_prompt,
        build_non_domain_scope_message,
        execute_query,  # compatibility export for tests/mocks
        get_route_decision_contract,
        get_route_plan,
        query_scope_class,  # compatibility export for tests/mocks
        resolve_db_followup_memory,  # compatibility export for tests/mocks
        resolve_execution_intent,  # compatibility export for tests/mocks
        should_clarify,  # compatibility export for tests/mocks
        should_use_knowledge_executor,  # compatibility export for tests/mocks
    )
    from query_routing.synthesizer import build_synthesis_context
    from executors.db_query_executor import prepare_db_query, stream_db_query
    from executors.db_support.response_helpers import serialize_timestamp_value
    from executors.env_query_langchain import (
        stream_answer_env_question,
    )
    from query_routing.query_use_cases import (
        build_stream_clarify_metadata,
        build_stream_db_metadata,
        build_stream_knowledge_metadata,
    )
    from http_routes.query_runtime import (
        build_query_context,
        execute_non_stream_query,
        fetch_db_context,
        fetch_knowledge_stats,
        normalize_allow_clarify,
        normalize_k,
        normalize_lab_name,
        resolve_agent_stream_runtime,
        resolve_stream_runtime,
    )
    from http_routes.route_helpers import (
        SSE_HEADERS,
        attach_policy_metadata,
        attach_conversation_metadata,
        persist_turn,
        route_plan_metadata,
    )
    from runtime_errors import log_exception, stream_error_payload
except ImportError:
    from ..core_settings import load_settings
    from ..http_schemas import QueryRequest, QueryResponse
    from ..query_routing.query_orchestrator import (
        build_clarify_prompt,
        build_non_domain_scope_message,
        execute_query,  # compatibility export for tests/mocks
        get_route_decision_contract,
        get_route_plan,
        query_scope_class,  # compatibility export for tests/mocks
        resolve_db_followup_memory,  # compatibility export for tests/mocks
        resolve_execution_intent,  # compatibility export for tests/mocks
        should_clarify,  # compatibility export for tests/mocks
        should_use_knowledge_executor,  # compatibility export for tests/mocks
    )
    from ..query_routing.synthesizer import build_synthesis_context
    from ..executors.db_query_executor import prepare_db_query, stream_db_query
    from ..executors.db_support.response_helpers import serialize_timestamp_value
    from ..executors.env_query_langchain import (
        stream_answer_env_question,
    )
    from ..query_routing.query_use_cases import (
        build_stream_clarify_metadata,
        build_stream_db_metadata,
        build_stream_knowledge_metadata,
    )
    from .query_runtime import (
        build_query_context,
        execute_non_stream_query,
        fetch_db_context,
        fetch_knowledge_stats,
        normalize_allow_clarify,
        normalize_k,
        normalize_lab_name,
        resolve_agent_stream_runtime,
        resolve_stream_runtime,
    )
    from .route_helpers import (
        SSE_HEADERS,
        attach_policy_metadata,
        attach_conversation_metadata,
        persist_turn,
        route_plan_metadata,
    )
    from ..runtime_errors import log_exception, stream_error_payload


router = APIRouter()
# Compatibility re-exports used by tests that patch route-layer symbols directly.
_ROUTE_TEST_COMPAT = (query_scope_class, resolve_execution_intent, should_clarify, should_use_knowledge_executor)


def _attach_agent_stream_metadata(meta: Dict, route_plan) -> Dict:
    merged = dict(meta or {})
    settings = load_settings()
    merged["agent_mode"] = "enabled" if settings.agentic_mode else "disabled"
    merged["agent_action"] = str(getattr(getattr(route_plan, "agent_action", None), "value", "finalize"))
    merged["tool_name"] = getattr(route_plan, "tool_name", None)
    merged["agent_stream_step_events"] = bool(settings.agent_stream_step_events)
    return merged


def _build_sql_preview_and_bindings(source: Dict, time_window: Dict, resolved_lab_name: str | None) -> Dict:
    metric = str(source.get("metric") or "co2")
    table = str(source.get("table") or "lab_ieq_final")
    lab_scope = source.get("lab_scope") or resolved_lab_name
    window_start = source.get("window_start") or (time_window or {}).get("start")
    window_end = source.get("window_end") or (time_window or {}).get("end")
    sql_preview = "\n".join(
        [
            f"SELECT bucket, {metric}",
            f"FROM {table}",
            "WHERE 1=1",
            "  AND lab_space = :lab_scope",
            "  AND bucket >= :window_start",
            "  AND bucket < :window_end",
            "ORDER BY bucket DESC",
            "LIMIT 200;",
        ]
    )
    return {
        "sql_preview": sql_preview,
        "sql_bindings": {
            "lab_scope": lab_scope,
            "window_start": window_start,
            "window_end": window_end,
        },
    }


@router.post("/query/route")
async def preview_query_route(request: QueryRequest) -> Dict:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    lab_name = normalize_lab_name(request.lab_name)
    allow_clarify = normalize_allow_clarify(request.allow_clarify)
    route_plan = await run_in_threadpool(get_route_plan, question, lab_name)
    route_contract = await run_in_threadpool(
        get_route_decision_contract,
        question,
        lab_name,
        allow_clarify,
        route_plan,
    )
    metadata = route_plan_metadata(route_contract.route_plan)
    metadata["selected_executor"] = route_contract.executor.value
    metadata["execution_intent"] = route_contract.execution_intent_value
    metadata["clarification_required"] = route_contract.executor.value == "clarify_gate"
    return attach_policy_metadata(metadata, route_contract)


@router.post("/query/db-proof")
async def query_db_proof(request: QueryRequest) -> Dict:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    lab_name = normalize_lab_name(request.lab_name)
    allow_clarify = normalize_allow_clarify(request.allow_clarify)
    route_plan = await run_in_threadpool(get_route_plan, question, lab_name)
    route_contract = await run_in_threadpool(
        get_route_decision_contract,
        question,
        lab_name,
        allow_clarify,
        route_plan,
    )
    if not route_contract.needs_measured_data:
        return {
            "available": False,
            "reason": "query_not_routed_to_db",
            "executor": route_contract.executor.value,
            "execution_intent": route_contract.execution_intent_value,
        }

    db_context = await fetch_db_context(
        latest_user_question=question,
        execution_intent=route_contract.execution_intent,
        lab_name=lab_name,
        planner_parameters=route_plan.planner_parameters,
        prepare_db_query_fn=prepare_db_query,
    )
    if db_context.get("invariant_violation"):
        return {
            "available": False,
            "reason": "db_invariant_violation",
            "executor": route_contract.executor.value,
            "execution_intent": route_contract.execution_intent_value,
            "invariant_violation": db_context.get("invariant_violation"),
            "time_window": db_context.get("time_window"),
            "resolved_lab_name": db_context.get("resolved_lab_name"),
        }

    rows = list(db_context.get("rows") or [])
    serialized_rows = [serialize_timestamp_value(item) for item in rows[:200]]
    db_source = next(
        (item for item in (db_context.get("sources") or []) if str(item.get("source_kind") or "") == "db_query"),
        {},
    )
    sql_bits = _build_sql_preview_and_bindings(
        source=db_source,
        time_window=dict(db_context.get("time_window") or {}),
        resolved_lab_name=db_context.get("resolved_lab_name"),
    )
    return {
        "available": True,
        "executor": route_contract.executor.value,
        "execution_intent": route_contract.execution_intent_value,
        "time_window": db_context.get("time_window"),
        "resolved_lab_name": db_context.get("resolved_lab_name"),
        "source": db_source,
        "rows_count": len(rows),
        "rows_preview": serialized_rows,
        **sql_bits,
    }


@router.post("/query", response_model=QueryResponse)
async def query_cards(request: QueryRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        k = normalize_k(request.k)
        lab_name = normalize_lab_name(request.lab_name)
        latest_user_question, normalized_conversation_id, conversation_context, context_applied = build_query_context(
            question=question,
            conversation_id=request.conversation_id,
        )
        non_stream = await execute_non_stream_query(
            question=question,
            latest_user_question=latest_user_question,
            conversation_context=conversation_context,
            k=k,
            lab_name=lab_name,
            allow_clarify=normalize_allow_clarify(request.allow_clarify),
            conversation_id=normalized_conversation_id,
            context_applied=context_applied,
            endpoint_key="query_sync",
            execute_query_fn=execute_query,
        )
        result = non_stream["result"]
        turn_index = non_stream["turn_index"]
        metadata = non_stream["metadata"]
        return QueryResponse(
            answer=result["answer"],
            timescale=result["timescale"],
            cards_retrieved=result["cards_retrieved"],
            recent_card=result["recent_card"],
            conversation_id=normalized_conversation_id,
            turn_index=turn_index,
            metadata=metadata,
            visualization_type=result.get("visualization_type", "none"),
            chart=result.get("chart"),
        )
    except Exception as exc:
        code = log_exception(exc, scope="query.non_stream")
        raise HTTPException(status_code=500, detail=f"[{code.value}] Error processing query: {exc}") from exc


@router.post("/query/stream")
async def query_cards_stream(request: QueryRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    k = normalize_k(request.k)
    lab_name = normalize_lab_name(request.lab_name)
    latest_user_question, normalized_conversation_id, conversation_context, context_applied = build_query_context(
        question=question,
        conversation_id=request.conversation_id,
    )
    agent_resolution = await resolve_agent_stream_runtime(
        latest_user_question=latest_user_question,
        k=k,
        lab_name=lab_name,
        allow_clarify=normalize_allow_clarify(request.allow_clarify),
        conversation_context=conversation_context,
        endpoint_key="query_stream",
        get_route_plan_fn=get_route_plan,
        query_scope_class_fn=query_scope_class,
        should_clarify_fn=should_clarify,
        should_use_knowledge_executor_fn=should_use_knowledge_executor,
        resolve_execution_intent_fn=resolve_execution_intent,
    )
    runtime = agent_resolution.runtime
    state = agent_resolution.state
    agent_step_trace = list(agent_resolution.agent_step_trace or [])
    tools_called = list(agent_resolution.tools_called or [])
    agent_finish_reason = str(agent_resolution.finish_reason or "unknown")
    route_plan = runtime.route_plan
    decision = runtime.decision
    memory_meta = {
        "applied": bool(state.effective_question != state.original_question or state.effective_lab != lab_name),
        "carried_lab_name": state.routing_memory.lab_name if state.effective_lab and not lab_name else None,
        "carried_time_phrase": state.routing_memory.time_phrase if state.effective_question != state.original_question else None,
        "carried_metric": state.routing_memory.metric if state.effective_question != state.original_question else None,
    }

    if runtime.should_clarify_response:
        async def clarify_generator():
            clarify_text = build_clarify_prompt(route_plan)
            meta = build_stream_clarify_metadata(
                route_plan=route_plan,
                decision=decision,
                k=k,
                lab_name=lab_name,
                resolved_lab_name=lab_name,
            )
            meta = attach_conversation_metadata(
                meta,
                conversation_id=normalized_conversation_id,
                conversation_context_applied=context_applied,
                turn_index=None,
            )
            meta = attach_policy_metadata(meta, runtime.route_contract)
            meta = _attach_agent_stream_metadata(meta, route_plan)
            meta["agent_step_trace"] = agent_step_trace
            meta["tools_called"] = tools_called
            meta["agent_steps"] = int(len(agent_step_trace))
            meta["agent_finish_reason"] = agent_finish_reason
            yield f"event: meta\ndata: {json.dumps(meta)}\n\n"
            if bool(meta.get("agent_stream_step_events")):
                for step_payload in agent_step_trace:
                    yield f"event: agent_step\ndata: {json.dumps(step_payload)}\n\n"
            lines = clarify_text.splitlines() or [""]
            payload = "\n".join([f"data: {line}" for line in lines])
            yield f"event: token\n{payload}\n\n"
            turn_index = persist_turn(
                conversation_id=normalized_conversation_id,
                question=question,
                answer=clarify_text,
            )
            if turn_index is not None:
                yield (
                    "event: conversation\n"
                    f"data: {json.dumps({'conversation_id': normalized_conversation_id, 'turn_index': turn_index})}\n\n"
                )
            yield "event: done\ndata: [DONE]\n\n"

        return StreamingResponse(
            clarify_generator(),
            media_type="text/event-stream",
            headers=SSE_HEADERS,
        )

    use_knowledge_executor = runtime.use_knowledge_executor
    execution_intent = runtime.execution_intent
    stream_signals = (route_plan.planner_parameters or {}).get("query_signals") or {}
    suppress_live_scope = bool(stream_signals.get("is_hypothetical_conditional")) and not bool(
        stream_signals.get("requests_current_measured_data")
    )
    active_question = state.effective_question
    active_lab_name = state.effective_lab
    knowledge_question = active_question if suppress_live_scope else (
        build_synthesis_context(
            tool_results=None,
            conversation_context=conversation_context,
            question=active_question,
        )
    )
    knowledge_lab_name = None if suppress_live_scope else active_lab_name

    async def event_generator():
        try:
            if use_knowledge_executor:
                knowledge_stats = await fetch_knowledge_stats(
                    knowledge_question,
                    k,
                    knowledge_lab_name,
                )
                meta = build_stream_knowledge_metadata(
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
                meta = attach_conversation_metadata(
                    meta,
                    conversation_id=normalized_conversation_id,
                    conversation_context_applied=context_applied,
                    turn_index=None,
                )
                meta = attach_policy_metadata(meta, runtime.route_contract)
                meta = _attach_agent_stream_metadata(meta, route_plan)
                meta["agent_step_trace"] = agent_step_trace
                meta["tools_called"] = tools_called
                meta["agent_steps"] = int(len(agent_step_trace))
                meta["agent_finish_reason"] = agent_finish_reason
                yield f"event: meta\ndata: {json.dumps(meta)}\n\n"
                if bool(meta.get("agent_stream_step_events")):
                    for step_payload in agent_step_trace:
                        yield f"event: agent_step\ndata: {json.dumps(step_payload)}\n\n"
                assembled_answer = ""
                async for chunk in stream_answer_env_question(
                    user_question=knowledge_question,
                    k=max(1, min(k, 8)),
                    space=knowledge_lab_name,
                ):
                    assembled_answer += chunk or ""
                    lines = (chunk or "").splitlines() or [""]
                    payload = "\n".join([f"data: {line}" for line in lines])
                    yield f"event: token\n{payload}\n\n"
                turn_index = persist_turn(
                    conversation_id=normalized_conversation_id,
                    question=question,
                    answer=assembled_answer,
                )
                if turn_index is not None:
                    yield (
                        "event: conversation\n"
                        f"data: {json.dumps({'conversation_id': normalized_conversation_id, 'turn_index': turn_index})}\n\n"
                    )
                yield "event: done\ndata: [DONE]\n\n"
                return

            effective_question = active_question
            effective_lab_name = active_lab_name
            db_context = await fetch_db_context(
                latest_user_question=effective_question,
                execution_intent=execution_intent,
                lab_name=effective_lab_name,
                planner_parameters=route_plan.planner_parameters,
                prepare_db_query_fn=prepare_db_query,
            )
            if db_context.get("invariant_violation"):
                meta = build_stream_clarify_metadata(
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
                meta = attach_conversation_metadata(
                    meta,
                    conversation_id=normalized_conversation_id,
                    conversation_context_applied=context_applied,
                    turn_index=None,
                )
                meta = attach_policy_metadata(meta, runtime.route_contract)
                meta = _attach_agent_stream_metadata(meta, route_plan)
                meta["agent_step_trace"] = agent_step_trace
                meta["tools_called"] = tools_called
                meta["agent_steps"] = int(len(agent_step_trace))
                meta["agent_finish_reason"] = agent_finish_reason
                yield f"event: meta\ndata: {json.dumps(meta)}\n\n"
                if bool(meta.get("agent_stream_step_events")):
                    for step_payload in agent_step_trace:
                        yield f"event: agent_step\ndata: {json.dumps(step_payload)}\n\n"
                clarify_text = str(db_context.get("fallback_answer") or "")
                lines = clarify_text.splitlines() or [""]
                payload = "\n".join([f"data: {line}" for line in lines])
                yield f"event: token\n{payload}\n\n"
                turn_index = persist_turn(
                    conversation_id=normalized_conversation_id,
                    question=question,
                    answer=clarify_text,
                )
                if turn_index is not None:
                    yield (
                        "event: conversation\n"
                        f"data: {json.dumps({'conversation_id': normalized_conversation_id, 'turn_index': turn_index})}\n\n"
                    )
                yield "event: done\ndata: [DONE]\n\n"
                return
            meta = build_stream_db_metadata(
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
            meta = attach_conversation_metadata(
                meta,
                conversation_id=normalized_conversation_id,
                conversation_context_applied=context_applied,
                turn_index=None,
            )
            meta = attach_policy_metadata(meta, runtime.route_contract)
            meta = _attach_agent_stream_metadata(meta, route_plan)
            meta["agent_step_trace"] = agent_step_trace
            meta["tools_called"] = tools_called
            meta["agent_steps"] = int(len(agent_step_trace))
            meta["agent_finish_reason"] = agent_finish_reason
            yield f"event: meta\ndata: {json.dumps(meta)}\n\n"
            if bool(meta.get("agent_stream_step_events")):
                for step_payload in agent_step_trace:
                    yield f"event: agent_step\ndata: {json.dumps(step_payload)}\n\n"

            assembled_answer = ""
            async for chunk in stream_db_query(
                question=effective_question,
                intent=execution_intent,
                lab_name=effective_lab_name,
                planner_hints=route_plan.planner_parameters,
                query_context=db_context,
            ):
                assembled_answer += chunk or ""
                lines = (chunk or "").splitlines() or [""]
                payload = "\n".join([f"data: {line}" for line in lines])
                yield f"event: token\n{payload}\n\n"

            turn_index = persist_turn(
                conversation_id=normalized_conversation_id,
                question=question,
                answer=assembled_answer,
            )
            if turn_index is not None:
                yield (
                    "event: conversation\n"
                    f"data: {json.dumps({'conversation_id': normalized_conversation_id, 'turn_index': turn_index})}\n\n"
                )

            yield "event: done\ndata: [DONE]\n\n"
        except Exception as exc:
            payload = stream_error_payload(
                exc,
                scope="query.stream",
                extra={"lab_name": lab_name, "k": k},
            )
            yield f"event: error\ndata: {json.dumps(payload)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )

