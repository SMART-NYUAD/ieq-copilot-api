"""Routed query endpoints (sync + stream)."""

from typing import Dict
import json

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

try:
    from http_schemas import QueryRequest, QueryResponse
    from evidence.citation_processor import (
        build_numbered_sources_block,
        extract_citation_indices_from_answer,
    )
    from query_routing.query_orchestrator import (
        build_clarify_prompt,
        execute_query,  # compatibility export for tests/mocks
        get_route_decision_contract,
        get_route_plan,
        query_scope_class,  # compatibility export for tests/mocks
        resolve_execution_intent,  # compatibility export for tests/mocks
        should_clarify,  # compatibility export for tests/mocks
        should_use_knowledge_executor,  # compatibility export for tests/mocks
    )
    from executors.db_query_executor import prepare_db_query, stream_db_query
    from executors.db_support.response_helpers import serialize_timestamp_value
    from executors.env_query_langchain import (
        get_guideline_records_for_question,
        stream_answer_env_question,
    )
    from query_routing.metadata_builders import (
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
    )
    from http_routes.route_helpers import (
        SSE_HEADERS,
        attach_policy_metadata,
        attach_conversation_metadata,
        persist_turn,
        route_plan_metadata,
    )
    from http_routes.stream_shared import (
        attach_agent_stream_metadata as _shared_attach_agent_stream_metadata,
        prepare_stream_execution_context,
    )
    from runtime_errors import log_exception, stream_error_payload
except ImportError:
    from ..http_schemas import QueryRequest, QueryResponse
    from ..evidence.citation_processor import (
        build_numbered_sources_block,
        extract_citation_indices_from_answer,
    )
    from ..query_routing.query_orchestrator import (
        build_clarify_prompt,
        execute_query,  # compatibility export for tests/mocks
        get_route_decision_contract,
        get_route_plan,
        query_scope_class,  # compatibility export for tests/mocks
        resolve_execution_intent,  # compatibility export for tests/mocks
        should_clarify,  # compatibility export for tests/mocks
        should_use_knowledge_executor,  # compatibility export for tests/mocks
    )
    from ..executors.db_query_executor import prepare_db_query, stream_db_query
    from ..executors.db_support.response_helpers import serialize_timestamp_value
    from ..executors.env_query_langchain import (
        get_guideline_records_for_question,
        stream_answer_env_question,
    )
    from ..query_routing.metadata_builders import (
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
    )
    from .route_helpers import (
        SSE_HEADERS,
        attach_policy_metadata,
        attach_conversation_metadata,
        persist_turn,
        route_plan_metadata,
    )
    from .stream_shared import (
        attach_agent_stream_metadata as _shared_attach_agent_stream_metadata,
        prepare_stream_execution_context,
    )
    from ..runtime_errors import log_exception, stream_error_payload


router = APIRouter()
# Compatibility re-exports used by tests that patch route-layer symbols directly.
_ROUTE_TEST_COMPAT = (query_scope_class, resolve_execution_intent, should_clarify, should_use_knowledge_executor)


def _attach_agent_stream_metadata(meta: Dict, route_plan) -> Dict:
    return _shared_attach_agent_stream_metadata(meta, route_plan)


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
            footnotes=list(result.get("footnotes") or []),
            citation_sources=list(result.get("citation_sources") or result.get("indexed_sources") or []),
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
    stream_ctx = await prepare_stream_execution_context(
        latest_user_question=latest_user_question,
        k=k,
        lab_name=lab_name,
        allow_clarify=normalize_allow_clarify(request.allow_clarify),
        conversation_context=conversation_context,
        endpoint_key="query_stream",
        resolve_agent_stream_runtime_fn=resolve_agent_stream_runtime,
        get_route_plan_fn=get_route_plan,
        query_scope_class_fn=query_scope_class,
        should_clarify_fn=should_clarify,
        should_use_knowledge_executor_fn=should_use_knowledge_executor,
        resolve_execution_intent_fn=resolve_execution_intent,
        fetch_knowledge_stats_fn=fetch_knowledge_stats,
        fetch_db_context_fn=fetch_db_context,
        prepare_db_query_fn=prepare_db_query,
        build_stream_clarify_metadata_fn=build_stream_clarify_metadata,
        build_stream_knowledge_metadata_fn=build_stream_knowledge_metadata,
        build_stream_db_metadata_fn=build_stream_db_metadata,
    )
    runtime = stream_ctx.runtime
    route_plan = stream_ctx.route_plan
    decision = stream_ctx.decision
    agent_step_trace = list(stream_ctx.agent_step_trace or [])
    tools_called = list(stream_ctx.tools_called or [])
    agent_finish_reason = stream_ctx.agent_finish_reason

    if stream_ctx.mode == "clarify":
        async def clarify_generator():
            clarify_text = build_clarify_prompt(route_plan)
            meta = dict(stream_ctx.meta or {})
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
    execution_intent = stream_ctx.execution_intent

    async def event_generator():
        try:
            db_context = stream_ctx.db_context or {}
            db_indexed_sources = list(db_context.get("indexed_sources") or [])
            if not db_indexed_sources:
                _, db_indexed_sources = build_numbered_sources_block(
                    list(db_context.get("guideline_records") or [])
                )
                db_context["indexed_sources"] = db_indexed_sources
            knowledge_indexed_sources = []
            effective_meta_sources = db_indexed_sources
            knowledge_guideline_records = []
            if use_knowledge_executor:
                knowledge_guideline_records = await run_in_threadpool(
                    get_guideline_records_for_question,
                    str(stream_ctx.effective_question or latest_user_question),
                    3,
                )
                _, effective_meta_sources = build_numbered_sources_block(knowledge_guideline_records)

            if use_knowledge_executor:
                meta = dict(stream_ctx.meta or {})
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
                meta["citation_sources"] = effective_meta_sources
                yield f"event: meta\ndata: {json.dumps(meta)}\n\n"
                if bool(meta.get("agent_stream_step_events")):
                    for step_payload in agent_step_trace:
                        yield f"event: agent_step\ndata: {json.dumps(step_payload)}\n\n"
                assembled_answer = ""
                async for chunk in stream_answer_env_question(
                    user_question=str(stream_ctx.knowledge_question or ""),
                    k=max(1, min(k, 8)),
                    space=stream_ctx.knowledge_lab_name,
                    guideline_records=knowledge_guideline_records,
                    indexed_sources_out=knowledge_indexed_sources,
                ):
                    assembled_answer += chunk or ""
                    lines = (chunk or "").splitlines() or [""]
                    payload = "\n".join([f"data: {line}" for line in lines])
                    yield f"event: token\n{payload}\n\n"
                used_sources = extract_citation_indices_from_answer(
                    answer_text=assembled_answer,
                    indexed_sources=knowledge_indexed_sources or effective_meta_sources,
                )
                if used_sources:
                    yield f"event: citations\ndata: {json.dumps(used_sources)}\n\n"
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

            if stream_ctx.mode == "db_clarify":
                meta = dict(stream_ctx.meta or {})
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
            meta = dict(stream_ctx.meta or {})
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
            meta["citation_sources"] = db_indexed_sources
            yield f"event: meta\ndata: {json.dumps(meta)}\n\n"
            if bool(meta.get("agent_stream_step_events")):
                for step_payload in agent_step_trace:
                    yield f"event: agent_step\ndata: {json.dumps(step_payload)}\n\n"

            assembled_answer = ""
            async for chunk in stream_db_query(
                question=str(stream_ctx.effective_question or ""),
                intent=execution_intent,
                lab_name=stream_ctx.effective_lab_name,
                planner_hints=route_plan.planner_parameters,
                query_context=db_context,
            ):
                assembled_answer += chunk or ""
                lines = (chunk or "").splitlines() or [""]
                payload = "\n".join([f"data: {line}" for line in lines])
                yield f"event: token\n{payload}\n\n"
            used_sources = extract_citation_indices_from_answer(
                answer_text=assembled_answer,
                indexed_sources=db_indexed_sources,
            )
            if used_sources:
                yield f"event: citations\ndata: {json.dumps(used_sources)}\n\n"

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

