"""Routed query endpoints (sync + stream)."""

from typing import Dict
import json

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

try:
    from http_schemas import QueryRequest, QueryResponse
    from query_routing.query_orchestrator import (
        build_clarify_prompt,
        build_non_domain_scope_message,
        execute_query,  # compatibility export for tests/mocks
        get_route_plan,
        query_scope_class,  # compatibility export for tests/mocks
        resolve_execution_intent,  # compatibility export for tests/mocks
        should_clarify,  # compatibility export for tests/mocks
        should_use_knowledge_executor,  # compatibility export for tests/mocks
    )
    from executors.db_query_executor import prepare_db_query, stream_db_query
    from executors.env_query_langchain import (
        stream_answer_env_question,
    )
    from http_routes.query_runtime import (
        build_query_context,
        execute_non_stream_query,
        fetch_db_context,
        fetch_knowledge_stats,
        normalize_allow_clarify,
        normalize_k,
        normalize_lab_name,
        resolve_stream_runtime,
    )
    from http_routes.route_helpers import (
        SSE_HEADERS,
        attach_conversation_metadata,
        persist_turn,
        route_plan_metadata,
    )
    from runtime_errors import log_exception, stream_error_payload
except ImportError:
    from ..http_schemas import QueryRequest, QueryResponse
    from ..query_routing.query_orchestrator import (
        build_clarify_prompt,
        build_non_domain_scope_message,
        execute_query,  # compatibility export for tests/mocks
        get_route_plan,
        query_scope_class,  # compatibility export for tests/mocks
        resolve_execution_intent,  # compatibility export for tests/mocks
        should_clarify,  # compatibility export for tests/mocks
        should_use_knowledge_executor,  # compatibility export for tests/mocks
    )
    from ..executors.db_query_executor import prepare_db_query, stream_db_query
    from ..executors.env_query_langchain import (
        stream_answer_env_question,
    )
    from .query_runtime import (
        build_query_context,
        execute_non_stream_query,
        fetch_db_context,
        fetch_knowledge_stats,
        normalize_allow_clarify,
        normalize_k,
        normalize_lab_name,
        resolve_stream_runtime,
    )
    from .route_helpers import (
        SSE_HEADERS,
        attach_conversation_metadata,
        persist_turn,
        route_plan_metadata,
    )
    from ..runtime_errors import log_exception, stream_error_payload


router = APIRouter()

@router.post("/query/route")
async def preview_query_route(request: QueryRequest) -> Dict:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    lab_name = normalize_lab_name(request.lab_name)
    route_plan = await run_in_threadpool(get_route_plan, question, lab_name)
    return route_plan_metadata(route_plan)


@router.post("/query", response_model=QueryResponse)
async def query_cards(request: QueryRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        k = normalize_k(request.k)
        lab_name = normalize_lab_name(request.lab_name)
        effective_question, normalized_conversation_id, context_applied = build_query_context(
            question=question,
            conversation_id=request.conversation_id,
        )
        non_stream = await execute_non_stream_query(
            question=question,
            effective_question=effective_question,
            k=k,
            lab_name=lab_name,
            allow_clarify=normalize_allow_clarify(request.allow_clarify),
            conversation_id=normalized_conversation_id,
            context_applied=context_applied,
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
    effective_question, normalized_conversation_id, context_applied = build_query_context(
        question=question,
        conversation_id=request.conversation_id,
    )
    runtime = await resolve_stream_runtime(
        effective_question=effective_question,
        lab_name=lab_name,
        allow_clarify=normalize_allow_clarify(request.allow_clarify),
        get_route_plan_fn=get_route_plan,
        query_scope_class_fn=query_scope_class,
        should_clarify_fn=should_clarify,
        should_use_knowledge_executor_fn=should_use_knowledge_executor,
        resolve_execution_intent_fn=resolve_execution_intent,
    )
    route_plan = runtime.route_plan
    decision = runtime.decision

    if runtime.should_clarify_response:
        async def clarify_generator():
            clarify_text = build_clarify_prompt(route_plan)
            meta = {
                "timescale": "clarify",
                "cards_retrieved": 0,
                "recent_card": False,
                **route_plan_metadata(route_plan),
                "clarification_required": True,
                "executor": "clarify_gate",
                "k_requested": k,
                "lab_name": lab_name,
                "resolved_lab_name": lab_name,
                "llm_used": False,
                "time_window": None,
                "sources": [],
                "forecast_model": None,
                "forecast_confidence": None,
                "forecast_confidence_score": None,
                "forecast_horizon_hours": None,
                "visualization_type": "none",
                "chart": None,
            }
            meta = attach_conversation_metadata(
                meta,
                conversation_id=normalized_conversation_id,
                conversation_context_applied=context_applied,
                turn_index=None,
            )
            yield f"event: meta\ndata: {json.dumps(meta)}\n\n"
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

    async def event_generator():
        try:
            if runtime.scope_class == "non_domain":
                guardrail_text = build_non_domain_scope_message(question)
                meta = {
                    "timescale": "knowledge",
                    "cards_retrieved": 0,
                    "knowledge_cards_retrieved": 0,
                    "recent_card": False,
                    **route_plan_metadata(route_plan, query_scope_override=runtime.scope_class),
                    "execution_intent": decision.intent.value,
                    "intent_rerouted_to_db": False,
                    "clarification_required": False,
                    "executor": "scope_guardrail",
                    "k_requested": k,
                    "lab_name": lab_name,
                    "resolved_lab_name": lab_name,
                    "llm_used": False,
                    "time_window": None,
                    "sources": [],
                    "forecast_model": None,
                    "forecast_confidence": None,
                    "forecast_confidence_score": None,
                    "forecast_horizon_hours": None,
                    "visualization_type": "none",
                    "chart": None,
                }
                meta = attach_conversation_metadata(
                    meta,
                    conversation_id=normalized_conversation_id,
                    conversation_context_applied=context_applied,
                    turn_index=None,
                )
                yield f"event: meta\ndata: {json.dumps(meta)}\n\n"
                lines = guardrail_text.splitlines() or [""]
                payload = "\n".join([f"data: {line}" for line in lines])
                yield f"event: token\n{payload}\n\n"
                turn_index = persist_turn(
                    conversation_id=normalized_conversation_id,
                    question=question,
                    answer=guardrail_text,
                )
                if turn_index is not None:
                    yield (
                        "event: conversation\n"
                        f"data: {json.dumps({'conversation_id': normalized_conversation_id, 'turn_index': turn_index})}\n\n"
                    )
                yield "event: done\ndata: [DONE]\n\n"
                return

            if use_knowledge_executor:
                knowledge_stats = await fetch_knowledge_stats(
                    effective_question,
                    k,
                    lab_name,
                    stats_fn=get_knowledge_context_stats,
                )
                meta = {
                    "timescale": "knowledge",
                    "cards_retrieved": int(knowledge_stats.get("cards_retrieved") or 0),
                    "knowledge_cards_retrieved": int(
                        knowledge_stats.get("knowledge_cards_retrieved") or 0
                    ),
                    "recent_card": False,
                    **route_plan_metadata(route_plan),
                    "execution_intent": decision.intent.value,
                    "intent_rerouted_to_db": False,
                    "clarification_required": False,
                    "executor": "knowledge_qa",
                    "k_requested": k,
                    "lab_name": lab_name,
                    "resolved_lab_name": lab_name,
                    "llm_used": True,
                    "time_window": None,
                    "sources": [],
                    "forecast_model": None,
                    "forecast_confidence": None,
                    "forecast_confidence_score": None,
                    "forecast_horizon_hours": None,
                    "visualization_type": "none",
                    "chart": None,
                }
                meta = attach_conversation_metadata(
                    meta,
                    conversation_id=normalized_conversation_id,
                    conversation_context_applied=context_applied,
                    turn_index=None,
                )
                yield f"event: meta\ndata: {json.dumps(meta)}\n\n"
                assembled_answer = ""
                async for chunk in stream_answer_env_question(
                    user_question=effective_question, k=max(1, min(k, 8)), space=lab_name
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

            db_context = await fetch_db_context(
                effective_question=effective_question,
                execution_intent=execution_intent,
                lab_name=lab_name,
                planner_parameters=route_plan.planner_parameters,
                prepare_db_query_fn=prepare_db_query,
            )
            meta = {
                "timescale": db_context["timescale"],
                    "cards_retrieved": int(db_context.get("cards_retrieved") or 0),
                "recent_card": False,
                **route_plan_metadata(route_plan),
                "execution_intent": execution_intent.value,
                "intent_rerouted_to_db": execution_intent != decision.intent,
                "clarification_required": False,
                "executor": "db_query",
                "k_requested": k,
                "lab_name": lab_name,
                "resolved_lab_name": db_context.get("resolved_lab_name"),
                "llm_used": True,
                "time_window": db_context.get("time_window"),
                "sources": db_context.get("sources", []),
                "forecast_model": (db_context.get("forecast") or {}).get("model"),
                "forecast_confidence": (db_context.get("forecast") or {}).get("confidence"),
                "forecast_confidence_score": (db_context.get("forecast") or {}).get("confidence_score"),
                "forecast_horizon_hours": (db_context.get("forecast") or {}).get("horizon_hours"),
                "visualization_type": db_context.get("visualization_type", "none"),
                "chart": db_context.get("chart"),
            }
            meta = attach_conversation_metadata(
                meta,
                conversation_id=normalized_conversation_id,
                conversation_context_applied=context_applied,
                turn_index=None,
            )
            yield f"event: meta\ndata: {json.dumps(meta)}\n\n"

            assembled_answer = ""
            async for chunk in stream_db_query(
                question=effective_question,
                intent=execution_intent,
                lab_name=lab_name,
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

