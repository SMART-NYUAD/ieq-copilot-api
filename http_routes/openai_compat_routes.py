"""OpenAI-compatible API facade for chat completions."""

from datetime import datetime, timezone
import json
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

try:
    from core_settings import load_settings
    from query_routing.query_orchestrator import (
        build_clarify_prompt,
        execute_query,  # compatibility export for tests/mocks
        get_route_plan,  # compatibility export for tests/mocks
        query_scope_class,  # compatibility export for tests/mocks
        resolve_db_followup_memory,  # compatibility export for tests/mocks
        resolve_execution_intent,  # compatibility export for tests/mocks
        should_clarify,  # compatibility export for tests/mocks
        should_use_knowledge_executor,  # compatibility export for tests/mocks
    )
    from query_routing.synthesizer import build_synthesis_context
    from executors.db_query_executor import prepare_db_query, stream_db_query
    from executors.env_query_langchain import (
        get_knowledge_context_stats,
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
        resolve_agent_stream_runtime,
        resolve_stream_runtime,
    )
    from query_routing.query_use_cases import (
        build_stream_clarify_metadata,
        build_stream_db_metadata,
        build_stream_knowledge_metadata,
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
    from ..query_routing.query_orchestrator import (
        build_clarify_prompt,
        execute_query,  # compatibility export for tests/mocks
        get_route_plan,  # compatibility export for tests/mocks
        query_scope_class,  # compatibility export for tests/mocks
        resolve_db_followup_memory,  # compatibility export for tests/mocks
        resolve_execution_intent,  # compatibility export for tests/mocks
        should_clarify,  # compatibility export for tests/mocks
        should_use_knowledge_executor,  # compatibility export for tests/mocks
    )
    from ..query_routing.synthesizer import build_synthesis_context
    from ..executors.db_query_executor import prepare_db_query, stream_db_query
    from ..executors.env_query_langchain import (
        get_knowledge_context_stats,
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
        resolve_agent_stream_runtime,
        resolve_stream_runtime,
    )
    from ..query_routing.query_use_cases import (
        build_stream_clarify_metadata,
        build_stream_db_metadata,
        build_stream_knowledge_metadata,
    )
    from .route_helpers import (
        SSE_HEADERS,
        attach_policy_metadata,
        attach_conversation_metadata,
        persist_turn,
        route_plan_metadata,
    )
    from ..runtime_errors import log_exception, stream_error_payload


class OpenAIMessage(BaseModel):
    role: str
    content: str


class OpenAIChatCompletionRequest(BaseModel):
    model: str = "rag-router"
    messages: List[OpenAIMessage]
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    user: Optional[str] = None
    k: Optional[int] = Field(default=5, ge=1, le=50)
    lab_name: Optional[str] = None
    think: Optional[bool] = None
    allow_clarify: Optional[bool] = True
    conversation_id: Optional[str] = None
    turn_index: Optional[int] = None


router = APIRouter(prefix="/v1", tags=["openai-compatible"])
# Compatibility re-exports used by tests that patch route-layer symbols directly.
_ROUTE_TEST_COMPAT = (get_route_plan, query_scope_class, resolve_execution_intent, should_clarify, should_use_knowledge_executor)


def _attach_agent_stream_metadata(meta: Dict[str, Any], route_plan: Any) -> Dict[str, Any]:
    merged = dict(meta or {})
    settings = load_settings()
    merged["agent_mode"] = "enabled" if bool(getattr(settings, "agentic_mode", False)) else "disabled"
    merged["agent_action"] = str(getattr(getattr(route_plan, "agent_action", None), "value", "finalize"))
    merged["tool_name"] = getattr(route_plan, "tool_name", None)
    merged["expected_observation"] = getattr(route_plan, "expected_observation", None)
    merged["agent_stream_step_events"] = bool(getattr(settings, "agent_stream_step_events", False))
    return merged


def _extract_last_user_message(messages: List[OpenAIMessage]) -> str:
    for msg in reversed(messages):
        if msg.role == "user" and (msg.content or "").strip():
            return msg.content.strip()
    return ""


def _build_non_stream_response(
    request: OpenAIChatCompletionRequest,
    answer: str,
    query_metadata: Dict[str, Any],
    chart: Optional[Dict[str, Any]] = None,
    visualization_type: Optional[str] = "none",
) -> Dict[str, Any]:
    created = int(time.time())
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    answer_tokens = max(1, len(answer.split()))
    prompt_tokens = sum(max(1, len((m.content or "").split())) for m in request.messages)
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": answer_tokens,
            "total_tokens": prompt_tokens + answer_tokens,
        },
        "system_fingerprint": "rag-router-v1",
        "x_router": query_metadata,
        "x_visualization_type": visualization_type or "none",
        "x_chart": chart,
    }


@router.get("/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "rag-router",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "rag_api_server",
            }
        ],
    }


@router.post("/chat/completions")
async def openai_chat_completions(request: OpenAIChatCompletionRequest):
    question = _extract_last_user_message(request.messages)
    if not question:
        raise HTTPException(status_code=400, detail="No user message found in messages.")

    k = normalize_k(request.k)
    lab_name = normalize_lab_name(request.lab_name)
    latest_user_question, normalized_conversation_id, conversation_context, context_applied = build_query_context(
        question=question,
        conversation_id=request.conversation_id,
    )

    if not request.stream:
        try:
            non_stream = await execute_non_stream_query(
                question=question,
                latest_user_question=latest_user_question,
                conversation_context=conversation_context,
                k=k,
                lab_name=lab_name,
                allow_clarify=normalize_allow_clarify(request.allow_clarify),
                conversation_id=normalized_conversation_id,
                context_applied=context_applied,
                endpoint_key="openai_sync",
                execute_query_fn=execute_query,
            )
            result = non_stream["result"]
            query_metadata = non_stream["metadata"]
            return _build_non_stream_response(
                request=request,
                answer=result["answer"],
                query_metadata=query_metadata,
                chart=result.get("chart"),
                visualization_type=result.get("visualization_type", "none"),
            )
        except Exception as exc:
            code = log_exception(exc, scope="openai.non_stream")
            raise HTTPException(status_code=500, detail=f"[{code.value}] Error processing completion: {exc}") from exc

    async def event_generator():
        created = int(time.time())
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        model = request.model
        agent_resolution = await resolve_agent_stream_runtime(
            latest_user_question=latest_user_question,
            k=k,
            lab_name=lab_name,
            allow_clarify=normalize_allow_clarify(request.allow_clarify),
            conversation_context=conversation_context,
            endpoint_key="openai_stream",
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
        should_clarify_response = runtime.should_clarify_response
        scope_class = runtime.scope_class
        use_knowledge_executor = runtime.use_knowledge_executor
        execution_intent = runtime.execution_intent
        stream_signals = (route_plan.planner_parameters or {}).get("query_signals") or {}
        suppress_live_scope = bool(stream_signals.get("is_hypothetical_conditional")) and not bool(
            stream_signals.get("requests_current_measured_data")
        )
        active_question = state.effective_question
        active_lab_name = state.effective_lab
        effective_question = active_question
        effective_lab_name = active_lab_name
        memory_meta = {
            "applied": bool(state.effective_question != state.original_question or state.effective_lab != lab_name),
            "carried_lab_name": state.routing_memory.lab_name if state.effective_lab and not lab_name else None,
            "carried_time_phrase": state.routing_memory.time_phrase if state.effective_question != state.original_question else None,
            "carried_metric": state.routing_memory.metric if state.effective_question != state.original_question else None,
        }
        visualization_type = "none"
        chart = None
        db_context = None
        x_router = {}
        try:
            if should_clarify_response:
                x_router = build_stream_clarify_metadata(
                    route_plan=route_plan,
                    decision=decision,
                    k=k,
                    lab_name=lab_name,
                    resolved_lab_name=active_lab_name,
                )
                x_router["memory_carryover_applied"] = bool(memory_meta.get("applied"))
                x_router["memory_carried_lab_name"] = memory_meta.get("carried_lab_name")
                x_router["memory_carried_time_phrase"] = memory_meta.get("carried_time_phrase")
                x_router["memory_carried_metric"] = memory_meta.get("carried_metric")
            elif use_knowledge_executor:
                knowledge_question = active_question if suppress_live_scope else (
                    build_synthesis_context(
                        tool_results=None,
                        conversation_context=conversation_context,
                        question=active_question,
                    )
                )
                knowledge_lab_name = None if suppress_live_scope else active_lab_name
                knowledge_stats = await fetch_knowledge_stats(
                    knowledge_question,
                    k,
                    knowledge_lab_name,
                )
                x_router = build_stream_knowledge_metadata(
                    route_plan=route_plan,
                    decision=decision,
                    k=k,
                    lab_name=lab_name,
                    resolved_lab_name=knowledge_lab_name,
                    cards_retrieved=int(knowledge_stats.get("cards_retrieved") or 0),
                    knowledge_cards_retrieved=int(
                        knowledge_stats.get("knowledge_cards_retrieved") or 0
                    ),
                )
            else:
                effective_question = active_question
                effective_lab_name = active_lab_name
                db_context = await fetch_db_context(
                    latest_user_question=effective_question,
                    execution_intent=execution_intent,
                    lab_name=effective_lab_name,
                    planner_parameters=route_plan.planner_parameters,
                    prepare_db_query_fn=prepare_db_query,
                )
                visualization_type = db_context.get("visualization_type", "none")
                chart = db_context.get("chart")
                if db_context.get("invariant_violation"):
                    x_router = build_stream_clarify_metadata(
                        route_plan=route_plan,
                        decision=decision,
                        k=k,
                        lab_name=lab_name,
                        resolved_lab_name=db_context.get("resolved_lab_name"),
                        time_window=db_context.get("time_window"),
                        invariant_violation=db_context.get("invariant_violation"),
                    )
                    x_router["execution_intent"] = execution_intent.value
                    x_router["intent_rerouted_to_db"] = execution_intent != decision.intent
                    x_router["db_clarify_text"] = str(db_context.get("fallback_answer") or "")
                    visualization_type = "none"
                    chart = None
                else:
                    x_router = build_stream_db_metadata(
                        route_plan=route_plan,
                        decision=decision,
                        execution_intent=execution_intent,
                        k=k,
                        lab_name=lab_name,
                        db_context=db_context,
                    )
                    x_router["memory_carryover_applied"] = bool(memory_meta.get("applied"))
                    x_router["memory_carried_lab_name"] = memory_meta.get("carried_lab_name")
                    x_router["memory_carried_time_phrase"] = memory_meta.get("carried_time_phrase")
                    x_router["memory_carried_metric"] = memory_meta.get("carried_metric")
        except Exception as exc:
            # Keep streaming functional even if context preparation fails.
            log_exception(exc, scope="openai.stream.prep", extra={"lab_name": lab_name, "k": k})
            visualization_type = "none"
            chart = None
            x_router = {
                **route_plan_metadata(route_plan),
                "execution_intent": execution_intent.value,
                "intent_rerouted_to_db": execution_intent != decision.intent,
                "clarification_required": should_clarify_response,
                "k_requested": k,
                "lab_name": lab_name,
                "sources": [],
            }
        x_router = attach_conversation_metadata(
            x_router,
            conversation_id=normalized_conversation_id,
            conversation_context_applied=context_applied,
            turn_index=None,
        )
        x_router = attach_policy_metadata(x_router, runtime.route_contract)
        x_router = _attach_agent_stream_metadata(x_router, route_plan)
        x_router["agent_step_trace"] = agent_step_trace
        x_router["tools_called"] = tools_called
        x_router["agent_steps"] = int(len(agent_step_trace))
        x_router["agent_finish_reason"] = agent_finish_reason
        try:
            first_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "x_router": x_router,
                "x_visualization_type": visualization_type,
                "x_chart": chart,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant"},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(first_chunk)}\n\n"
            if bool(x_router.get("agent_stream_step_events")):
                for step_payload in agent_step_trace:
                    step_chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "x_router": {"agent_step": step_payload},
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(step_chunk)}\n\n"

            if should_clarify_response:
                clarify_text = build_clarify_prompt(route_plan)
                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": clarify_text},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                turn_index = persist_turn(
                    conversation_id=normalized_conversation_id,
                    question=question,
                    answer=clarify_text,
                )
                x_router["turn_index"] = turn_index
            elif use_knowledge_executor:
                knowledge_question = active_question if suppress_live_scope else (
                    build_synthesis_context(
                        tool_results=None,
                        conversation_context=conversation_context,
                        question=active_question,
                    )
                )
                knowledge_lab_name = None if suppress_live_scope else active_lab_name
                chunk_stream = stream_answer_env_question(
                    user_question=knowledge_question,
                    k=max(1, min(k, 8)),
                    space=knowledge_lab_name,
                    think=request.think,
                )
            else:
                if db_context and db_context.get("invariant_violation"):
                    async def _clarify_chunk_stream():
                        yield str(db_context.get("fallback_answer") or "")
                    chunk_stream = _clarify_chunk_stream()
                else:
                    chunk_stream = stream_db_query(
                        question=effective_question,
                        intent=execution_intent,
                        lab_name=effective_lab_name,
                        planner_hints=route_plan.planner_parameters,
                        query_context=db_context,
                        think=request.think,
                    )

            streamed_answer = ""
            if not should_clarify_response:
                async for chunk_text in chunk_stream:
                    if not chunk_text:
                        continue
                    streamed_answer += chunk_text
                    chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": chunk_text},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"

                turn_index = persist_turn(
                    conversation_id=normalized_conversation_id,
                    question=question,
                    answer=streamed_answer,
                )
                x_router["turn_index"] = turn_index

            final_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as exc:
            payload = stream_error_payload(
                exc,
                scope="openai.stream",
                extra={"lab_name": lab_name, "k": k},
            )
            error_chunk = {"error": {"message": payload["detail"], "type": "api_error", "param": None, "code": payload["code"]}}
            yield f"data: {json.dumps(error_chunk)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )

