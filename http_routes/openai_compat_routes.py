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
    from evidence.citation_processor import (
        build_numbered_sources_block,
        extract_citation_indices_from_answer,
    )
    from query_routing.query_orchestrator import (
        build_clarify_prompt,
        execute_query,  # compatibility export for tests/mocks
        get_route_plan,  # compatibility export for tests/mocks
        query_scope_class,  # compatibility export for tests/mocks
        resolve_execution_intent,  # compatibility export for tests/mocks
        should_clarify,  # compatibility export for tests/mocks
        should_use_knowledge_executor,  # compatibility export for tests/mocks
    )
    from executors.db_query_executor import prepare_db_query, stream_db_query
    from executors.env_query_langchain import (
        get_guideline_records_for_question,
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
    )
    from query_routing.metadata_builders import (
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
    from http_routes.stream_shared import (
        attach_agent_stream_metadata as _shared_attach_agent_stream_metadata,
        prepare_stream_execution_context,
    )
    from runtime_errors import log_exception, stream_error_payload
except ImportError:
    from ..evidence.citation_processor import (
        build_numbered_sources_block,
        extract_citation_indices_from_answer,
    )
    from ..query_routing.query_orchestrator import (
        build_clarify_prompt,
        execute_query,  # compatibility export for tests/mocks
        get_route_plan,  # compatibility export for tests/mocks
        query_scope_class,  # compatibility export for tests/mocks
        resolve_execution_intent,  # compatibility export for tests/mocks
        should_clarify,  # compatibility export for tests/mocks
        should_use_knowledge_executor,  # compatibility export for tests/mocks
    )
    from ..executors.db_query_executor import prepare_db_query, stream_db_query
    from ..executors.env_query_langchain import (
        get_guideline_records_for_question,
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
    )
    from ..query_routing.metadata_builders import (
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
    from .stream_shared import (
        attach_agent_stream_metadata as _shared_attach_agent_stream_metadata,
        prepare_stream_execution_context,
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
    return _shared_attach_agent_stream_metadata(meta, route_plan, include_expected_observation=True)


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
    footnotes: Optional[List[Dict[str, Any]]] = None,
    citation_sources: Optional[List[Dict[str, Any]]] = None,
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
        "x_footnotes": list(footnotes or []),
        "x_citation_sources": list(citation_sources or []),
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
                footnotes=list(result.get("footnotes") or []),
                citation_sources=list(result.get("citation_sources") or result.get("indexed_sources") or []),
            )
        except Exception as exc:
            code = log_exception(exc, scope="openai.non_stream")
            raise HTTPException(status_code=500, detail=f"[{code.value}] Error processing completion: {exc}") from exc

    async def event_generator():
        created = int(time.time())
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        model = request.model
        stream_ctx = await prepare_stream_execution_context(
            latest_user_question=latest_user_question,
            k=k,
            lab_name=lab_name,
            allow_clarify=normalize_allow_clarify(request.allow_clarify),
            conversation_context=conversation_context,
            endpoint_key="openai_stream",
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
        should_clarify_response = stream_ctx.mode == "clarify"
        use_knowledge_executor = stream_ctx.mode == "knowledge"
        execution_intent = stream_ctx.execution_intent
        agent_step_trace = list(stream_ctx.agent_step_trace or [])
        tools_called = list(stream_ctx.tools_called or [])
        agent_finish_reason = stream_ctx.agent_finish_reason
        visualization_type = "none"
        chart = None
        db_context = stream_ctx.db_context
        x_router = {}
        citation_sources: List[Dict[str, Any]] = []
        knowledge_guideline_records: List[Dict[str, Any]] = []
        try:
            if should_clarify_response:
                x_router = dict(stream_ctx.meta or {})
            elif use_knowledge_executor:
                x_router = dict(stream_ctx.meta or {})
                knowledge_guideline_records = await run_in_threadpool(
                    get_guideline_records_for_question,
                    str(stream_ctx.effective_question or latest_user_question),
                    3,
                )
                _, citation_sources = build_numbered_sources_block(knowledge_guideline_records)
            else:
                db_context = stream_ctx.db_context
                visualization_type = db_context.get("visualization_type", "none")
                chart = db_context.get("chart")
                citation_sources = list((db_context or {}).get("indexed_sources") or [])
                if not citation_sources:
                    _, citation_sources = build_numbered_sources_block(
                        list((db_context or {}).get("guideline_records") or [])
                    )
                    db_context["indexed_sources"] = citation_sources
                if stream_ctx.mode == "db_clarify":
                    x_router = dict(stream_ctx.meta or {})
                    x_router["db_clarify_text"] = str(db_context.get("fallback_answer") or "")
                    visualization_type = "none"
                    chart = None
                else:
                    x_router = dict(stream_ctx.meta or {})
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
                "x_citation_sources": citation_sources,
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
                knowledge_indexed_sources: List[Dict[str, Any]] = []
                chunk_stream = stream_answer_env_question(
                    user_question=str(stream_ctx.knowledge_question or ""),
                    k=max(1, min(k, 8)),
                    space=stream_ctx.knowledge_lab_name,
                    think=request.think,
                    guideline_records=knowledge_guideline_records,
                    indexed_sources_out=knowledge_indexed_sources,
                )
            else:
                if stream_ctx.mode == "db_clarify":
                    async def _clarify_chunk_stream():
                        yield str(db_context.get("fallback_answer") or "")
                    chunk_stream = _clarify_chunk_stream()
                else:
                    chunk_stream = stream_db_query(
                        question=str(stream_ctx.effective_question or ""),
                        intent=execution_intent,
                        lab_name=stream_ctx.effective_lab_name,
                        planner_hints=route_plan.planner_parameters,
                        query_context=db_context,
                        think=request.think,
                    )

            streamed_answer = ""
            footnotes: List[Dict[str, Any]] = []
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

                if use_knowledge_executor:
                    footnotes = extract_citation_indices_from_answer(
                        answer_text=streamed_answer,
                        indexed_sources=knowledge_indexed_sources or citation_sources,
                    )
                else:
                    footnotes = extract_citation_indices_from_answer(
                        answer_text=streamed_answer,
                        indexed_sources=citation_sources,
                    )
                if footnotes:
                    footnote_chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "x_footnotes": footnotes,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(footnote_chunk)}\n\n"

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

