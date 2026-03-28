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
    from query_routing.query_orchestrator import (
        build_clarify_prompt,
        build_non_domain_scope_message,
        execute_query,  # compatibility export for tests/mocks
        get_route_plan,  # compatibility export for tests/mocks
        query_scope_class,  # compatibility export for tests/mocks
        resolve_execution_intent,  # compatibility export for tests/mocks
        should_clarify,  # compatibility export for tests/mocks
        should_use_knowledge_executor,  # compatibility export for tests/mocks
    )
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
    from ..query_routing.query_orchestrator import (
        build_clarify_prompt,
        build_non_domain_scope_message,
        execute_query,  # compatibility export for tests/mocks
        get_route_plan,  # compatibility export for tests/mocks
        query_scope_class,  # compatibility export for tests/mocks
        resolve_execution_intent,  # compatibility export for tests/mocks
        should_clarify,  # compatibility export for tests/mocks
        should_use_knowledge_executor,  # compatibility export for tests/mocks
    )
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
        resolve_stream_runtime,
    )
    from .route_helpers import (
        SSE_HEADERS,
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
    effective_question, normalized_conversation_id, context_applied = build_query_context(
        question=question,
        conversation_id=request.conversation_id,
    )

    if not request.stream:
        try:
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
        should_clarify_response = runtime.should_clarify_response
        scope_class = runtime.scope_class
        use_knowledge_executor = runtime.use_knowledge_executor
        execution_intent = runtime.execution_intent
        visualization_type = "none"
        chart = None
        db_context = None
        x_router = {
            **route_plan_metadata(route_plan),
            "execution_intent": execution_intent.value,
            "intent_rerouted_to_db": execution_intent != decision.intent,
            "clarification_required": should_clarify_response,
            "k_requested": k,
            "lab_name": lab_name,
        }
        x_router = attach_conversation_metadata(
            x_router,
            conversation_id=normalized_conversation_id,
            conversation_context_applied=context_applied,
            turn_index=None,
        )
        try:
            if should_clarify_response:
                x_router["executor"] = "clarify_gate"
                x_router["sources"] = []
                x_router["cards_retrieved"] = 0
            elif scope_class == "non_domain":
                x_router["executor"] = "scope_guardrail"
                x_router["sources"] = []
                x_router["cards_retrieved"] = 0
                x_router["knowledge_cards_retrieved"] = 0
                x_router["execution_intent"] = decision.intent.value
                x_router["intent_rerouted_to_db"] = False
            elif use_knowledge_executor:
                knowledge_stats = await fetch_knowledge_stats(
                    effective_question,
                    k,
                    lab_name,
                    stats_fn=get_knowledge_context_stats,
                )
                x_router["executor"] = "knowledge_qa"
                x_router["execution_intent"] = decision.intent.value
                x_router["intent_rerouted_to_db"] = False
                x_router["sources"] = []
                x_router["cards_retrieved"] = int(knowledge_stats.get("cards_retrieved") or 0)
                x_router["knowledge_cards_retrieved"] = int(
                    knowledge_stats.get("knowledge_cards_retrieved") or 0
                )
            else:
                db_context = await fetch_db_context(
                    effective_question=effective_question,
                    execution_intent=execution_intent,
                    lab_name=lab_name,
                    planner_parameters=route_plan.planner_parameters,
                prepare_db_query_fn=prepare_db_query,
                )
                visualization_type = db_context.get("visualization_type", "none")
                chart = db_context.get("chart")
                x_router["sources"] = db_context.get("sources", [])
        except Exception as exc:
            # Keep streaming functional even if context preparation fails.
            log_exception(exc, scope="openai.stream.prep", extra={"lab_name": lab_name, "k": k})
            visualization_type = "none"
            chart = None
            x_router["sources"] = []
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
            elif scope_class == "non_domain":
                guardrail_text = build_non_domain_scope_message(question)
                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": guardrail_text},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                turn_index = persist_turn(
                    conversation_id=normalized_conversation_id,
                    question=question,
                    answer=guardrail_text,
                )
                x_router["turn_index"] = turn_index
            elif use_knowledge_executor:
                chunk_stream = stream_answer_env_question(
                    user_question=effective_question,
                    k=max(1, min(k, 8)),
                    space=lab_name,
                    think=request.think,
                )
            else:
                chunk_stream = stream_db_query(
                    question=effective_question,
                    intent=execution_intent,
                    lab_name=lab_name,
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

