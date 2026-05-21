"""Routed query endpoints (sync + stream)."""

from typing import Dict

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

try:
    from http_schemas import QueryRequest, QueryResponse
    from query_routing.query_orchestrator import execute_query, stream_query
    from http_routes.route_helpers import (
        SSE_HEADERS,
        attach_conversation_metadata,
        build_query_inputs,
        persist_turn,
    )
    from runtime_errors import log_exception, stream_error_payload
except ImportError:
    from ..http_schemas import QueryRequest, QueryResponse
    from ..query_routing.query_orchestrator import execute_query, stream_query
    from .route_helpers import (
        SSE_HEADERS,
        attach_conversation_metadata,
        build_query_inputs,
        persist_turn,
    )
    from ..runtime_errors import log_exception, stream_error_payload


router = APIRouter()


def _normalize_k(k) -> int:
    return int(k or 5)


def _normalize_lab(lab_name) -> str | None:
    return (lab_name or "").strip() or None


def _normalize_allow_clarify(flag) -> bool:
    return bool(flag if flag is not None else True)


@router.post("/query", response_model=QueryResponse)
async def query_cards(request: QueryRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    try:
        k = _normalize_k(request.k)
        lab_name = _normalize_lab(request.lab_name)
        latest_user_question, conversation_id, conversation_context, context_applied = build_query_inputs(
            question=question,
            conversation_id=request.conversation_id,
        )
        result = await run_in_threadpool(
            execute_query,
            latest_user_question,
            k,
            lab_name,
            _normalize_allow_clarify(request.allow_clarify),
            "query_sync",
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
        return QueryResponse(
            answer=result["answer"],
            timescale=result["timescale"],
            cards_retrieved=result["cards_retrieved"],
            recent_card=result["recent_card"],
            conversation_id=conversation_id,
            turn_index=turn_index,
            metadata=metadata,
            footnotes=list(result.get("footnotes") or []),
            citation_sources=list(result.get("citation_sources") or []),
        )
    except Exception as exc:
        code = log_exception(exc, scope="query.non_stream")
        raise HTTPException(status_code=500, detail=f"[{code.value}] Error processing query: {exc}") from exc


@router.post("/query/stream")
async def query_cards_stream(request: QueryRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    k = _normalize_k(request.k)
    lab_name = _normalize_lab(request.lab_name)
    latest_user_question, conversation_id, conversation_context, _ = build_query_inputs(
        question=question,
        conversation_id=request.conversation_id,
    )

    async def _generate():
        try:
            async for chunk in stream_query(
                question=latest_user_question,
                k=k,
                lab_name=lab_name,
                endpoint_key="query_stream",
                conversation_context=conversation_context,
            ):
                yield chunk
        except Exception as exc:
            log_exception(exc, scope="query.stream")
            yield stream_error_payload(exc)

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )
