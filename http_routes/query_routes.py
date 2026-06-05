"""Routed query endpoints (sync + stream)."""

import json
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
        build_query_context,
        persist_turn,
    )
    from runtime_errors import log_exception, stream_error_payload
except ImportError:
    from ..http_schemas import QueryRequest, QueryResponse
    from ..query_routing.query_orchestrator import execute_query, stream_query
    from .route_helpers import (
        SSE_HEADERS,
        attach_conversation_metadata,
        build_query_context,
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
        ctx = build_query_context(question, _normalize_lab(request.lab_name), request.conversation_id)

        result = await run_in_threadpool(
            execute_query,
            ctx,
            k,
            _normalize_allow_clarify(request.allow_clarify),
            "query_sync",
        )
        turn_index = persist_turn(
            conversation_id=ctx.conversation_id,
            question=question,
            answer=str(result.get("answer") or ""),
        )
        metadata = attach_conversation_metadata(
            dict(result.get("metadata") or {}),
            conversation_id=ctx.conversation_id,
            conversation_context_applied=bool(ctx.raw_block),
            turn_index=turn_index,
        )
        return QueryResponse(
            answer=result["answer"],
            timescale=result["timescale"],
            cards_retrieved=result["cards_retrieved"],
            recent_card=result["recent_card"],
            conversation_id=ctx.conversation_id,
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
    ctx = build_query_context(question, _normalize_lab(request.lab_name), request.conversation_id)

    async def _generate():
        accumulated: list[str] = []
        try:
            async for chunk in stream_query(ctx, k=k, endpoint_key="query_stream"):
                try:
                    raw = chunk.removeprefix("data: ").strip()
                    if raw:
                        evt = json.loads(raw)
                        if evt.get("event") == "token":
                            accumulated.append(str(evt.get("text") or ""))
                except Exception:
                    pass
                yield chunk
        except Exception as exc:
            log_exception(exc, scope="query.stream")
            yield stream_error_payload(exc)
            return

        persist_turn(
            conversation_id=ctx.conversation_id,
            question=question,
            answer="".join(accumulated),
        )

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )
