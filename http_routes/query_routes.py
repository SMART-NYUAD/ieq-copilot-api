"""Routed query endpoints (sync + stream)."""

import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

from http_schemas import QueryRequest, QueryResponse
from query_routing.query_orchestrator import execute_query, stream_query
from http_routes.auth import require_api_key
from http_routes.route_helpers import (
    SSE_HEADERS,
    attach_conversation_metadata,
    build_query_context,
    persist_turn,
)
from storage.conversation_store import ConversationAccessError
from runtime_errors import log_exception, stream_error_payload


router = APIRouter()


def _normalize_k(k) -> int:
    return int(k or 5)


def _normalize_lab(lab_name) -> str | None:
    return (lab_name or "").strip() or None


def _normalize_allow_clarify(flag) -> bool:
    return bool(flag if flag is not None else True)


def _query_context_or_403(question: str, lab_name, conversation_id, caller_id: str, role=None):
    """Build the turn context, mapping a foreign conversation_id to HTTP 403."""
    try:
        return build_query_context(
            question, lab_name, conversation_id, owner=caller_id, role=role
        )
    except ConversationAccessError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


def _sticky_role(ctx) -> str | None:
    """The role to record on the conversation, or None to leave the stored one alone.

    Only an explicit request role is written back. Persisting a resolved default would
    freeze today's default into the row, so a later change to
    ``DEFAULT_STAKEHOLDER_ROLE`` would not reach conversations that never chose a role.
    """
    return ctx.role if ctx.role_source == "request" else None


@router.post("/query", response_model=QueryResponse)
async def query_cards(request: QueryRequest, caller_id: str = Depends(require_api_key)):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    ctx = _query_context_or_403(
        question,
        _normalize_lab(request.lab_name),
        request.conversation_id,
        caller_id,
        role=request.role,
    )
    try:
        k = _normalize_k(request.k)

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
            owner=caller_id,
            role=_sticky_role(ctx),
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
async def query_cards_stream(request: QueryRequest, caller_id: str = Depends(require_api_key)):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    k = _normalize_k(request.k)
    ctx = _query_context_or_403(
        question,
        _normalize_lab(request.lab_name),
        request.conversation_id,
        caller_id,
        role=request.role,
    )

    async def _generate():
        accumulated: list[str] = []
        try:
            async for chunk in stream_query(
                ctx,
                k=k,
                allow_clarify=_normalize_allow_clarify(request.allow_clarify),
                endpoint_key="query_stream",
            ):
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
            # stream_error_payload logs the exception and returns the error+done frames.
            yield stream_error_payload(exc, scope="query.stream")
        finally:
            # Persist whatever was produced — including on client disconnect (GeneratorExit)
            # — so the turn is not lost from conversation context. Skip empties so a failed
            # turn doesn't pollute the history with a blank answer.
            answer = "".join(accumulated).strip()
            if answer:
                try:
                    persist_turn(
                        conversation_id=ctx.conversation_id,
                        question=question,
                        answer=answer,
                        owner=caller_id,
                        role=_sticky_role(ctx),
                    )
                except Exception as exc:
                    log_exception(exc, scope="query.stream.persist")

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )
