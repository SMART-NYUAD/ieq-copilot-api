"""Top-level query orchestration: route → execute → return."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, Optional
from urllib.parse import urlencode

from fastapi.concurrency import run_in_threadpool

from core_settings import download_base_url, download_sensor_alias
from executors.db_query_executor import prepare_db_query, run_db_query, stream_db_tokens
from executors.db_support.query_parsing import extract_time_window, has_explicit_time_hint
from executors.ifc_executor import (
    answer_ifc_question_with_metadata,
    stream_ifc_tokens,
)
from executors.knowledge_executor import (
    answer_env_question_with_metadata,
    stream_knowledge_tokens,
)
from query_routing.intent_classifier import IntentType
from query_routing.llm_router_planner import plan_route, plan_route_async
from query_routing.metadata_builders import derive_ui_contract
from query_routing.router_types import RoutePlan, RouteExecutor
from storage.conversation_context import ConversationContext

_KNOWLEDGE_INTENTS = {IntentType.DEFINITION_EXPLANATION, IntentType.UNKNOWN_FALLBACK}

_VIEWER_CONFIRMATIONS = {
    "splat": "Opening the Gaussian Splat view...",
    "ifc": "Opening the IFC / floor plan view...",
    "pc": "Opening the Point Cloud view...",
    "pano": "Opening the Panorama view...",
}

_HEATMAP_METRIC_LABELS = {
    "temperature": "temperature",
    "humidity": "humidity",
    "voc": "VOC",
    "pm25": "PM2.5",
}

_UNKNOWN_FALLBACK_ANSWER = (
    "I can help with indoor environmental quality, sensor readings, building-model questions, "
    "viewer controls, or the heatmap overlay. Please ask about one of those topics."
)


def _heatmap_confirmation(action: str, metric: Optional[str]) -> str:
    if action == "off":
        return "Turning off the heatmap..."
    label = _HEATMAP_METRIC_LABELS.get(metric or "")
    if label:
        return f"Turning on the {label} heatmap..."
    return "Turning on the heatmap..."


def _choose_executor(route: RoutePlan) -> RouteExecutor:
    if route.intent == IntentType.VIEWER_CONTROL:
        return RouteExecutor.VIEWER_CONTROL
    if route.intent == IntentType.HEATMAP_CONTROL:
        return RouteExecutor.HEATMAP_CONTROL
    if route.intent == IntentType.DOWNLOAD_DATA:
        return RouteExecutor.DOWNLOAD_DATA
    if route.intent == IntentType.IFC_MODEL_QA:
        return RouteExecutor.IFC_QA
    if route.intent in _KNOWLEDGE_INTENTS:
        return RouteExecutor.KNOWLEDGE_QA
    return RouteExecutor.DB_QUERY


def _build_planner_hints(route: RoutePlan, carried_time_phrase: Optional[str] = None) -> Dict[str, Any]:
    hints: Dict[str, Any] = {
        "metrics_priority": list(route.metrics),
        "needs_cards": route.intent in _KNOWLEDGE_INTENTS,
        "card_topics": ["definitions", "metric_explanations"] if route.intent in _KNOWLEDGE_INTENTS else ["metric_explanations"],
        "max_cards": 2,
        "second_lab_name": route.second_lab_name,
    }
    if carried_time_phrase:
        hints["carried_time_phrase"] = carried_time_phrase
    return hints


def _fetch_live_sensor_data(
    question: str, lab_name: Optional[str], route: RoutePlan
) -> Optional[Dict[str, Any]]:
    """Pre-fetch current sensor readings to ground knowledge-path answers with real data.
    Returns the DB payload dict when rows exist, None otherwise."""
    try:
        db_ctx = prepare_db_query(
            question=question,
            intent=IntentType.CURRENT_STATUS_DB,
            lab_name=lab_name,
            planner_hints={
                "metrics_priority": list(route.metrics),
                "needs_cards": False,
                "card_topics": [],
                "max_cards": 0,
                "second_lab_name": None,
            },
        )
        if db_ctx.get("rows"):
            return db_ctx.get("payload")
    except Exception:
        pass
    return None


def _execute_knowledge(
    question: str,
    k: int,
    lab_name: Optional[str],
    route: RoutePlan,
) -> Dict[str, Any]:
    live_sensor_data = _fetch_live_sensor_data(question, lab_name, route)
    result = answer_env_question_with_metadata(
        user_question=question,
        k=max(1, min(k, 8)),
        space=lab_name,
        live_sensor_data=live_sensor_data,
    )
    return {
        "answer": str(result.get("answer") or ""),
        "footnotes": list(result.get("footnotes") or []),
        "citation_sources": list(result.get("indexed_sources") or []),
        "timescale": "knowledge",
        "cards_retrieved": int(result.get("cards_retrieved") or 0),
        "recent_card": False,
        "metadata": {
            "executor": "knowledge_qa",
            "intent": route.intent.value,
            "lab_name": lab_name,
            "llm_used": True,
            "route_confidence": route.confidence,
            "planner_model": route.model,
            "fallback_used": route.fallback_used,
            "ui": {"mode": "conversational", "panel": "overview", "metrics": [], "transition": "fade"},
        },
        "data": None,
    }


def _execute_unknown_fallback(route: RoutePlan) -> Dict[str, Any]:
    return {
        "answer": _UNKNOWN_FALLBACK_ANSWER,
        "footnotes": [],
        "citation_sources": [],
        "timescale": "guardrail",
        "cards_retrieved": 0,
        "recent_card": False,
        "metadata": {
            "executor": "guardrail",
            "intent": route.intent.value,
            "lab_name": None,
            "llm_used": False,
            "route_confidence": route.confidence,
            "planner_model": route.model,
            "fallback_used": route.fallback_used,
            "ui": {"mode": "conversational", "panel": "overview", "metrics": [], "transition": "fade"},
        },
        "data": None,
    }


def _execute_db(
    question: str,
    k: int,
    lab_name: Optional[str],
    route: RoutePlan,
    llm_history: str = "",
    carried_time_phrase: Optional[str] = None,
) -> Dict[str, Any]:
    planner_hints = _build_planner_hints(route, carried_time_phrase=carried_time_phrase)
    db_result = run_db_query(
        question=question,
        intent=route.intent,
        lab_name=lab_name,
        planner_hints=planner_hints,
        conversation_context=llm_history,
    )
    metrics = list(db_result.get("metrics_used") or planner_hints.get("metrics_priority") or [])
    ui = derive_ui_contract(
        execution_intent=route.intent,
        metrics=metrics,
        has_floor_comparison=False,
        clarification_required="clarify" in str(db_result.get("timescale", "")),
        use_knowledge_executor=False,
    )
    return {
        "answer": str(db_result.get("answer") or ""),
        "footnotes": list(db_result.get("footnotes") or []),
        "citation_sources": list(db_result.get("indexed_sources") or []),
        "timescale": db_result.get("timescale", "1hour"),
        "cards_retrieved": int(db_result.get("cards_retrieved") or 0),
        "recent_card": False,
        "metadata": {
            "executor": "db_query",
            "intent": route.intent.value,
            "lab_name": lab_name,
            "resolved_lab_name": db_result.get("resolved_lab_name"),
            "time_window": db_result.get("time_window"),
            "llm_used": db_result.get("llm_used", False),
            "route_confidence": route.confidence,
            "planner_model": route.model,
            "fallback_used": route.fallback_used,
            "ui": ui,
        },
        "data": db_result.get("data"),
    }


async def _build_stream_meta(
    route: RoutePlan,
    effective_lab: Optional[str],
    use_knowledge_executor: bool,
) -> Dict[str, Any]:
    metrics = list(route.metrics)
    ui = derive_ui_contract(
        execution_intent=route.intent,
        metrics=metrics,
        has_floor_comparison=False,
        clarification_required=False,
        use_knowledge_executor=use_knowledge_executor,
    )
    executor = "knowledge_qa" if use_knowledge_executor else "db_query"
    timescale = "knowledge" if use_knowledge_executor else "pending"
    return {
        "executor": executor,
        "intent": route.intent.value,
        "lab_name": effective_lab,
        "llm_used": True,
        "route_confidence": route.confidence,
        "planner_model": route.model,
        "fallback_used": route.fallback_used,
        "ui": ui,
        "timescale": timescale,
        "cards_retrieved": 0,
        "recent_card": False,
        "visualization_type": "none",
        "chart": None,
        "citation_sources": [],
        "footnotes": [],
    }


def _execute_ifc(question: str, route: RoutePlan) -> Dict[str, Any]:
    result = answer_ifc_question_with_metadata(user_question=question)
    # The IFC model has no per-claim numbered citations, so citation_sources stays
    # empty; provenance (the model file) is surfaced in metadata instead.
    return {
        "answer": str(result.get("answer") or ""),
        "footnotes": [],
        "citation_sources": [],
        "timescale": "model",
        "cards_retrieved": 0,
        "recent_card": False,
        "metadata": {
            "executor": "ifc_qa",
            "intent": route.intent.value,
            "lab_name": None,
            "llm_used": bool(result.get("llm_used", False)),
            "model_available": bool(result.get("model_available", True)),
            "model_source": list(result.get("indexed_sources") or []),
            "route_confidence": route.confidence,
            "planner_model": route.model,
            "fallback_used": route.fallback_used,
            "ui": {"mode": "conversational", "panel": "ifc", "metrics": [], "transition": "fade"},
        },
        "data": None,
    }


def _execute_viewer_control(route: RoutePlan) -> Dict[str, Any]:
    viewer_type = route.viewer_type or "splat"
    confirmation = _VIEWER_CONFIRMATIONS.get(viewer_type, f"Opening the {viewer_type} view...")
    return {
        "answer": confirmation,
        "footnotes": [],
        "citation_sources": [],
        "timescale": "instant",
        "cards_retrieved": 0,
        "recent_card": False,
        "metadata": {
            "executor": "viewer_control",
            "intent": route.intent.value,
            "lab_name": None,
            "llm_used": False,
            "route_confidence": route.confidence,
            "planner_model": route.model,
            "fallback_used": route.fallback_used,
            "ui": {"viewer_type": viewer_type},
        },
        "data": None,
    }


def _execute_heatmap_control(route: RoutePlan) -> Dict[str, Any]:
    action = route.heatmap_action or "on"
    metric = route.heatmap_metric
    confirmation = _heatmap_confirmation(action, metric)
    return {
        "answer": confirmation,
        "footnotes": [],
        "citation_sources": [],
        "timescale": "instant",
        "cards_retrieved": 0,
        "recent_card": False,
        "metadata": {
            "executor": "heatmap_control",
            "intent": route.intent.value,
            "lab_name": None,
            "llm_used": False,
            "route_confidence": route.confidence,
            "planner_model": route.model,
            "fallback_used": route.fallback_used,
            "ui": {"heatmap_action": action, "heatmap_metric": metric},
        },
        "data": None,
    }


# No explicit window in a download request → export a wide range (one year) rather than
# the 24h DB default, since a user asking to "download the data" usually wants the full history.
_DOWNLOAD_DEFAULT_HOURS = 24 * 365


def _build_download(route: RoutePlan, question: str) -> Dict[str, Any]:
    """Resolve the download request into a ready-to-use sensor-readings URL plus display fields.

    The time window is resolved server-side (mirroring the DB path) so the frontend never has to
    reconstruct date ranges — it just opens the URL or renders a button pointing at it.
    """
    default_hours = _DOWNLOAD_DEFAULT_HOURS if not has_explicit_time_hint(question) else 24
    start, end, window_label = extract_time_window(question, default_hours=default_hours)
    fmt = route.download_format or "csv"
    dtype = route.download_type or "aggregated"
    params = {
        "sensor_alias": download_sensor_alias(),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "type": dtype,
        "format": fmt,
    }
    url = f"{download_base_url()}?{urlencode(params)}"
    return {
        "url": url,
        "format": fmt,
        "type": dtype,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "window_label": window_label,
    }


def _execute_download_control(route: RoutePlan, question: str) -> Dict[str, Any]:
    dl = _build_download(route, question)
    confirmation = (
        f"Here's your {dl['format'].upper()} download for {dl['window_label']} — "
        "use the button to save the sensor readings."
    )
    return {
        "answer": confirmation,
        "footnotes": [],
        "citation_sources": [],
        "timescale": "instant",
        "cards_retrieved": 0,
        "recent_card": False,
        "metadata": {
            "executor": "download_data",
            "intent": route.intent.value,
            "lab_name": None,
            "llm_used": False,
            "route_confidence": route.confidence,
            "planner_model": route.model,
            "fallback_used": route.fallback_used,
            "ui": {
                "download_url": dl["url"],
                "download_format": dl["format"],
                "download_type": dl["type"],
                "download_start": dl["start"],
                "download_end": dl["end"],
            },
        },
        "data": None,
    }


def execute_query(ctx: ConversationContext, k: int, allow_clarify: bool = True, endpoint_key: str = "query_sync") -> Dict[str, Any]:
    """Execute a query given a fully-resolved ConversationContext."""
    route = plan_route(ctx.effective_question, ctx.effective_lab, ctx.routing_snippet)
    executor = _choose_executor(route)

    if executor == RouteExecutor.VIEWER_CONTROL:
        return _execute_viewer_control(route)
    if executor == RouteExecutor.HEATMAP_CONTROL:
        return _execute_heatmap_control(route)
    if executor == RouteExecutor.DOWNLOAD_DATA:
        return _execute_download_control(route, ctx.effective_question)
    if executor == RouteExecutor.IFC_QA:
        return _execute_ifc(ctx.effective_question, route)
    if route.intent == IntentType.UNKNOWN_FALLBACK:
        return _execute_unknown_fallback(route)
    if executor == RouteExecutor.KNOWLEDGE_QA:
        return _execute_knowledge(ctx.effective_question, k, ctx.effective_lab, route)
    return _execute_db(ctx.effective_question, k, ctx.effective_lab, route, ctx.llm_history,
                       carried_time_phrase=ctx.carried_time_phrase)


def _status_event(stage: str, message: str) -> str:
    return f"data: {json.dumps({'event': 'status', 'stage': stage, 'message': message})}\n\n"


async def stream_query(ctx: ConversationContext, k: int, endpoint_key: str = "query_stream") -> AsyncIterator[str]:
    """Stream a query given a fully-resolved ConversationContext."""
    yield _status_event("routing", "Classifying question…")
    route = await plan_route_async(ctx.effective_question, ctx.effective_lab, ctx.routing_snippet)
    executor = _choose_executor(route)

    if executor == RouteExecutor.VIEWER_CONTROL:
        viewer_type = route.viewer_type or "splat"
        confirmation = _VIEWER_CONFIRMATIONS.get(viewer_type, f"Opening the {viewer_type} view...")
        yield f"data: {json.dumps({'event': 'meta', 'executor': 'viewer_control', 'intent': route.intent.value, 'lab_name': None, 'llm_used': False, 'route_confidence': route.confidence, 'planner_model': route.model, 'fallback_used': route.fallback_used, 'ui': {'viewer_type': viewer_type}, 'timescale': 'instant', 'cards_retrieved': 0, 'recent_card': False, 'visualization_type': 'none', 'chart': None, 'citation_sources': [], 'footnotes': []})}\n\n"
        yield f"data: {json.dumps({'event': 'token', 'text': confirmation})}\n\n"
        yield f"data: {json.dumps({'event': 'done'})}\n\n"
        return

    if executor == RouteExecutor.HEATMAP_CONTROL:
        action = route.heatmap_action or "on"
        metric = route.heatmap_metric
        confirmation = _heatmap_confirmation(action, metric)
        yield f"data: {json.dumps({'event': 'meta', 'executor': 'heatmap_control', 'intent': route.intent.value, 'lab_name': None, 'llm_used': False, 'route_confidence': route.confidence, 'planner_model': route.model, 'fallback_used': route.fallback_used, 'ui': {'heatmap_action': action, 'heatmap_metric': metric}, 'timescale': 'instant', 'cards_retrieved': 0, 'recent_card': False, 'visualization_type': 'none', 'chart': None, 'citation_sources': [], 'footnotes': []})}\n\n"
        yield f"data: {json.dumps({'event': 'token', 'text': confirmation})}\n\n"
        yield f"data: {json.dumps({'event': 'done'})}\n\n"
        return

    if executor == RouteExecutor.DOWNLOAD_DATA:
        result = _execute_download_control(route, ctx.effective_question)
        meta = {
            "event": "meta",
            "executor": "download_data",
            "intent": route.intent.value,
            "lab_name": None,
            "llm_used": False,
            "route_confidence": route.confidence,
            "planner_model": route.model,
            "fallback_used": route.fallback_used,
            "ui": result["metadata"]["ui"],
            "timescale": "instant",
            "cards_retrieved": 0,
            "recent_card": False,
            "visualization_type": "none",
            "chart": None,
            "citation_sources": [],
            "footnotes": [],
        }
        yield f"data: {json.dumps(meta)}\n\n"
        yield f"data: {json.dumps({'event': 'token', 'text': result['answer']})}\n\n"
        yield f"data: {json.dumps({'event': 'done'})}\n\n"
        return

    if executor == RouteExecutor.IFC_QA:
        ifc_meta = {
            "executor": "ifc_qa",
            "intent": route.intent.value,
            "lab_name": None,
            "llm_used": True,
            "route_confidence": route.confidence,
            "planner_model": route.model,
            "fallback_used": route.fallback_used,
            "ui": {"mode": "conversational", "panel": "ifc", "metrics": [], "transition": "fade"},
            "timescale": "model",
            "cards_retrieved": 0,
            "recent_card": False,
            "visualization_type": "none",
            "chart": None,
            "citation_sources": [],
            "footnotes": [],
        }
        yield f"data: {json.dumps({'event': 'meta', **ifc_meta})}\n\n"
        yield _status_event("reading_model", "Reading building model…")
        async for chunk in stream_ifc_tokens(user_question=ctx.effective_question):
            yield chunk
        return

    if route.intent == IntentType.UNKNOWN_FALLBACK:
        meta = {
            "event": "meta",
            "executor": "guardrail",
            "intent": route.intent.value,
            "lab_name": None,
            "llm_used": False,
            "route_confidence": route.confidence,
            "planner_model": route.model,
            "fallback_used": route.fallback_used,
            "ui": {"mode": "conversational", "panel": "overview", "metrics": [], "transition": "fade"},
            "timescale": "guardrail",
            "cards_retrieved": 0,
            "recent_card": False,
            "visualization_type": "none",
            "chart": None,
            "citation_sources": [],
            "footnotes": [],
        }
        yield f"data: {json.dumps(meta)}\n\n"
        yield f"data: {json.dumps({'event': 'token', 'text': _UNKNOWN_FALLBACK_ANSWER})}\n\n"
        yield f"data: {json.dumps({'event': 'done'})}\n\n"
        return

    use_knowledge_executor = executor == RouteExecutor.KNOWLEDGE_QA

    meta = await _build_stream_meta(route, ctx.effective_lab, use_knowledge_executor)
    yield f"data: {json.dumps({'event': 'meta', **meta})}\n\n"

    if use_knowledge_executor:
        yield _status_event("searching_knowledge", "Searching knowledge base…")
        live_sensor_data = await run_in_threadpool(
            _fetch_live_sensor_data, ctx.effective_question, ctx.effective_lab, route
        )
        async for chunk in stream_knowledge_tokens(
            user_question=ctx.effective_question,
            k=max(1, min(k, 8)),
            space=ctx.effective_lab,
            live_sensor_data=live_sensor_data,
        ):
            yield chunk
        return

    planner_hints = _build_planner_hints(route, carried_time_phrase=ctx.carried_time_phrase)

    yield _status_event("querying_db", "Fetching sensor data…")
    query_context = await run_in_threadpool(
        prepare_db_query,
        ctx.effective_question,
        route.intent,
        ctx.effective_lab,
        planner_hints,
    )

    time_window = query_context.get("time_window")
    if time_window:
        yield f"data: {json.dumps({'event': 'meta_update', 'time_window': time_window, 'resolved_lab_name': query_context.get('resolved_lab_name'), 'metrics_used': list(query_context.get('metrics_used') or [])})}\n\n"

    yield _status_event("building_response", "Building response…")
    async for chunk in stream_db_tokens(
        question=ctx.effective_question,
        intent=route.intent,
        lab_name=ctx.effective_lab,
        planner_hints=planner_hints,
        query_context=query_context,
        conversation_context=ctx.llm_history,
    ):
        yield chunk


# Legacy compatibility shims.
def get_route_plan(question: str, lab_name: Optional[str] = None) -> RoutePlan:
    return plan_route(question, lab_name)


def resolve_execution_intent(intent: IntentType) -> IntentType:
    """Return a DB-executable intent (maps semantic intents to current_status_db)."""
    if intent in _KNOWLEDGE_INTENTS:
        return IntentType.CURRENT_STATUS_DB
    return intent
