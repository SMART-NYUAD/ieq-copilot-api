"""Top-level query orchestration: route → plan one branch → render it.

The intent-to-branch decision happens exactly once, in :func:`plan_branch`. Both
response shapes are then produced by rendering the same :class:`Branch`:

    execute_query  → render_sync   → one JSON body
    stream_query   → render_stream → a sequence of SSE frames

Keeping a single ladder is deliberate. When the sync and stream paths each carried
their own copy, they drifted — the stream silently stopped reporting citations while
the sync response kept returning them. A new executor is now added by writing one
branch factory; there is no second place that can forget it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Dict, Optional, Tuple

from fastapi.concurrency import run_in_threadpool

from core_settings import download_default_interval, slugify_space
from executors.db_query_executor import prepare_db_query, run_db_query, stream_db_tokens
from executors.db_support.query_parsing import extract_time_window
from executors.ifc_executor import (
    answer_ifc_question_with_metadata,
    stream_ifc_tokens,
)
from executors.knowledge_executor import (
    answer_env_question_with_metadata,
    stream_knowledge_tokens,
)
from executors.sensor_inspection_executor import (
    answer_sensor_question_with_metadata,
    stream_sensor_tokens,
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

_CONVERSATIONAL_UI = {"mode": "conversational", "panel": "overview", "metrics": [], "transition": "fade"}


# ---------------------------------------------------------------------------
# Branch model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Branch:
    """One resolved execution branch — what to run, and how to describe it.

    A branch is either *instant* (``answer`` is set: a fixed confirmation or question,
    no model call) or *generated* (``run_sync`` / ``open_stream`` produce the answer).
    Any blocking pre-work goes in ``prepare`` so the stream renderer can move it off the
    event loop and both renderers share its result.
    """

    name: str                                   # metadata.executor
    route: RoutePlan
    ui: Dict[str, Any]
    timescale: str
    lab_name: Optional[str] = None
    llm_used: bool = False

    # Instant branch.
    answer: Optional[str] = None

    # Generated branch.
    prepare: Optional[Callable[[], Any]] = None
    run_sync: Optional[Callable[[Any], Dict[str, Any]]] = None
    open_stream: Optional[Callable[[Any], AsyncIterator[str]]] = None
    status_before_prepare: Optional[Tuple[str, str]] = None
    status_before_render: Optional[Tuple[str, str]] = None
    # Stream-only: reconcile the placeholder meta once ``prepare`` has run.
    stream_meta_update: Optional[Callable[[Any], Dict[str, Any]]] = None

    extra_meta: Dict[str, Any] = field(default_factory=dict)


def _core_result(
    answer: str,
    *,
    footnotes: Optional[list] = None,
    citation_sources: Optional[list] = None,
    data: Any = None,
    cards_retrieved: int = 0,
    llm_used: bool = False,
    timescale: Optional[str] = None,
    ui: Optional[Dict[str, Any]] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Canonical shape returned by every ``run_sync``.

    ``timescale`` and ``ui`` override the branch defaults when execution resolved
    something better than the pre-execution guess (the DB path narrows both).
    """
    return {
        "answer": str(answer or ""),
        "footnotes": list(footnotes or []),
        "citation_sources": list(citation_sources or []),
        "data": data,
        "cards_retrieved": int(cards_retrieved or 0),
        "llm_used": bool(llm_used),
        "timescale": timescale,
        "ui": ui,
        "meta": dict(meta or {}),
    }


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
    if route.intent == IntentType.SENSOR_INSPECTION:
        return RouteExecutor.SENSOR_INSPECTION
    if route.intent in _KNOWLEDGE_INTENTS:
        return RouteExecutor.KNOWLEDGE_QA
    return RouteExecutor.DB_QUERY


def _build_planner_hints(
    route: RoutePlan,
    carried_time_phrase: Optional[str] = None,
    carried_metric: Optional[str] = None,
) -> Dict[str, Any]:
    # Regex carry-over (carried_metric / carried_time_phrase) is EMERGENCY-ONLY: it
    # applies only when the LLM router was unreachable (``fallback_used``). When the
    # router ran, ``RoutePlan.resolved_question`` already folds every prior-turn
    # reference into the question text — that is the canonical, LLM-driven carry-over.
    # Injecting the regex-guessed prior window on top of it double-resolves and can
    # override the day the user actually asked for: a bare day-of-month like
    # "what about on the 11" is invisible to the time regex, so the stale prior date
    # ("june 9") would win and answer the wrong day. See feat/llm-context-resolution.
    apply_regex_carryover = route.fallback_used
    metrics_priority = list(route.metrics)
    # Carry the prior turn's metric only when the current question named none
    # (the LLM/regex produced no metric for this turn). ``carried_metric`` is
    # already gated upstream so it is only set when the question omits a metric,
    # but guard here too so an explicit current metric always wins.
    if apply_regex_carryover and carried_metric and not metrics_priority:
        metrics_priority = [carried_metric]
    hints: Dict[str, Any] = {
        "metrics_priority": metrics_priority,
        "needs_cards": route.intent in _KNOWLEDGE_INTENTS,
        "card_topics": ["definitions", "metric_explanations"] if route.intent in _KNOWLEDGE_INTENTS else ["metric_explanations"],
        "max_cards": 2,
        "second_lab_name": route.second_lab_name,
        # LLM-driven root-cause signal: when set to "diagnostic" the DB executor
        # decomposes the named index into all contributing sub-scores/metrics
        # rather than returning the single named value.
        "analysis_mode": route.analysis_mode,
        # LLM-chosen metric family (see metric_planning). None => the DB executor infers
        # the scope from question text.
        "metric_scope": route.metric_scope,
    }
    if apply_regex_carryover and carried_time_phrase:
        hints["carried_time_phrase"] = carried_time_phrase
    return hints


# ---------------------------------------------------------------------------
# Branch factories — one per executor
# ---------------------------------------------------------------------------

def _clarify_branch(route: RoutePlan) -> Branch:
    """Ask the router's clarifying question instead of guessing.

    The decision is the router's (``RoutePlan.needs_clarification``); this branch only
    renders it, so the "when to ask" logic lives with the rest of the language understanding.
    """
    return Branch(
        name="clarify_gate",
        route=route,
        ui=derive_ui_contract(
            execution_intent=route.intent,
            metrics=list(route.metrics),
            has_floor_comparison=False,
            clarification_required=True,
            use_knowledge_executor=False,
        ),
        timescale="clarify",
        answer=str(route.clarification_question or ""),
    )


def _viewer_branch(route: RoutePlan) -> Branch:
    viewer_type = route.viewer_type or "splat"
    return Branch(
        name="viewer_control",
        route=route,
        ui={"viewer_type": viewer_type},
        timescale="instant",
        answer=_VIEWER_CONFIRMATIONS.get(viewer_type, f"Opening the {viewer_type} view..."),
    )


def _heatmap_branch(route: RoutePlan) -> Branch:
    action = route.heatmap_action or "on"
    metric = route.heatmap_metric
    return Branch(
        name="heatmap_control",
        route=route,
        ui={"heatmap_action": action, "heatmap_metric": metric},
        timescale="instant",
        answer=_heatmap_confirmation(action, metric),
    )


def _unknown_branch(route: RoutePlan) -> Branch:
    return Branch(
        name="guardrail",
        route=route,
        ui=dict(_CONVERSATIONAL_UI),
        timescale="guardrail",
        answer=_UNKNOWN_FALLBACK_ANSWER,
    )


# A download request with no explicit window defaults to the last 24 hours.
_DOWNLOAD_DEFAULT_HOURS = 24

# Canonical metric → metric_type path segment for /spaces/{slug}/metrics/{metric_type}/...
# The endpoint accepts: co2, humidity, light, noise, pm25, temperature, voc.
_DOWNLOAD_METRIC_TYPES: Dict[str, str] = {
    "temperature": "temperature",
    "humidity": "humidity",
    "co2": "co2",
    "voc": "voc",
    "pm25": "pm25",
}

# Human-friendly metric names for the "which metric?" follow-up question.
_DOWNLOAD_METRIC_LABELS = "temperature, humidity, CO₂, VOC, or PM2.5"


def _to_download_interval(interval: str) -> str:
    """Normalize the canonical interval suffix (m/h/d) to the form the
    ``download-agg-summary`` endpoint accepts.

    The router and inference produce compact suffixes (``1h``, ``15m``, ``1d``), but the
    download endpoint expects hours spelled as ``hr`` — it rejects ``1h``. Minutes and days
    pass through unchanged. Idempotent: an already-normalized value (``1hr``) is returned as-is.
    """
    m = re.fullmatch(r"\s*(\d+)\s*(m|h|d)\s*", (interval or "").lower())
    if not m:
        return interval
    num, unit = m.group(1), m.group(2)
    return f"{num}{'hr' if unit == 'h' else unit}"


def _build_download(route: RoutePlan, question: str) -> Dict[str, Any]:
    """Resolve the download request into the parameters the frontend needs for the
    ``/spaces/{slug}/metrics/{metric_type}/download-agg-summary`` endpoint.

    We hand the frontend the discrete parameters (not a pre-built URL) so it can call the
    endpoint itself. The time window is resolved server-side (mirroring the DB path, defaulting
    to the last 24 hours) so the frontend never reconstructs date ranges.
    """
    start, end, window_label = extract_time_window(question, default_hours=_DOWNLOAD_DEFAULT_HOURS)
    fmt = route.download_format or "csv"
    metric_type = _DOWNLOAD_METRIC_TYPES.get(route.download_metric or "", route.download_metric or "")
    interval = _to_download_interval(route.download_interval or download_default_interval())
    return {
        "slug": slugify_space(route.lab_name),
        "metric_type": metric_type,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "interval": interval,
        "format": fmt,
        "window_label": window_label,
    }


def _download_branch(route: RoutePlan, question: str) -> Branch:
    # A metric is required to build a download. When it is missing, ask a follow-up question
    # instead of handing back parameters — the frontend re-prompts the user for the metric.
    if not route.download_metric:
        return Branch(
            name="download_data",
            route=route,
            ui={"download_needs_metric": True},
            timescale="instant",
            answer=(
                f"Which metric would you like to download? You can choose {_DOWNLOAD_METRIC_LABELS}."
            ),
        )

    dl = _build_download(route, question)
    return Branch(
        name="download_data",
        route=route,
        ui={
            "download_needs_metric": False,
            "download_slug": dl["slug"],
            "download_metric_type": dl["metric_type"],
            "download_start": dl["start"],
            "download_end": dl["end"],
            "download_interval": dl["interval"],
            "download_format": dl["format"],
        },
        timescale="instant",
        answer=(
            f"Here's your {dl['format'].upper()} download of {dl['metric_type']} for "
            f"{dl['window_label']} — use the button to save the readings."
        ),
    )


def _ifc_branch(route: RoutePlan, question: str) -> Branch:
    def _run(_prepared: Any) -> Dict[str, Any]:
        result = answer_ifc_question_with_metadata(user_question=question)
        # The IFC model has no per-claim numbered citations, so citation_sources stays
        # empty; provenance (the model file) is surfaced in metadata instead.
        return _core_result(
            result.get("answer") or "",
            llm_used=bool(result.get("llm_used", False)),
            meta={
                "model_available": bool(result.get("model_available", True)),
                "model_source": list(result.get("indexed_sources") or []),
            },
        )

    return Branch(
        name="ifc_qa",
        route=route,
        ui={"mode": "conversational", "panel": "ifc", "metrics": [], "transition": "fade"},
        timescale="model",
        llm_used=True,
        run_sync=_run,
        open_stream=lambda _prepared: stream_ifc_tokens(user_question=question),
        status_before_render=("reading_model", "Reading building model…"),
    )


def _sensor_branch(route: RoutePlan, question: str, lab_name: Optional[str]) -> Branch:
    def _run(_prepared: Any) -> Dict[str, Any]:
        result = answer_sensor_question_with_metadata(user_question=question, space=lab_name)
        # Narrative-only, like the IFC branch: provenance goes to metadata, not citations.
        return _core_result(
            result.get("answer") or "",
            llm_used=bool(result.get("llm_used", False)),
            meta={"model_source": list(result.get("indexed_sources") or [])},
        )

    return Branch(
        name="sensor_inspection",
        route=route,
        ui={"mode": "conversational", "panel": "sensors", "metrics": [], "transition": "fade"},
        timescale="sensors",
        lab_name=lab_name,
        llm_used=True,
        run_sync=_run,
        open_stream=lambda _prepared: stream_sensor_tokens(user_question=question, space=lab_name),
        status_before_render=("reading_sensors", "Reading sensor status…"),
    )


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


def _knowledge_branch(route: RoutePlan, question: str, lab_name: Optional[str], k: int) -> Branch:
    effective_k = max(1, min(k, 8))

    def _run(live_sensor_data: Any) -> Dict[str, Any]:
        result = answer_env_question_with_metadata(
            user_question=question,
            k=effective_k,
            space=lab_name,
            live_sensor_data=live_sensor_data,
        )
        return _core_result(
            result.get("answer") or "",
            footnotes=list(result.get("footnotes") or []),
            citation_sources=list(result.get("indexed_sources") or []),
            cards_retrieved=int(result.get("cards_retrieved") or 0),
            llm_used=bool(result.get("llm_used", False)),
        )

    return Branch(
        name="knowledge_qa",
        route=route,
        ui=dict(_CONVERSATIONAL_UI),
        timescale="knowledge",
        lab_name=lab_name,
        llm_used=True,
        # Both renderers ground the answer in the same live reading snapshot.
        prepare=lambda: _fetch_live_sensor_data(question, lab_name, route),
        run_sync=_run,
        open_stream=lambda live_sensor_data: stream_knowledge_tokens(
            user_question=question,
            k=effective_k,
            space=lab_name,
            live_sensor_data=live_sensor_data,
        ),
        status_before_prepare=("searching_knowledge", "Searching knowledge base…"),
    )


def _db_branch(
    route: RoutePlan,
    question: str,
    lab_name: Optional[str],
    llm_history: str,
    planner_hints: Dict[str, Any],
) -> Branch:
    def _resolved_ui(metrics: list) -> Dict[str, Any]:
        return derive_ui_contract(
            execution_intent=route.intent,
            metrics=metrics or list(route.metrics),
            has_floor_comparison=False,
            clarification_required=False,
            use_knowledge_executor=False,
        )

    def _run(_prepared: Any) -> Dict[str, Any]:
        db_result = run_db_query(
            question=question,
            intent=route.intent,
            lab_name=lab_name,
            planner_hints=planner_hints,
            conversation_context=llm_history,
        )
        metrics = list(db_result.get("metrics_used") or planner_hints.get("metrics_priority") or [])
        return _core_result(
            db_result.get("answer") or "",
            footnotes=list(db_result.get("footnotes") or []),
            citation_sources=list(db_result.get("indexed_sources") or []),
            data=db_result.get("data"),
            cards_retrieved=int(db_result.get("cards_retrieved") or 0),
            llm_used=bool(db_result.get("llm_used", False)),
            timescale=db_result.get("timescale") or "1hour",
            ui=_resolved_ui(metrics),
            meta={
                "resolved_lab_name": db_result.get("resolved_lab_name"),
                "time_window": db_result.get("time_window"),
            },
        )

    def _meta_update(query_context: Dict[str, Any]) -> Dict[str, Any]:
        # Reconcile the placeholder meta (emitted before the query ran, with timescale
        # "pending" and a route-metric-derived UI) with the resolved values. This keeps the
        # streamed metadata in parity with the sync response, which derives its UI and
        # timescale from the same resolved facts.
        metrics_used = list((query_context or {}).get("metrics_used") or [])
        return {
            "timescale": (query_context or {}).get("timescale"),
            "time_window": (query_context or {}).get("time_window"),
            "resolved_lab_name": (query_context or {}).get("resolved_lab_name"),
            "metrics_used": metrics_used,
            "ui": _resolved_ui(metrics_used),
        }

    return Branch(
        name="db_query",
        route=route,
        ui=_resolved_ui([]),
        # The stream advertises "pending" until the query resolves the real granularity;
        # the sync renderer takes the resolved value from the result instead.
        timescale="pending",
        lab_name=lab_name,
        llm_used=True,
        prepare=lambda: prepare_db_query(question, route.intent, lab_name, planner_hints),
        run_sync=_run,
        open_stream=lambda query_context: stream_db_tokens(
            question=question,
            intent=route.intent,
            lab_name=lab_name,
            planner_hints=planner_hints,
            query_context=query_context,
            conversation_context=llm_history,
        ),
        status_before_prepare=("querying_db", "Fetching sensor data…"),
        status_before_render=("building_response", "Building response…"),
        stream_meta_update=_meta_update,
    )


# ---------------------------------------------------------------------------
# The single ladder
# ---------------------------------------------------------------------------

def _resolved_question(ctx: ConversationContext, route: RoutePlan) -> str:
    """The self-contained question the executors and answer LLM should act on.

    Prefer the router's LLM-resolved rewrite (references filled from prior turns);
    fall back to the clean current-turn question when the router produced none
    (e.g. regex fallback, or an already self-contained question)."""
    resolved = str(route.resolved_question or "").strip()
    return resolved or ctx.effective_question


def plan_branch(
    ctx: ConversationContext,
    route: RoutePlan,
    k: int = 5,
    allow_clarify: bool = True,
) -> Branch:
    """Map a route plan to the one branch that will answer it.

    This is the only intent ladder in the system — both renderers consume its result.
    """
    question = _resolved_question(ctx, route)

    # The router asks for clarification only when answering would mean guessing. A caller
    # that opted out (allow_clarify=False) gets the best-effort answer instead.
    if allow_clarify and route.needs_clarification:
        return _clarify_branch(route)

    executor = _choose_executor(route)
    if executor == RouteExecutor.VIEWER_CONTROL:
        return _viewer_branch(route)
    if executor == RouteExecutor.HEATMAP_CONTROL:
        return _heatmap_branch(route)
    if executor == RouteExecutor.DOWNLOAD_DATA:
        return _download_branch(route, question)
    if executor == RouteExecutor.IFC_QA:
        return _ifc_branch(route, question)
    if executor == RouteExecutor.SENSOR_INSPECTION:
        return _sensor_branch(route, question, ctx.effective_lab)
    if route.intent == IntentType.UNKNOWN_FALLBACK:
        return _unknown_branch(route)
    if executor == RouteExecutor.KNOWLEDGE_QA:
        return _knowledge_branch(route, question, ctx.effective_lab, k)
    return _db_branch(
        route,
        question,
        ctx.effective_lab,
        ctx.llm_history,
        _build_planner_hints(
            route,
            carried_time_phrase=ctx.carried_time_phrase,
            carried_metric=ctx.carried_metric,
        ),
    )


def _branch_metadata(
    branch: Branch,
    *,
    ui: Optional[Dict[str, Any]] = None,
    llm_used: Optional[bool] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Metadata common to both renderers, so neither can drift from the other."""
    meta: Dict[str, Any] = {
        "executor": branch.name,
        "intent": branch.route.intent.value,
        "lab_name": branch.lab_name,
        "llm_used": branch.llm_used if llm_used is None else bool(llm_used),
        "route_confidence": branch.route.confidence,
        "planner_model": branch.route.model,
        "fallback_used": branch.route.fallback_used,
        "ui": ui if ui is not None else branch.ui,
    }
    meta.update(branch.extra_meta)
    meta.update(extra or {})
    return meta


# ---------------------------------------------------------------------------
# Renderer 1: sync JSON
# ---------------------------------------------------------------------------

def render_sync(branch: Branch) -> Dict[str, Any]:
    """Run a branch and return the ``/query`` response body."""
    if branch.answer is not None:
        return {
            "answer": branch.answer,
            "footnotes": [],
            "citation_sources": [],
            "timescale": branch.timescale,
            "cards_retrieved": 0,
            "recent_card": False,
            "metadata": _branch_metadata(branch),
            "data": None,
        }

    prepared = branch.prepare() if branch.prepare else None
    result = branch.run_sync(prepared)
    return {
        "answer": result["answer"],
        "footnotes": result["footnotes"],
        "citation_sources": result["citation_sources"],
        "timescale": result["timescale"] or branch.timescale,
        "cards_retrieved": result["cards_retrieved"],
        "recent_card": False,
        "metadata": _branch_metadata(
            branch,
            ui=result["ui"],
            llm_used=result["llm_used"],
            extra=result["meta"],
        ),
        "data": result["data"],
    }


def execute_query(
    ctx: ConversationContext,
    k: int,
    allow_clarify: bool = True,
    endpoint_key: str = "query_sync",
) -> Dict[str, Any]:
    """Execute a query given a fully-resolved ConversationContext."""
    route = plan_route(ctx.effective_question, ctx.effective_lab, ctx.routing_snippet)
    branch = plan_branch(ctx, route, k=k, allow_clarify=allow_clarify)
    result = render_sync(branch)
    _attach_resolved_question(result.get("metadata"), ctx, route)
    return result


def _attach_resolved_question(
    metadata: Optional[Dict[str, Any]],
    ctx: ConversationContext,
    route: RoutePlan,
) -> None:
    """Surface the resolved question for observability when context was actually applied."""
    if not isinstance(metadata, dict):
        return
    question = _resolved_question(ctx, route)
    if question.strip() != ctx.original_question.strip():
        metadata["resolved_question"] = question


# ---------------------------------------------------------------------------
# Renderer 2: SSE stream
# ---------------------------------------------------------------------------

def _sse(payload: Dict[str, Any]) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _status_event(stage: str, message: str) -> str:
    return _sse({"event": "status", "stage": stage, "message": message})


def _stream_meta_frame(branch: Branch, extra: Optional[Dict[str, Any]] = None) -> str:
    """The ``meta`` frame: branch metadata plus the flat fields the stream contract adds.

    ``citation_sources``/``footnotes`` are empty here by construction — the stream cannot
    know which sources the answer cites until it has been generated, so executors send
    them in the terminal ``sources`` frame instead.
    """
    meta = _branch_metadata(branch, extra=extra)
    meta.update(
        {
            "timescale": branch.timescale,
            "cards_retrieved": 0,
            "recent_card": False,
            "visualization_type": "none",
            "chart": None,
            "citation_sources": [],
            "footnotes": [],
        }
    )
    return _sse({"event": "meta", **meta})


async def render_stream(branch: Branch, meta_extra: Optional[Dict[str, Any]] = None) -> AsyncIterator[str]:
    """Run a branch and emit the ``/query/stream`` SSE frames."""
    yield _stream_meta_frame(branch, extra=meta_extra)

    if branch.answer is not None:
        yield _sse({"event": "token", "text": branch.answer})
        yield _sse({"event": "done"})
        return

    prepared = None
    if branch.prepare:
        if branch.status_before_prepare:
            yield _status_event(*branch.status_before_prepare)
        # Branch pre-work is blocking (HTTP calls, file parsing) — keep it off the loop.
        prepared = await run_in_threadpool(branch.prepare)
        if branch.stream_meta_update:
            yield _sse({"event": "meta_update", **branch.stream_meta_update(prepared)})

    if branch.status_before_render:
        yield _status_event(*branch.status_before_render)

    async for chunk in branch.open_stream(prepared):
        yield chunk


async def stream_query(
    ctx: ConversationContext,
    k: int,
    allow_clarify: bool = True,
    endpoint_key: str = "query_stream",
) -> AsyncIterator[str]:
    """Stream a query given a fully-resolved ConversationContext."""
    yield _status_event("routing", "Classifying question…")
    route = await plan_route_async(ctx.effective_question, ctx.effective_lab, ctx.routing_snippet)
    branch = plan_branch(ctx, route, k=k, allow_clarify=allow_clarify)

    meta_extra: Dict[str, Any] = {}
    _attach_resolved_question(meta_extra, ctx, route)

    async for chunk in render_stream(branch, meta_extra=meta_extra):
        yield chunk
