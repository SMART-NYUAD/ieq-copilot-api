"""Top-level query orchestration: route → execute → return."""

from __future__ import annotations

from typing import Any, AsyncIterator, Dict, List, Optional

try:
    from executors.db_query_executor import run_db_query, stream_db_query
    from executors.knowledge_executor import (
        answer_env_question_with_metadata,
        stream_answer_env_question,
    )
    from query_routing.intent_classifier import IntentType
    from query_routing.llm_router_planner import plan_route
    from query_routing.router_types import RoutePlan, RouteExecutor
    from storage.conversation_memory import apply_routing_memory, extract_routing_memory
except ImportError:
    from ..executors.db_query_executor import run_db_query, stream_db_query
    from ..executors.knowledge_executor import (
        answer_env_question_with_metadata,
        stream_answer_env_question,
    )
    from .intent_classifier import IntentType
    from .llm_router_planner import plan_route
    from .router_types import RoutePlan, RouteExecutor
    from ..storage.conversation_memory import apply_routing_memory, extract_routing_memory

_KNOWLEDGE_INTENTS = {IntentType.DEFINITION_EXPLANATION, IntentType.UNKNOWN_FALLBACK}


def _choose_executor(route: RoutePlan) -> RouteExecutor:
    if route.intent in _KNOWLEDGE_INTENTS:
        return RouteExecutor.KNOWLEDGE_QA
    return RouteExecutor.DB_QUERY


def _build_planner_hints(route: RoutePlan) -> Dict[str, Any]:
    return {
        "metrics_priority": list(route.metrics),
        "needs_cards": route.intent in _KNOWLEDGE_INTENTS,
        "card_topics": ["definitions", "metric_explanations"] if route.intent in _KNOWLEDGE_INTENTS else ["metric_explanations"],
        "max_cards": 2,
        "second_lab_name": route.second_lab_name,
    }


def _execute_knowledge(
    question: str,
    k: int,
    lab_name: Optional[str],
    route: RoutePlan,
) -> Dict[str, Any]:
    result = answer_env_question_with_metadata(
        user_question=question,
        k=max(1, min(k, 8)),
        space=lab_name,
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
        },
        "data": None,
        "visualization_type": "none",
        "chart": None,
    }


def _execute_db(
    question: str,
    k: int,
    lab_name: Optional[str],
    route: RoutePlan,
) -> Dict[str, Any]:
    planner_hints = _build_planner_hints(route)
    db_result = run_db_query(
        question=question,
        intent=route.intent,
        lab_name=lab_name,
        planner_hints=planner_hints,
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
            "visualization_type": db_result.get("visualization_type", "none"),
        },
        "data": db_result.get("data"),
        "visualization_type": db_result.get("visualization_type", "none"),
        "chart": db_result.get("chart"),
    }


def _resolve_context(
    question: str,
    lab_name: Optional[str],
    conversation_context: str,
) -> tuple[str, Optional[str]]:
    """Apply conversation memory carry-over to question and lab."""
    current_signals: Dict[str, Any] = {}
    routing_memory = extract_routing_memory(
        conversation_context=conversation_context,
        current_signals=current_signals,
    )
    effective_question, effective_lab, _ = apply_routing_memory(
        question=question,
        lab_name=lab_name,
        memory=routing_memory,
        current_signals=current_signals,
    )
    return effective_question, effective_lab


def execute_query(
    question: str,
    k: int,
    lab_name: Optional[str],
    allow_clarify: bool = True,
    endpoint_key: str = "query_sync",
    conversation_context: str = "",
) -> Dict[str, Any]:
    effective_question, effective_lab = _resolve_context(question, lab_name, conversation_context)
    route = plan_route(effective_question, effective_lab)
    executor = _choose_executor(route)

    if executor == RouteExecutor.KNOWLEDGE_QA:
        return _execute_knowledge(effective_question, k, effective_lab, route)
    return _execute_db(effective_question, k, effective_lab, route)


async def stream_query(
    question: str,
    k: int,
    lab_name: Optional[str],
    endpoint_key: str = "query_stream",
    conversation_context: str = "",
) -> AsyncIterator[str]:
    effective_question, effective_lab = _resolve_context(question, lab_name, conversation_context)
    route = plan_route(effective_question, effective_lab)
    executor = _choose_executor(route)

    if executor == RouteExecutor.KNOWLEDGE_QA:
        async for chunk in stream_answer_env_question(
            user_question=effective_question,
            k=max(1, min(k, 8)),
            space=effective_lab,
        ):
            yield chunk
        return

    planner_hints = _build_planner_hints(route)
    async for chunk in stream_db_query(
        question=effective_question,
        intent=route.intent,
        lab_name=effective_lab,
        planner_hints=planner_hints,
    ):
        yield chunk


# Legacy compatibility shims.
def get_route_plan(question: str, lab_name: Optional[str] = None) -> RoutePlan:
    return plan_route(question, lab_name)


_SEMANTIC_INTENTS = {IntentType.DEFINITION_EXPLANATION, IntentType.UNKNOWN_FALLBACK}


def resolve_execution_intent(intent: IntentType) -> IntentType:
    """Return a DB-executable intent (maps semantic intents to current_status_db)."""
    if intent in _SEMANTIC_INTENTS:
        return IntentType.CURRENT_STATUS_DB
    return intent
