"""Application use cases for routed query execution.

This module isolates branch-specific execution logic from the orchestrator so
the orchestrator can remain a thin coordinator of routing + post-processing.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

try:
    from evidence.evidence_layer import build_repaired_evidence, normalize_evidence
    from executors.db_query_executor import run_db_query
    from executors.env_query_langchain import answer_env_question_with_metadata
except ImportError:
    from ..evidence.evidence_layer import build_repaired_evidence, normalize_evidence
    from ..executors.db_query_executor import run_db_query
    from ..executors.env_query_langchain import answer_env_question_with_metadata


def _query_scope_class(route_plan: Any) -> Optional[str]:
    """Derive normalized query scope class from planner signals."""
    return (
        str((route_plan.planner_parameters.get("query_signals") or {}).get("query_scope_class") or "")
        .strip()
        .lower()
        or None
    )


def _base_route_metadata(route_plan: Any, decision: Any) -> Dict[str, Any]:
    """Build route metadata fields shared by all execution branches."""
    return {
        "route_source": route_plan.route_source,
        "route_type": decision.intent.value,
        "intent_category": route_plan.intent_category.value,
        "route_confidence": decision.confidence,
        "route_reason": decision.reason,
        "planner_model": route_plan.planner_model,
        "planner_fallback_used": route_plan.planner_fallback_used,
        "planner_fallback_reason": route_plan.planner_fallback_reason,
        "answer_strategy": route_plan.answer_strategy.value,
        "secondary_intents": [item.value for item in route_plan.secondary_intents],
        "decomposition_template": (
            route_plan.decomposition_template.value if route_plan.decomposition_template else None
        ),
        "query_signals": route_plan.planner_parameters.get("query_signals", {}),
        "query_scope_class": _query_scope_class(route_plan),
    }


def build_clarify_result(
    *,
    route_plan: Any,
    decision: Any,
    k: int,
    lab_name: Optional[str],
    clarify_threshold: float,
    clarify_text: str,
) -> Dict[str, Any]:
    """Return a standardized clarification-gate response payload."""
    metadata = {
        **_base_route_metadata(route_plan, decision),
        "clarify_threshold": clarify_threshold,
        "clarification_required": True,
        "executor": "clarify_gate",
        "k_requested": k,
        "lab_name": lab_name,
        "evidence": build_repaired_evidence(
            executor="clarify_gate", lab_name=lab_name, reason="clarification_required"
        ),
    }
    return {
        "answer": clarify_text,
        "timescale": "clarify",
        "cards_retrieved": 0,
        "recent_card": False,
        "metadata": metadata,
        "data": None,
        "visualization_type": "none",
        "chart": None,
    }


def execute_knowledge_use_case(
    *,
    question: str,
    k: int,
    lab_name: Optional[str],
    route_plan: Any,
    decision: Any,
    scope_guardrail_builder: Callable[[str], str],
    answer_with_metadata_fn: Callable[..., Dict[str, Any]] = answer_env_question_with_metadata,
) -> Dict[str, Any]:
    """Execute knowledge-answer path and return standard response payload."""
    knowledge_result = answer_with_metadata_fn(
        user_question=question, k=max(1, min(k, 8)), space=lab_name
    )
    answer_text = str(knowledge_result.get("answer") or "")
    fallback_applied = False
    evidence = normalize_evidence(
        raw=knowledge_result.get("evidence"),
        executor="knowledge_qa",
        lab_name=lab_name,
    )
    metadata = {
        **_base_route_metadata(route_plan, decision),
        "clarification_required": False,
        "executor": "knowledge_qa",
        "scope_guardrail_applied": fallback_applied,
        "execution_intent": decision.intent.value,
        "intent_rerouted_to_db": False,
        "k_requested": k,
        "lab_name": lab_name,
        "resolved_lab_name": lab_name,
        "llm_used": True,
        "time_window": None,
        "sources": [],
        "knowledge_cards_retrieved": int(knowledge_result.get("knowledge_cards_retrieved") or 0),
        "visualization_type": "none",
        "evidence": evidence,
    }
    return {
        "answer": answer_text,
        "timescale": "knowledge",
        "cards_retrieved": int(knowledge_result.get("cards_retrieved") or 0),
        "recent_card": False,
        "metadata": metadata,
        "data": None,
        "visualization_type": "none",
        "chart": None,
    }


def execute_db_use_case(
    *,
    question: str,
    k: int,
    lab_name: Optional[str],
    route_plan: Any,
    decision: Any,
    execution_intent: Any,
    run_db_query_fn: Callable[..., Dict[str, Any]] = run_db_query,
) -> Dict[str, Any]:
    """Execute DB path and return standard response payload."""
    db_result = run_db_query_fn(
        question=question,
        intent=execution_intent,
        lab_name=lab_name,
        planner_hints=route_plan.planner_parameters,
    )
    if db_result.get("invariant_violation"):
        invariant = dict(db_result.get("invariant_violation") or {})
        metadata = {
            **_base_route_metadata(route_plan, decision),
            "clarification_required": True,
            "executor": "clarify_gate",
            "execution_intent": execution_intent.value,
            "intent_rerouted_to_db": execution_intent != decision.intent,
            "k_requested": k,
            "lab_name": lab_name,
            "resolved_lab_name": db_result.get("resolved_lab_name"),
            "llm_used": False,
            "time_window": db_result.get("time_window"),
            "sources": [],
            "visualization_type": "none",
            "db_invariant_violation": invariant,
            "evidence": db_result.get("evidence"),
        }
        return {
            "answer": str(db_result.get("answer") or ""),
            "timescale": "clarify",
            "cards_retrieved": 0,
            "recent_card": False,
            "metadata": metadata,
            "data": None,
            "visualization_type": "none",
            "chart": None,
        }
    evidence = normalize_evidence(
        raw=db_result.get("evidence"),
        executor="db_query",
        lab_name=lab_name,
    )
    metadata = {
        **_base_route_metadata(route_plan, decision),
        "clarification_required": False,
        "executor": "db_query",
        "execution_intent": execution_intent.value,
        "intent_rerouted_to_db": execution_intent != decision.intent,
        "k_requested": k,
        "lab_name": lab_name,
        "resolved_lab_name": db_result.get("resolved_lab_name"),
        "llm_used": db_result.get("llm_used", False),
        "time_window": db_result.get("time_window"),
        "sources": db_result.get("sources", []),
        "forecast_model": (db_result.get("forecast") or {}).get("model"),
        "forecast_confidence": (db_result.get("forecast") or {}).get("confidence"),
        "forecast_confidence_score": (db_result.get("forecast") or {}).get("confidence_score"),
        "forecast_horizon_hours": (db_result.get("forecast") or {}).get("horizon_hours"),
        "correlation": db_result.get("correlation"),
        "visualization_type": db_result.get("visualization_type", "none"),
        "evidence": evidence,
    }
    return {
        "answer": db_result["answer"],
        "timescale": db_result["timescale"],
        "cards_retrieved": int(db_result.get("cards_retrieved") or 0),
        "recent_card": False,
        "metadata": metadata,
        "data": db_result.get("data"),
        "visualization_type": db_result.get("visualization_type", "none"),
        "chart": db_result.get("chart"),
    }


def _sources_to_evidence_provenance(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    provenance: List[Dict[str, Any]] = []
    for src in sources:
        provenance.append(
            {
                "source_kind": str(src.get("source_kind") or "unknown"),
                "table": src.get("table"),
                "operation": src.get("operation"),
                "metric": src.get("metric"),
                "source_label": src.get("source_label"),
                "topic": src.get("topic"),
                "title": src.get("title"),
                "details": {
                    "window_label": src.get("window_label"),
                    "row_count": src.get("row_count"),
                    "lab_scope": src.get("lab_scope"),
                    "metrics": src.get("metrics"),
                },
            }
        )
    return provenance


def build_stream_clarify_metadata(
    *,
    route_plan: Any,
    decision: Any,
    k: int,
    lab_name: Optional[str],
    resolved_lab_name: Optional[str],
    time_window: Optional[Dict[str, Any]] = None,
    invariant_violation: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    evidence = build_repaired_evidence(
        executor="clarify_gate",
        lab_name=resolved_lab_name or lab_name,
        reason="clarification_required",
    )
    metadata = {
        **_base_route_metadata(route_plan, decision),
        "timescale": "clarify",
        "cards_retrieved": 0,
        "recent_card": False,
        "clarification_required": True,
        "executor": "clarify_gate",
        "execution_intent": decision.intent.value,
        "intent_rerouted_to_db": False,
        "k_requested": k,
        "lab_name": lab_name,
        "resolved_lab_name": resolved_lab_name,
        "llm_used": False,
        "time_window": time_window,
        "sources": [],
        "forecast_model": None,
        "forecast_confidence": None,
        "forecast_confidence_score": None,
        "forecast_horizon_hours": None,
        "visualization_type": "none",
        "chart": None,
        "evidence": evidence,
    }
    if invariant_violation:
        metadata["db_invariant_violation"] = invariant_violation
    return metadata


def build_stream_knowledge_metadata(
    *,
    route_plan: Any,
    decision: Any,
    k: int,
    lab_name: Optional[str],
    resolved_lab_name: Optional[str],
    cards_retrieved: int,
    knowledge_cards_retrieved: int,
) -> Dict[str, Any]:
    evidence = normalize_evidence(
        raw={
            "evidence_kind": "knowledge_qa",
            "intent": decision.intent.value,
            "strategy": "direct",
            "metric_aliases": [],
            "resolved_scope": resolved_lab_name or lab_name,
            "resolved_time_window": None,
            "provenance_sources": [],
            "confidence_notes": [],
            "recommendation_allowed": True,
        },
        executor="knowledge_qa",
        lab_name=resolved_lab_name or lab_name,
    )
    return {
        **_base_route_metadata(route_plan, decision),
        "timescale": "knowledge",
        "cards_retrieved": int(cards_retrieved or 0),
        "knowledge_cards_retrieved": int(knowledge_cards_retrieved or 0),
        "recent_card": False,
        "clarification_required": False,
        "executor": "knowledge_qa",
        "execution_intent": decision.intent.value,
        "intent_rerouted_to_db": False,
        "k_requested": k,
        "lab_name": lab_name,
        "resolved_lab_name": resolved_lab_name,
        "llm_used": True,
        "time_window": None,
        "sources": [],
        "forecast_model": None,
        "forecast_confidence": None,
        "forecast_confidence_score": None,
        "forecast_horizon_hours": None,
        "visualization_type": "none",
        "chart": None,
        "evidence": evidence,
    }


def build_stream_db_metadata(
    *,
    route_plan: Any,
    decision: Any,
    execution_intent: Any,
    k: int,
    lab_name: Optional[str],
    db_context: Dict[str, Any],
) -> Dict[str, Any]:
    sources = list(db_context.get("sources") or [])
    confidence_notes: List[str] = []
    if not list(db_context.get("rows") or []) and not db_context.get("forecast"):
        confidence_notes.append("low_data_coverage")
    evidence = normalize_evidence(
        raw={
            "evidence_kind": "db_query",
            "intent": execution_intent.value,
            "strategy": "direct",
            "metric_aliases": [str(db_context.get("metric_alias") or "")],
            "resolved_scope": db_context.get("resolved_lab_name") or lab_name,
            "resolved_time_window": db_context.get("time_window"),
            "provenance_sources": _sources_to_evidence_provenance(sources),
            "confidence_notes": confidence_notes,
            "recommendation_allowed": True,
        },
        executor="db_query",
        lab_name=db_context.get("resolved_lab_name") or lab_name,
    )
    return {
        **_base_route_metadata(route_plan, decision),
        "timescale": db_context.get("timescale", "1hour"),
        "cards_retrieved": int(db_context.get("cards_retrieved") or 0),
        "recent_card": False,
        "clarification_required": False,
        "executor": "db_query",
        "execution_intent": execution_intent.value,
        "intent_rerouted_to_db": execution_intent != decision.intent,
        "k_requested": k,
        "lab_name": lab_name,
        "resolved_lab_name": db_context.get("resolved_lab_name"),
        "llm_used": True,
        "time_window": db_context.get("time_window"),
        "sources": sources,
        "forecast_model": (db_context.get("forecast") or {}).get("model"),
        "forecast_confidence": (db_context.get("forecast") or {}).get("confidence"),
        "forecast_confidence_score": (db_context.get("forecast") or {}).get("confidence_score"),
        "forecast_horizon_hours": (db_context.get("forecast") or {}).get("horizon_hours"),
        "visualization_type": db_context.get("visualization_type", "none"),
        "chart": db_context.get("chart"),
        "evidence": evidence,
    }

