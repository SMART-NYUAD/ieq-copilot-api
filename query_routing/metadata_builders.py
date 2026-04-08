"""Canonical metadata builders shared by routing and HTTP adapters."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from evidence.evidence_layer import build_repaired_evidence, normalize_evidence
except ImportError:
    from ..evidence.evidence_layer import build_repaired_evidence, normalize_evidence


def query_scope_class(route_plan: Any) -> Optional[str]:
    return (
        str((route_plan.planner_parameters.get("query_signals") or {}).get("query_scope_class") or "")
        .strip()
        .lower()
        or None
    )


def base_route_metadata(
    route_plan: Any,
    decision: Any,
    *,
    include_decomposition_template: bool = True,
    include_query_signals: bool = True,
    query_scope_override: Optional[str] = None,
) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
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
        "query_scope_class": query_scope_override if query_scope_override is not None else query_scope_class(route_plan),
        "agent_action": str(getattr(getattr(route_plan, "agent_action", None), "value", "finalize")),
        "tool_name": getattr(route_plan, "tool_name", None),
        "expected_observation": getattr(route_plan, "expected_observation", None),
    }
    if include_decomposition_template:
        meta["decomposition_template"] = (
            route_plan.decomposition_template.value if route_plan.decomposition_template else None
        )
    if include_query_signals:
        meta["query_signals"] = route_plan.planner_parameters.get("query_signals", {})
    return meta


def route_plan_metadata(
    route_plan: Any,
    *,
    include_decomposition_template: bool = True,
    query_scope_override: Optional[str] = None,
) -> Dict[str, Any]:
    return base_route_metadata(
        route_plan=route_plan,
        decision=route_plan.decision,
        include_decomposition_template=include_decomposition_template,
        include_query_signals=False,
        query_scope_override=query_scope_override,
    )


def attach_conversation_metadata(
    metadata: Dict[str, Any],
    *,
    conversation_id: Optional[str],
    conversation_context_applied: bool,
    turn_index: Optional[int],
) -> Dict[str, Any]:
    merged = dict(metadata or {})
    merged["conversation_id"] = conversation_id
    merged["conversation_context_applied"] = conversation_context_applied
    merged["turn_index"] = turn_index
    return merged


def attach_policy_metadata(metadata: Dict[str, Any], route_contract: Optional[Any]) -> Dict[str, Any]:
    merged = dict(metadata or {})
    if route_contract is None:
        return merged
    merged["latest_question_hash"] = getattr(route_contract, "latest_question_hash", None)
    merged["needs_measured_data"] = bool(getattr(route_contract, "needs_measured_data", False))
    merged["policy_version"] = str(getattr(route_contract, "policy_version", "") or "")
    merged["rule_trace"] = list(getattr(route_contract, "rule_trace", ()) or [])
    return merged


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
        **base_route_metadata(route_plan, decision),
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
        **base_route_metadata(route_plan, decision),
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
        **base_route_metadata(route_plan, decision),
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
