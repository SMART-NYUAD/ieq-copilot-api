"""Shared helpers for HTTP route adapters.

This module centralizes repeated route-layer mechanics so endpoint files can
focus on transport specifics instead of metadata and conversation plumbing.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

try:
    from storage.conversation_store import append_conversation_turn, build_compact_context
except ImportError:
    from ..storage.conversation_store import append_conversation_turn, build_compact_context


SSE_HEADERS: Dict[str, str] = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def build_query_inputs(
    question: str,
    conversation_id: Optional[str],
) -> Tuple[str, Optional[str], str, bool]:
    """Split latest question and conversation context for deterministic routing."""
    normalized_conversation_id, context_block = build_compact_context(conversation_id)
    if context_block:
        return str(question or "").strip(), normalized_conversation_id, context_block, True
    return str(question or "").strip(), normalized_conversation_id, "", False


def persist_turn(conversation_id: Optional[str], question: str, answer: str) -> Optional[int]:
    """Persist one conversation turn if a conversation id is provided."""
    if not conversation_id:
        return None
    return append_conversation_turn(
        conversation_id=conversation_id,
        user_message=question,
        assistant_message=answer,
    )


def route_plan_metadata(
    route_plan: Any,
    *,
    include_decomposition_template: bool = True,
    query_scope_override: Optional[str] = None,
) -> Dict[str, Any]:
    """Build route metadata fields shared across endpoints.

    ``route_plan`` is intentionally typed as ``Any`` to keep this helper
    transport-focused and avoid importing planner dataclasses at route layer.
    """
    decision = route_plan.decision
    query_scope_class = query_scope_override
    if query_scope_class is None:
        query_scope_class = (
            str((route_plan.planner_parameters.get("query_signals") or {}).get("query_scope_class") or "")
            .strip()
            .lower()
            or None
        )

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
        "query_scope_class": query_scope_class,
        "agent_action": str(getattr(getattr(route_plan, "agent_action", None), "value", "finalize")),
        "tool_name": getattr(route_plan, "tool_name", None),
        "expected_observation": getattr(route_plan, "expected_observation", None),
    }
    if include_decomposition_template:
        meta["decomposition_template"] = (
            route_plan.decomposition_template.value if route_plan.decomposition_template else None
        )
    return meta


def attach_conversation_metadata(
    metadata: Dict[str, Any],
    *,
    conversation_id: Optional[str],
    conversation_context_applied: bool,
    turn_index: Optional[int],
) -> Dict[str, Any]:
    """Attach normalized conversation metadata fields to a payload."""
    merged = dict(metadata or {})
    merged["conversation_id"] = conversation_id
    merged["conversation_context_applied"] = conversation_context_applied
    merged["turn_index"] = turn_index
    return merged


def attach_policy_metadata(metadata: Dict[str, Any], route_contract: Optional[Any]) -> Dict[str, Any]:
    """Attach deterministic policy-engine fields when available."""
    merged = dict(metadata or {})
    if route_contract is None:
        return merged
    merged["latest_question_hash"] = getattr(route_contract, "latest_question_hash", None)
    merged["needs_measured_data"] = bool(getattr(route_contract, "needs_measured_data", False))
    merged["policy_version"] = str(getattr(route_contract, "policy_version", "") or "")
    merged["rule_trace"] = list(getattr(route_contract, "rule_trace", ()) or [])
    return merged

