"""Shared helpers for HTTP route adapters.

This module centralizes repeated route-layer mechanics so endpoint files can
focus on transport specifics instead of metadata and conversation plumbing.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

try:
    from query_routing.metadata_builders import (
        attach_conversation_metadata as shared_attach_conversation_metadata,
        attach_policy_metadata as shared_attach_policy_metadata,
        route_plan_metadata as shared_route_plan_metadata,
    )
    from storage.conversation_store import append_conversation_turn, build_compact_context
except ImportError:
    from ..query_routing.metadata_builders import (
        attach_conversation_metadata as shared_attach_conversation_metadata,
        attach_policy_metadata as shared_attach_policy_metadata,
        route_plan_metadata as shared_route_plan_metadata,
    )
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
    return shared_route_plan_metadata(
        route_plan=route_plan,
        include_decomposition_template=include_decomposition_template,
        query_scope_override=query_scope_override,
    )


def attach_conversation_metadata(
    metadata: Dict[str, Any],
    *,
    conversation_id: Optional[str],
    conversation_context_applied: bool,
    turn_index: Optional[int],
) -> Dict[str, Any]:
    return shared_attach_conversation_metadata(
        metadata=metadata,
        conversation_id=conversation_id,
        conversation_context_applied=conversation_context_applied,
        turn_index=turn_index,
    )


def attach_policy_metadata(metadata: Dict[str, Any], route_contract: Optional[Any]) -> Dict[str, Any]:
    return shared_attach_policy_metadata(metadata=metadata, route_contract=route_contract)

