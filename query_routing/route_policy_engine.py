"""Centralized deterministic routing policy engine."""

from __future__ import annotations

from hashlib import sha256
from typing import Dict, Optional, Tuple

try:
    from core_settings import load_settings
    from query_routing.intent_classifier import IntentType
    from query_routing.router_types import (
        AnswerStrategy,
        QueryScopeClass,
        RouteDecisionContract,
        RouteExecutor,
        RoutePlan,
    )
except ImportError:
    from ..core_settings import load_settings
    from .intent_classifier import IntentType
    from .router_types import (
        AnswerStrategy,
        QueryScopeClass,
        RouteDecisionContract,
        RouteExecutor,
        RoutePlan,
    )


POLICY_VERSION = "route-policy-v1"
SEMANTIC_INTENTS = {IntentType.DEFINITION_EXPLANATION, IntentType.UNKNOWN_FALLBACK}
_PLANNER_CLARIFY_REASONS = {"no_lab", "no_second_lab", "ambiguous_intent"}


def _question_hash(latest_user_question: str) -> str:
    text = str(latest_user_question or "").strip().lower()
    return sha256(text.encode("utf-8")).hexdigest()[:16]


def _query_signals(route_plan: RoutePlan) -> Dict[str, object]:
    planner_parameters = route_plan.planner_parameters or {}
    raw = planner_parameters.get("query_signals")
    return raw if isinstance(raw, dict) else {}


def _query_scope_class(route_plan: RoutePlan) -> str:
    scope = str(_query_signals(route_plan).get("query_scope_class") or "").strip().lower()
    return scope or QueryScopeClass.AMBIGUOUS.value


def _planner_clarify_reason(route_plan: RoutePlan) -> Optional[str]:
    reason = str(route_plan.clarify_reason or "").strip().lower()
    if reason in _PLANNER_CLARIFY_REASONS:
        return reason
    planner_parameters = route_plan.planner_parameters or {}
    reason = str(planner_parameters.get("clarify_reason") or "").strip().lower()
    if reason in _PLANNER_CLARIFY_REASONS:
        return reason
    return None


def _has_explicit_scope(route_plan: RoutePlan) -> bool:
    if bool(route_plan.has_explicit_scope):
        return True
    planner_parameters = route_plan.planner_parameters or {}
    declared = planner_parameters.get("has_explicit_scope")
    if isinstance(declared, bool):
        return declared
    signals = _query_signals(route_plan)
    return bool(signals.get("has_lab_reference")) or bool(signals.get("has_time_window_hint"))


def _resolved_lab_name(route_plan: RoutePlan) -> Optional[str]:
    if isinstance(route_plan.resolved_lab_name, str) and route_plan.resolved_lab_name.strip():
        return route_plan.resolved_lab_name.strip().lower()
    planner_parameters = route_plan.planner_parameters or {}
    raw = planner_parameters.get("resolved_lab_name")
    if isinstance(raw, str) and raw.strip():
        return raw.strip().lower()
    return None


def _clarify_threshold() -> float:
    return float(load_settings().router_clarify_threshold)


def _should_clarify(route_plan: RoutePlan, scope_class: str, allow_clarify: bool) -> bool:
    if not allow_clarify:
        return False
    if scope_class == QueryScopeClass.NON_DOMAIN.value:
        return False
    if route_plan.decision.intent in SEMANTIC_INTENTS:
        return False
    explicit_measured_scope = _has_explicit_scope(route_plan) and _needs_measured_data(route_plan, scope_class)
    if route_plan.answer_strategy == AnswerStrategy.CLARIFY:
        # Avoid unnecessary clarification when measured DB scope is already explicit.
        if explicit_measured_scope:
            return False
        return True
    confidence = float(route_plan.decision.confidence)
    threshold = _clarify_threshold()
    # Keep an ambiguity floor, but avoid over-triggering clarify for moderately
    # confident planner outputs.
    if scope_class == QueryScopeClass.AMBIGUOUS.value and confidence < max(threshold, 0.58):
        return True
    return confidence < threshold


def _needs_measured_data(route_plan: RoutePlan, scope_class: str) -> bool:
    planner_parameters = route_plan.planner_parameters or {}
    signals = _query_signals(route_plan)

    # Guardrail: semantic intents without explicit measured-data evidence should
    # not be forced into DB/clarify by planner defaults.
    if route_plan.decision.intent in SEMANTIC_INTENTS:
        if not bool(signals.get("asks_for_db_facts")) and not bool(signals.get("requests_current_measured_data")):
            return False

    declared = planner_parameters.get("needs_measured_data")
    if isinstance(declared, bool):
        return declared
    if bool(signals.get("is_general_knowledge_question")):
        return False
    if bool(signals.get("requests_current_measured_data")):
        return True
    return False


def _resolve_execution_intent(decision_intent: IntentType, executor: RouteExecutor) -> IntentType:
    """Ensure DB execution uses DB-capable intent families."""
    if executor == RouteExecutor.DB_QUERY and decision_intent in SEMANTIC_INTENTS:
        return IntentType.CURRENT_STATUS_DB
    return decision_intent


def _choose_executor(
    route_plan: RoutePlan,
    scope_class: str,
    allow_clarify: bool,
) -> Tuple[RouteExecutor, bool, Tuple[str, ...]]:
    signals = _query_signals(route_plan)
    decision_intent = route_plan.decision.intent
    planner_reason = _planner_clarify_reason(route_plan)
    explicit_scope = _has_explicit_scope(route_plan)
    resolved_lab = _resolved_lab_name(route_plan)
    trace = []

    if _should_clarify(route_plan=route_plan, scope_class=scope_class, allow_clarify=allow_clarify):
        trace.append("clarify_gate_confidence_or_strategy")
        return RouteExecutor.CLARIFY_GATE, False, tuple(trace)

    needs_measured_data = _needs_measured_data(route_plan=route_plan, scope_class=scope_class)

    if planner_reason == "ambiguous_intent" and allow_clarify:
        trace.append("planner_clarify_reason:ambiguous_intent")
        return RouteExecutor.CLARIFY_GATE, False, tuple(trace)
    if (
        planner_reason == "no_lab"
        and allow_clarify
        and needs_measured_data
        and (not explicit_scope and not resolved_lab)
    ):
        trace.append("planner_clarify_reason:no_lab")
        return RouteExecutor.CLARIFY_GATE, False, tuple(trace)
    if (
        planner_reason == "no_second_lab"
        and allow_clarify
        and needs_measured_data
        and decision_intent == IntentType.COMPARISON_DB
    ):
        trace.append(f"planner_clarify_reason:{planner_reason}")
        return RouteExecutor.CLARIFY_GATE, False, tuple(trace)

    if scope_class == QueryScopeClass.NON_DOMAIN.value and not needs_measured_data:
        trace.append("non_domain_scope_forces_knowledge")
        return RouteExecutor.KNOWLEDGE_QA, False, tuple(trace)

    if decision_intent in SEMANTIC_INTENTS and not needs_measured_data:
        trace.append("conceptual_semantic_forces_knowledge")
        return RouteExecutor.KNOWLEDGE_QA, False, tuple(trace)

    if bool(signals.get("is_general_knowledge_question")) and not needs_measured_data:
        trace.append("general_knowledge_without_scope_forces_knowledge")
        return RouteExecutor.KNOWLEDGE_QA, False, tuple(trace)

    if needs_measured_data:
        trace.append("measured_scope_forces_db")
        return RouteExecutor.DB_QUERY, True, tuple(trace)

    response_mode = str((route_plan.planner_parameters or {}).get("response_mode") or "").strip().lower()
    if response_mode == "knowledge_only":
        trace.append("planner_knowledge_mode")
        return RouteExecutor.KNOWLEDGE_QA, False, tuple(trace)

    if decision_intent in SEMANTIC_INTENTS:
        trace.append("semantic_default_knowledge")
        return RouteExecutor.KNOWLEDGE_QA, False, tuple(trace)

    trace.append("db_default_for_non_semantic")
    return RouteExecutor.DB_QUERY, True, tuple(trace)


def build_route_decision_contract(
    *,
    latest_user_question: str,
    route_plan: RoutePlan,
    allow_clarify: bool = True,
) -> RouteDecisionContract:
    scope_class = _query_scope_class(route_plan)
    executor, needs_measured_data, trace = _choose_executor(
        route_plan=route_plan,
        scope_class=scope_class,
        allow_clarify=allow_clarify,
    )
    execution_intent = _resolve_execution_intent(route_plan.decision.intent, executor)
    if execution_intent != route_plan.decision.intent:
        trace = tuple(list(trace) + ["semantic_intent_mapped_to_current_status_db"])
    if executor != RouteExecutor.DB_QUERY:
        trace = tuple(list(trace) + ["execution_intent_passthrough_non_db"])
    return RouteDecisionContract(
        latest_user_question=str(latest_user_question or "").strip(),
        latest_question_hash=_question_hash(latest_user_question),
        policy_version=POLICY_VERSION,
        route_plan=route_plan,
        needs_measured_data=needs_measured_data,
        executor=executor,
        execution_intent=execution_intent,
        execution_intent_value=execution_intent.value,
        query_scope_class=scope_class,
        rule_trace=trace,
    )
