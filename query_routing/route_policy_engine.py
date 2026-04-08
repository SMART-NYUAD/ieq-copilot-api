"""Centralized deterministic routing policy engine."""

from __future__ import annotations

from hashlib import sha256
from typing import Dict, Optional, Tuple

try:
    from core_settings import load_settings
    from query_routing.intent_classifier import IntentType
    from query_routing.router_types import (
        AgentAction,
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
        AgentAction,
        AnswerStrategy,
        QueryScopeClass,
        RouteDecisionContract,
        RouteExecutor,
        RoutePlan,
    )


POLICY_VERSION = "route-policy-v1"
SEMANTIC_INTENTS = {IntentType.DEFINITION_EXPLANATION, IntentType.UNKNOWN_FALLBACK}


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


def _clarify_threshold() -> float:
    return float(load_settings().router_clarify_threshold)


def _strict_mode_enabled() -> bool:
    raw = getattr(load_settings(), "agent_routing_strict", False)
    if isinstance(raw, bool):
        return raw
    text = str(raw or "").strip().lower()
    return text in {"1", "true", "yes", "on"}


def _should_clarify(route_plan: RoutePlan, scope_class: str, allow_clarify: bool) -> bool:
    if not allow_clarify:
        return False
    signals = _query_signals(route_plan)
    explicit_measured_scope = (
        bool(signals.get("asks_for_db_facts"))
        and (
            bool(signals.get("has_lab_reference"))
            or bool(signals.get("has_time_window_hint"))
            or bool(signals.get("has_db_scope_phrase"))
            or bool(signals.get("has_metric_reference"))
            or bool(signals.get("is_air_assessment_phrase"))
            or bool(signals.get("is_comfort_assessment_phrase"))
            or bool(signals.get("is_diagnostic_phrase"))
        )
    )
    if _strict_mode_enabled():
        planner_requests_clarify = (
            route_plan.answer_strategy == AnswerStrategy.CLARIFY
            or getattr(route_plan, "agent_action", AgentAction.FINALIZE) == AgentAction.CLARIFY
        )
        if planner_requests_clarify and explicit_measured_scope:
            return False
        return planner_requests_clarify
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
    if _strict_mode_enabled():
        if scope_class == QueryScopeClass.NON_DOMAIN.value:
            return False
        return route_plan.decision.intent not in SEMANTIC_INTENTS
    signals = _query_signals(route_plan)
    is_hypothetical = bool(signals.get("is_hypothetical_conditional"))
    requests_live_data = bool(signals.get("requests_current_measured_data"))
    is_single_lab_baseline = bool(signals.get("single_explicit_lab_with_baseline_reference"))
    is_semantic_intent = route_plan.decision.intent in SEMANTIC_INTENTS
    has_time_scope = bool(signals.get("has_time_window_hint"))
    has_lab_scope = bool(signals.get("has_lab_reference"))
    has_db_phrase = bool(signals.get("has_db_scope_phrase"))
    if is_hypothetical and not requests_live_data:
        return False
    if is_single_lab_baseline:
        return True
    if scope_class == QueryScopeClass.NON_DOMAIN.value:
        return False
    if (
        is_semantic_intent
        and bool(signals.get("is_general_knowledge_question"))
        and not has_time_scope
        and not has_lab_scope
        and not has_db_phrase
    ):
        return False
    if bool(signals.get("asks_for_db_facts")):
        return True
    if scope_class == QueryScopeClass.DOMAIN.value:
        return True
    if has_lab_scope or has_time_scope:
        return True
    if bool(signals.get("has_metric_reference")) and has_db_phrase:
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
    strict_mode = _strict_mode_enabled()
    signals = _query_signals(route_plan)
    decision_intent = route_plan.decision.intent
    is_hypothetical = bool(signals.get("is_hypothetical_conditional"))
    requests_live_data = bool(signals.get("requests_current_measured_data"))
    is_single_lab_baseline = bool(signals.get("single_explicit_lab_with_baseline_reference"))
    trace = []

    if scope_class == QueryScopeClass.NON_DOMAIN.value:
        trace.append("non_domain_scope_forces_knowledge")
        return RouteExecutor.KNOWLEDGE_QA, False, tuple(trace)

    if strict_mode:
        if _should_clarify(route_plan=route_plan, scope_class=scope_class, allow_clarify=allow_clarify):
            trace.append("strict_mode_planner_clarify")
            return RouteExecutor.CLARIFY_GATE, False, tuple(trace)
        if decision_intent in SEMANTIC_INTENTS:
            trace.append("strict_mode_semantic_intent_knowledge")
            return RouteExecutor.KNOWLEDGE_QA, False, tuple(trace)
        trace.append("strict_mode_non_semantic_db")
        return RouteExecutor.DB_QUERY, True, tuple(trace)

    if is_hypothetical and not requests_live_data:
        trace.append("hypothetical_without_live_scope_forces_knowledge")
        return RouteExecutor.KNOWLEDGE_QA, False, tuple(trace)

    if is_single_lab_baseline:
        trace.append("single_lab_baseline_forces_db")
        return RouteExecutor.DB_QUERY, True, tuple(trace)

    if _should_clarify(route_plan=route_plan, scope_class=scope_class, allow_clarify=allow_clarify):
        trace.append("clarify_gate_confidence_or_strategy")
        return RouteExecutor.CLARIFY_GATE, False, tuple(trace)

    needs_measured_data = _needs_measured_data(route_plan=route_plan, scope_class=scope_class)
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
