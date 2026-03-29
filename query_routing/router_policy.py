"""Policy and normalization helpers for router planning."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

try:
    from query_routing.intent_classifier import IntentType, RouteDecision, classify_intent
    from query_routing.router_settings import legacy_fallback_enabled
    from query_routing.router_types import (
        AnswerStrategy,
        DecompositionTemplate,
        IntentCategory,
        QueryScopeClass,
        RoutePlan,
    )
except ImportError:
    from .intent_classifier import IntentType, RouteDecision, classify_intent
    from .router_settings import legacy_fallback_enabled
    from .router_types import (
        AnswerStrategy,
        DecompositionTemplate,
        IntentCategory,
        QueryScopeClass,
        RoutePlan,
    )


ALLOWED_PLANNER_METRICS = {
    "air_contribution",
    "co2",
    "pm25",
    "temperature",
    "humidity",
    "tvoc",
    "light",
    "sound",
    "ieq",
}
ALLOWED_CARD_TOPICS = {
    "definitions",
    "metric_explanations",
    "ieq_subindex_explanations",
    "recommendations",
    "caveats",
}
ALLOWED_CATEGORY_INTENTS = {
    IntentCategory.SEMANTIC_EXPLANATORY: {
        IntentType.DEFINITION_EXPLANATION,
        IntentType.UNKNOWN_FALLBACK,
    },
    IntentCategory.STRUCTURED_FACTUAL_DB: {
        IntentType.CURRENT_STATUS_DB,
        IntentType.POINT_LOOKUP_DB,
        IntentType.AGGREGATION_DB,
    },
    IntentCategory.ANALYTICAL_VISUALIZATION: {
        IntentType.COMPARISON_DB,
        IntentType.ANOMALY_ANALYSIS_DB,
    },
    IntentCategory.PREDICTION: {
        IntentType.FORECAST_DB,
    },
}
CATEGORY_BY_INTENT = {
    IntentType.DEFINITION_EXPLANATION: IntentCategory.SEMANTIC_EXPLANATORY,
    IntentType.UNKNOWN_FALLBACK: IntentCategory.SEMANTIC_EXPLANATORY,
    IntentType.CURRENT_STATUS_DB: IntentCategory.STRUCTURED_FACTUAL_DB,
    IntentType.POINT_LOOKUP_DB: IntentCategory.STRUCTURED_FACTUAL_DB,
    IntentType.AGGREGATION_DB: IntentCategory.STRUCTURED_FACTUAL_DB,
    IntentType.COMPARISON_DB: IntentCategory.ANALYTICAL_VISUALIZATION,
    IntentType.ANOMALY_ANALYSIS_DB: IntentCategory.ANALYTICAL_VISUALIZATION,
    IntentType.FORECAST_DB: IntentCategory.PREDICTION,
}
DB_INTENTS = {
    IntentType.CURRENT_STATUS_DB,
    IntentType.POINT_LOOKUP_DB,
    IntentType.AGGREGATION_DB,
    IntentType.COMPARISON_DB,
    IntentType.ANOMALY_ANALYSIS_DB,
    IntentType.FORECAST_DB,
}


def _is_air_quality_query_text(question: str) -> bool:
    q = (question or "").lower()
    issue_hints = ("issue", "issues", "problem", "problems", "anything wrong", "wrong")
    currentness_hints = ("right now", "now", "current", "currently", "latest", "today", "at this moment")
    if any(hint in q for hint in issue_hints) and (
        any(hint in q for hint in currentness_hints)
        or "_lab" in q
        or " lab" in q
    ):
        return True
    if any(hint in q for hint in ("air quality", "indoor air quality", "ieq")):
        return True
    if "how is the air" in q or "how was the air" in q:
        return True
    return "the air" in q and ("_lab" in q or " lab" in q)


def _is_comfort_assessment_query_text(question: str) -> bool:
    q = (question or "").lower()
    return any(hint in q for hint in ("comfortable", "comfort", "too hot", "too cold", "stuffy", "dry", "humid"))


def normalize_strategy(raw_plan: Dict[str, Any]) -> AnswerStrategy:
    raw_strategy = str(raw_plan.get("strategy") or "").strip().lower()
    if not raw_strategy:
        return AnswerStrategy.DIRECT
    try:
        return AnswerStrategy(raw_strategy)
    except ValueError:
        return AnswerStrategy.DIRECT


def normalize_secondary_intents(raw_plan: Dict[str, Any], primary_intent: IntentType) -> Tuple[IntentType, ...]:
    raw_secondary = raw_plan.get("secondary_intents")
    if not isinstance(raw_secondary, list):
        return tuple()
    parsed: list[IntentType] = []
    for item in raw_secondary:
        candidate = str(item or "").strip().lower()
        if not candidate:
            continue
        try:
            intent = IntentType(candidate)
        except ValueError:
            continue
        if intent == primary_intent or intent in parsed:
            continue
        parsed.append(intent)
        if len(parsed) >= 1:
            break
    return tuple(parsed)


def map_decomposition_template(strategy: AnswerStrategy, primary_intent: IntentType) -> Optional[DecompositionTemplate]:
    if strategy != AnswerStrategy.DECOMPOSE:
        return None
    if primary_intent in {IntentType.CURRENT_STATUS_DB, IntentType.POINT_LOOKUP_DB}:
        return DecompositionTemplate.STATE_RECOMMENDATION
    if primary_intent in {IntentType.AGGREGATION_DB, IntentType.COMPARISON_DB}:
        return DecompositionTemplate.TREND_INTERPRETATION
    if primary_intent == IntentType.ANOMALY_ANALYSIS_DB:
        return DecompositionTemplate.ANOMALY_EXPLANATION
    return None


def normalize_plan(raw_plan: Dict[str, Any]) -> Tuple[IntentCategory, RouteDecision, AnswerStrategy, Tuple[IntentType, ...], Optional[DecompositionTemplate]]:
    raw_intent = str(raw_plan.get("intent") or "").strip().lower()
    raw_category = str(raw_plan.get("intent_category") or "").strip().lower()
    raw_confidence = raw_plan.get("confidence", 0.0)
    raw_reason = str(raw_plan.get("reason") or "llm_planner")

    try:
        intent = IntentType(raw_intent)
    except ValueError as exc:
        raise ValueError(f"invalid_intent:{raw_intent}") from exc

    if raw_category:
        try:
            category = IntentCategory(raw_category)
        except ValueError as exc:
            raise ValueError(f"invalid_intent_category:{raw_category}") from exc
    else:
        category = CATEGORY_BY_INTENT.get(intent, IntentCategory.SEMANTIC_EXPLANATORY)

    allowed = ALLOWED_CATEGORY_INTENTS.get(category, set())
    if intent not in allowed:
        raise ValueError(f"intent_category_mismatch:{category.value}:{intent.value}")

    try:
        confidence = float(raw_confidence)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    decision = RouteDecision(intent=intent, confidence=confidence, reason=(raw_reason or "llm_planner").strip()[:240])
    strategy = normalize_strategy(raw_plan)
    secondary_intents = normalize_secondary_intents(raw_plan, primary_intent=intent)
    template = map_decomposition_template(strategy=strategy, primary_intent=intent)
    if strategy == AnswerStrategy.DECOMPOSE and template is None:
        strategy = AnswerStrategy.DIRECT
        secondary_intents = tuple()
    elif strategy == AnswerStrategy.DECOMPOSE and not secondary_intents:
        secondary_intents = (IntentType.DEFINITION_EXPLANATION,)
    return category, decision, strategy, secondary_intents, template


def _default_metrics_for(question: str, intent: IntentType) -> list[str]:
    is_air_quality_query = _is_air_quality_query_text(question)
    is_comfort_query = _is_comfort_assessment_query_text(question)
    rerouted_semantic_intents = {IntentType.DEFINITION_EXPLANATION, IntentType.UNKNOWN_FALLBACK}
    if is_comfort_query and intent in {IntentType.CURRENT_STATUS_DB, IntentType.POINT_LOOKUP_DB, IntentType.COMPARISON_DB, IntentType.AGGREGATION_DB, *rerouted_semantic_intents}:
        return ["ieq", "temperature", "humidity", "co2", "pm25", "tvoc", "sound"]
    if is_air_quality_query and intent in {IntentType.COMPARISON_DB, IntentType.AGGREGATION_DB, *rerouted_semantic_intents}:
        return ["ieq", "co2", "pm25", "humidity", "tvoc"]
    if is_air_quality_query and intent in {IntentType.POINT_LOOKUP_DB, IntentType.CURRENT_STATUS_DB}:
        return ["ieq", "co2", "pm25", "humidity", "tvoc"]
    if intent == IntentType.FORECAST_DB:
        return ["pm25"]
    if intent in {IntentType.POINT_LOOKUP_DB, IntentType.CURRENT_STATUS_DB, IntentType.COMPARISON_DB, IntentType.AGGREGATION_DB, IntentType.ANOMALY_ANALYSIS_DB}:
        return ["ieq", "co2", "pm25", "humidity", "tvoc"]
    return ["ieq"]


def normalize_planner_parameters(raw_plan: Dict[str, Any], question: str, intent: IntentType, query_signals: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    rerouted_semantic_intents = {IntentType.DEFINITION_EXPLANATION, IntentType.UNKNOWN_FALLBACK}
    required_air_pack: list[str] = []
    if _is_comfort_assessment_query_text(question) and intent in {IntentType.CURRENT_STATUS_DB, IntentType.POINT_LOOKUP_DB, IntentType.COMPARISON_DB, IntentType.AGGREGATION_DB, *rerouted_semantic_intents}:
        required_air_pack = ["ieq", "temperature", "humidity", "co2", "pm25", "tvoc", "sound"]
    elif _is_air_quality_query_text(question) and intent in {IntentType.CURRENT_STATUS_DB, IntentType.POINT_LOOKUP_DB, IntentType.COMPARISON_DB, IntentType.AGGREGATION_DB, *rerouted_semantic_intents}:
        required_air_pack = ["ieq", "co2", "pm25", "humidity", "tvoc"]

    raw_metrics = raw_plan.get("metrics_priority")
    raw_response_mode = str(raw_plan.get("response_mode") or "").strip().lower()
    raw_needs_cards = raw_plan.get("needs_cards")
    raw_card_topics = raw_plan.get("card_topics")
    raw_max_cards = raw_plan.get("max_cards")
    signals = query_signals or {}
    scope_class = str(signals.get("query_scope_class") or "").strip().lower()

    default_response_mode = "knowledge_only" if scope_class == QueryScopeClass.NON_DOMAIN.value else "db"
    if intent in {IntentType.DEFINITION_EXPLANATION, IntentType.UNKNOWN_FALLBACK} and bool(signals.get("is_general_knowledge_question")):
        default_response_mode = "knowledge_only"

    response_mode = raw_response_mode if raw_response_mode in {"db", "knowledge_only"} else default_response_mode
    if scope_class == QueryScopeClass.NON_DOMAIN.value:
        response_mode = "knowledge_only"
    elif bool(signals.get("asks_for_db_facts")):
        response_mode = "db"

    needs_cards_default = intent == IntentType.DEFINITION_EXPLANATION or response_mode == "knowledge_only"
    needs_cards = bool(raw_needs_cards) if isinstance(raw_needs_cards, bool) else needs_cards_default

    topics: list[str] = []
    if isinstance(raw_card_topics, list):
        for item in raw_card_topics:
            topic = str(item or "").strip().lower().replace(" ", "_")
            if topic in ALLOWED_CARD_TOPICS and topic not in topics:
                topics.append(topic)
    if not topics and needs_cards:
        topics = ["definitions"] if response_mode == "knowledge_only" else ["metric_explanations"]

    try:
        max_cards = int(raw_max_cards)
    except (TypeError, ValueError):
        max_cards = 2
    max_cards = max(1, min(4, max_cards))

    if not isinstance(raw_metrics, list):
        base = _default_metrics_for(question=question, intent=intent)
        cleaned = required_air_pack + [m for m in base if m not in required_air_pack]
    else:
        cleaned: list[str] = []
        for item in raw_metrics:
            metric = str(item or "").strip().lower().replace(" ", "_")
            if metric in ALLOWED_PLANNER_METRICS and metric not in cleaned:
                cleaned.append(metric)
        if not cleaned:
            cleaned = _default_metrics_for(question=question, intent=intent)
        if required_air_pack:
            cleaned = required_air_pack + [m for m in cleaned if m not in required_air_pack]

    return {
        "metrics_priority": cleaned,
        "response_mode": response_mode,
        "needs_cards": needs_cards,
        "card_topics": topics,
        "max_cards": max_cards,
        "query_signals": signals,
    }


def enforce_non_domain_block(decision: RouteDecision, category: IntentCategory, strategy: AnswerStrategy, secondary_intents: Tuple[IntentType, ...], template: Optional[DecompositionTemplate], query_signals: Dict[str, Any]) -> Tuple[RouteDecision, IntentCategory, AnswerStrategy, Tuple[IntentType, ...], Optional[DecompositionTemplate]]:
    scope_class = str(query_signals.get("query_scope_class") or "").strip().lower()
    if scope_class == QueryScopeClass.NON_DOMAIN.value and decision.intent in DB_INTENTS:
        decision = RouteDecision(intent=IntentType.DEFINITION_EXPLANATION, confidence=min(float(decision.confidence), 0.55), reason="non_domain_scope_db_block")
        category = IntentCategory.SEMANTIC_EXPLANATORY
        strategy = AnswerStrategy.DIRECT
        secondary_intents = tuple()
        template = None
    return decision, category, strategy, secondary_intents, template


def fallback_plan(question: str, model: str, fallback_reason: str, query_signals: Optional[Dict[str, Any]] = None) -> RoutePlan:
    decision = classify_intent(question)
    category = CATEGORY_BY_INTENT.get(decision.intent, IntentCategory.SEMANTIC_EXPLANATORY)
    route_source = "legacy_classifier" if legacy_fallback_enabled() else "planner_rule_fallback"
    return RoutePlan(
        decision=decision,
        intent_category=category,
        route_source=route_source,
        planner_model=model,
        planner_fallback_used=True,
        planner_fallback_reason=fallback_reason,
        planner_raw=None,
        planner_parameters=normalize_planner_parameters(raw_plan={}, question=question, intent=decision.intent, query_signals=query_signals or {}),
    )


def legacy_plan(question: str, model: str, reason: str) -> RoutePlan:
    decision = classify_intent(question)
    category = CATEGORY_BY_INTENT.get(decision.intent, IntentCategory.SEMANTIC_EXPLANATORY)
    return RoutePlan(
        decision=decision,
        intent_category=category,
        route_source="legacy_classifier",
        planner_model=model,
        planner_fallback_used=True,
        planner_fallback_reason=reason,
        planner_raw=None,
        planner_parameters={},
    )
