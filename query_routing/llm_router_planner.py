"""LLM-based single-tool query router with robust JSON guarantees."""

from __future__ import annotations

import json
import random
import re
import time
from typing import Any, Dict, Optional

import requests

try:
    from core_settings import (
        router_base_url,
        router_max_retries,
        router_model,
        router_semantic_rewrite_enabled,
        router_semantic_rewrite_timeout_seconds,
        router_retry_jitter_ms,
        router_temperature,
        router_thinking_enabled,
        router_timeout_seconds,
    )
    from query_routing.intent_classifier import IntentType, RouteDecision
    from query_routing.router_policy import (
        ALLOWED_CARD_TOPICS,
        ALLOWED_PLANNER_METRICS,
        enforce_non_domain_block,
        fallback_plan,
        normalize_plan,
        normalize_planner_parameters,
    )
    from query_routing.router_signals import extract_query_signals
    from query_routing.router_types import (
        AgentAction,
        AnswerStrategy,
        DecompositionTemplate,
        IntentCategory,
        QueryScopeClass,
        RoutePlan,
    )
except ImportError:
    from ..core_settings import (
        router_base_url,
        router_max_retries,
        router_model,
        router_semantic_rewrite_enabled,
        router_semantic_rewrite_timeout_seconds,
        router_retry_jitter_ms,
        router_temperature,
        router_thinking_enabled,
        router_timeout_seconds,
    )
    from .intent_classifier import IntentType, RouteDecision
    from .router_policy import (
        ALLOWED_CARD_TOPICS,
        ALLOWED_PLANNER_METRICS,
        enforce_non_domain_block,
        fallback_plan,
        normalize_plan,
        normalize_planner_parameters,
    )
    from .router_signals import extract_query_signals
    from .router_types import (
        AgentAction,
        AnswerStrategy,
        DecompositionTemplate,
        IntentCategory,
        QueryScopeClass,
        RoutePlan,
    )

_ALLOWED_AGENT_TOOLS = (
    "query_db",
    "search_knowledge_cards",
    "compare_spaces",
    "forecast_metric",
    "analyze_anomaly",
)
_ALLOWED_AGENT_GOALS = ("compare", "explain", "recommend")
_LOW_CONFIDENCE_THRESHOLD = 0.78
_REWRITE_EVIDENCE_THRESHOLD = 0.45
_METRIC_ORDER = ("ieq", "co2", "pm25", "tvoc", "humidity", "temperature", "sound", "light")
_METRIC_PATTERNS = {
    "ieq": re.compile(r"\bieq\b"),
    "co2": re.compile(r"\bco2\b"),
    "pm25": re.compile(r"\b(pm\s*2\.?\s*5|pm2\.?5|pm25)\b"),
    "tvoc": re.compile(r"\b(tvoc|voc)\b"),
    "humidity": re.compile(r"\bhumidity\b"),
    "temperature": re.compile(r"\b(temperature|temp)\b"),
    "sound": re.compile(r"\b(sound|noise)\b"),
    "light": re.compile(r"\b(light|lux)\b"),
}


def _normalize_goal_coverage(raw_plan: Dict[str, Any]) -> tuple[str, ...]:
    raw_goals = raw_plan.get("goal_coverage")
    if not isinstance(raw_goals, list):
        return tuple()
    goals: list[str] = []
    for item in raw_goals:
        goal = str(item or "").strip().lower()
        if goal in _ALLOWED_AGENT_GOALS and goal not in goals:
            goals.append(goal)
    return tuple(goals)


def _build_router_prompt(
    question: str,
    lab_name: Optional[str],
    query_signals: Optional[Dict[str, Any]] = None,
) -> str:
    allowed_intents = ", ".join(sorted([intent.value for intent in IntentType]))
    signals_json = json.dumps(query_signals or {}, ensure_ascii=True)
    return (
        "You route indoor air-quality questions.\n"
        "Focus on the user question first, then use deterministic hints as secondary guidance.\n"
        "Deterministic hints can be noisy and are for confidence calibration, not intent override.\n"
        "Task: classify only. Return ONLY JSON keys: intent_category, intent, confidence, reason.\n"
        "confidence must be between 0 and 1.\n"
        "Do not include markdown or extra text.\n\n"
        f"Allowed intent values: [{allowed_intents}]\n"
        "Allowed intent_category values: [semantic_explanatory, structured_factual_db, analytical_visualization, prediction]\n"
        "Use category-intent pairs that are semantically consistent.\n\n"
        "Routing policy:\n"
        "- if query_scope_class=non_domain, do NOT choose any DB intent.\n"
        "- if query_scope_class=domain, DB intents are allowed.\n"
        "- if query_scope_class=ambiguous, choose best intent and lower confidence when uncertain.\n\n"
        "Routing examples for diagnostic questions (prefer aggregation_db with a full metric pack when measured DB scope is explicit):\n"
        "- \"which metric is driving poor IEQ\" -> aggregation_db, structured_factual_db,\n"
        "  metrics_priority: [co2, pm25, tvoc, humidity, temperature, ieq, sound, light]\n"
        "- \"what is causing the IEQ drop\" -> same\n"
        "- \"why is IEQ low in smart_lab\" -> same\n"
        "- \"what is behind the poor IEQ\" -> same\n"
        "- \"which factor is responsible for IEQ decline\" -> same\n"
        "If diagnostic wording lacks measured DB scope (no concrete lab/time/metric/current-data ask),\n"
        "you may choose definition_explanation with knowledge_only.\n"
        "For scoped diagnostics, include full metric pack.\n\n"
        f"Deterministic query signals: {signals_json}\n"
        f"Question: {question}\n"
        f"Lab hint: {lab_name or ''}\n"
    )


def _calibrate_intent_confidence(
    *,
    raw_plan: Dict[str, Any],
    query_signals: Dict[str, Any],
) -> Dict[str, Any]:
    calibrated = dict(raw_plan or {})
    try:
        confidence = float(calibrated.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    intent = str(calibrated.get("intent") or "").strip().lower()
    metric_strength = float(query_signals.get("metric_signal_strength") or 0.0)
    scope_strength = float(query_signals.get("scope_signal_strength") or 0.0)
    diagnostic_strength = float(query_signals.get("diagnostic_signal_strength") or 0.0)
    asks_db = bool(query_signals.get("asks_for_db_facts"))
    non_domain = str(query_signals.get("query_scope_class") or "").strip().lower() == QueryScopeClass.NON_DOMAIN.value
    db_intents = {i.value for i in IntentType if i not in {IntentType.DEFINITION_EXPLANATION, IntentType.UNKNOWN_FALLBACK}}
    semantic_intents = {IntentType.DEFINITION_EXPLANATION.value, IntentType.UNKNOWN_FALLBACK.value}
    if intent in db_intents:
        confidence += 0.06 * metric_strength + 0.08 * scope_strength + 0.04 * diagnostic_strength
        if asks_db:
            confidence += 0.06
        if non_domain:
            confidence -= 0.25
    elif intent in semantic_intents:
        if non_domain:
            confidence += 0.08
        if asks_db:
            confidence -= 0.1
    calibrated["confidence"] = max(0.0, min(1.0, round(confidence, 4)))
    return calibrated


def _normalize_classification(raw_classification: Dict[str, Any]) -> Dict[str, Any]:
    normalized = {
        "intent_category": raw_classification.get("intent_category"),
        "intent": raw_classification.get("intent"),
        "confidence": raw_classification.get("confidence"),
        "reason": raw_classification.get("reason"),
    }
    return normalized


def _rank_relevant_metrics(
    *,
    question: str,
    query_signals: Dict[str, Any],
    is_diagnostic: bool,
) -> list[str]:
    q = str(question or "").lower()
    mentioned: list[str] = []
    for metric in _METRIC_ORDER:
        pattern = _METRIC_PATTERNS.get(metric)
        if pattern is not None and pattern.search(q):
            mentioned.append(metric)
    scored = list(mentioned)
    scope_strength = float(query_signals.get("scope_signal_strength") or 0.0)
    metric_strength = float(query_signals.get("metric_signal_strength") or 0.0)
    if not scored and metric_strength > 0.5:
        scored.extend(["ieq", "co2"])
    if is_diagnostic:
        defaults = ["co2", "pm25", "tvoc", "humidity", "temperature"]
        if scope_strength < 0.35:
            defaults = defaults[:3]
        for metric in defaults:
            if metric not in scored:
                scored.append(metric)
    elif not scored:
        scored.extend(["ieq", "co2"])
    # Include IEQ as a default anchor only when no explicit metric was asked.
    if not mentioned and "ieq" not in scored:
        scored.insert(0, "ieq")
    # Keep metrics adaptive and compact to avoid over-fetching.
    return scored[:5]


def _build_plan_defaults_from_classification(
    *,
    raw_classification: Dict[str, Any],
    question: str,
    query_signals: Dict[str, Any],
) -> Dict[str, Any]:
    intent = str(raw_classification.get("intent") or "").strip().lower()
    scope = str(query_signals.get("query_scope_class") or "").strip().lower()
    asks_db = bool(query_signals.get("asks_for_db_facts"))
    is_general_knowledge = bool(query_signals.get("is_general_knowledge_question"))
    is_diagnostic = bool(query_signals.get("is_diagnostic_phrase"))
    if scope == QueryScopeClass.NON_DOMAIN.value:
        response_mode = "knowledge_only"
    elif asks_db:
        response_mode = "db"
    elif intent in {IntentType.DEFINITION_EXPLANATION.value, IntentType.UNKNOWN_FALLBACK.value} and is_general_knowledge:
        response_mode = "knowledge_only"
    else:
        response_mode = "db"
    if response_mode == "db":
        metrics_priority = _rank_relevant_metrics(
            question=question,
            query_signals=query_signals,
            is_diagnostic=is_diagnostic,
        )
    else:
        metrics_priority = ["ieq"]
    return {
        "strategy": "direct",
        "action": "finalize",
        "response_mode": response_mode,
        "needs_cards": response_mode == "knowledge_only",
        "card_topics": ["definitions", "metric_explanations"] if response_mode == "knowledge_only" else ["metric_explanations"],
        "max_cards": 2,
        "metrics_priority": metrics_priority,
    }


def _planning_needed(
    *,
    classification: Dict[str, Any],
    query_signals: Dict[str, Any],
) -> bool:
    confidence = float(classification.get("confidence") or 0.0)
    scope = str(query_signals.get("query_scope_class") or "").strip().lower()
    db_scoped = scope == QueryScopeClass.DOMAIN.value or bool(query_signals.get("asks_for_db_facts"))
    return confidence < _LOW_CONFIDENCE_THRESHOLD or (db_scoped and confidence < 0.9)


def _merge_planning_fields(
    *,
    base_plan: Dict[str, Any],
    planning_fields: Dict[str, Any],
    query_signals: Dict[str, Any],
    calibrated_confidence: float,
) -> Dict[str, Any]:
    merged = dict(base_plan)
    if not planning_fields:
        return merged
    # Deterministic defaults are primary. Planning LLM can only fill uncertain
    # or unset details.
    scope_class = str(query_signals.get("query_scope_class") or "").strip().lower()
    db_scoped = scope_class == QueryScopeClass.DOMAIN.value or bool(query_signals.get("asks_for_db_facts"))
    uncertain_context = (
        calibrated_confidence < _LOW_CONFIDENCE_THRESHOLD
        or scope_class == QueryScopeClass.AMBIGUOUS.value
        or db_scoped
    )
    fill_only_keys = {
        "secondary_intents",
        "tool_name",
        "tool_arguments",
        "expected_observation",
        "enough_evidence",
        "goal_coverage",
        "metrics_priority",
        "response_mode",
        "needs_cards",
        "card_topics",
        "max_cards",
    }
    uncertain_override_keys = {
        "strategy",
        "action",
    }
    for key, value in planning_fields.items():
        if value is None:
            continue
        if key in fill_only_keys and not merged.get(key):
            merged[key] = value
        elif key in uncertain_override_keys and uncertain_context:
            merged[key] = value
    return merged


def _extract_planning_fields(raw_details: Dict[str, Any]) -> Dict[str, Any]:
    allowed = {
        "strategy",
        "secondary_intents",
        "action",
        "tool_name",
        "tool_arguments",
        "expected_observation",
        "enough_evidence",
        "goal_coverage",
        "metrics_priority",
        "response_mode",
        "needs_cards",
        "card_topics",
        "max_cards",
    }
    return {key: raw_details.get(key) for key in allowed if key in raw_details}


def _build_repair_prompt(
    question: str,
    lab_name: Optional[str],
    query_signals: Optional[Dict[str, Any]],
    validation_error: str,
    prior_response: str,
) -> str:
    return (
        "Your previous JSON output failed validation.\n"
        f"Validation error: {validation_error}\n"
        "Re-answer with ONLY valid JSON and no extra text.\n"
        "Required keys: intent_category, intent, confidence.\n"
        "Optional keys: reason, strategy, secondary_intents, action, tool_name, tool_arguments, expected_observation, enough_evidence, goal_coverage.\n\n"
        f"Previous output: {prior_response}\n\n"
        + _build_router_prompt(
            question=question,
            lab_name=lab_name,
            query_signals=query_signals,
        )
    )


def _build_planning_prompt(
    *,
    question: str,
    lab_name: Optional[str],
    query_signals: Optional[Dict[str, Any]],
    classification: Dict[str, Any],
) -> str:
    classification_json = json.dumps(classification or {}, ensure_ascii=True)
    signals_json = json.dumps(query_signals or {}, ensure_ascii=True)
    return (
        "You are in planning step after classification is already decided.\n"
        "Do not change classification. Keep intent_category, intent, confidence, reason from the provided classification.\n"
        "Return ONLY JSON with required keys: intent_category, intent, confidence.\n"
        "You may add optional planning keys only: strategy, secondary_intents, action, tool_name, tool_arguments, expected_observation, enough_evidence, goal_coverage, metrics_priority, response_mode, needs_cards, card_topics, max_cards.\n"
        "Prefer concise plans. Use clarify only when scope is genuinely underspecified.\n\n"
        f"Classification (fixed): {classification_json}\n"
        f"Deterministic query signals: {signals_json}\n"
        f"Question: {question}\n"
        f"Lab hint: {lab_name or ''}\n"
    )


def _extract_json_object(raw_text: str) -> Dict[str, Any]:
    text = str(raw_text or "").strip()
    if not text:
        raise ValueError("empty_response")
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)

    fenced = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not fenced:
        raise ValueError("missing_json_object")
    return json.loads(fenced.group(0))


def _call_router_planner(
    question: str,
    lab_name: Optional[str],
    query_signals: Optional[Dict[str, Any]] = None,
    prompt_override: Optional[str] = None,
) -> Dict[str, Any]:
    schema: Dict[str, Any] = {
        "type": "object",
        "required": ["intent_category", "intent", "confidence"],
        "properties": {
            "intent_category": {
                "type": "string",
                "enum": [category.value for category in IntentCategory],
            },
            "intent": {
                "type": "string",
                "enum": [intent.value for intent in IntentType],
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reason": {"type": "string"},
            "strategy": {
                "type": "string",
                "enum": [strategy.value for strategy in AnswerStrategy],
            },
            "action": {
                "type": "string",
                "enum": [action.value for action in AgentAction],
            },
            "secondary_intents": {
                "type": "array",
                "maxItems": 2,
                "items": {"type": "string", "enum": [intent.value for intent in IntentType]},
            },
            "tool_name": {
                "type": "string",
                "enum": list(_ALLOWED_AGENT_TOOLS),
            },
            "tool_arguments": {"type": "object"},
            "expected_observation": {"type": "string"},
            "enough_evidence": {"type": "boolean"},
            "goal_coverage": {
                "type": "array",
                "maxItems": 3,
                "items": {"type": "string", "enum": list(_ALLOWED_AGENT_GOALS)},
            },
            "metrics_priority": {
                "type": "array",
                "minItems": 1,
                "maxItems": 7,
                "items": {"type": "string", "enum": sorted(list(ALLOWED_PLANNER_METRICS))},
            },
            "response_mode": {"type": "string", "enum": ["db", "knowledge_only"]},
            "needs_cards": {"type": "boolean"},
            "card_topics": {
                "type": "array",
                "items": {"type": "string", "enum": sorted(list(ALLOWED_CARD_TOPICS))},
                "maxItems": 5,
            },
            "max_cards": {"type": "integer", "minimum": 1, "maximum": 4},
        },
        "additionalProperties": False,
    }
    payload: Dict[str, Any] = {
        "model": router_model(),
        "prompt": prompt_override
        or _build_router_prompt(
            question=question,
            lab_name=lab_name,
            query_signals=query_signals,
        ),
        "stream": False,
        "format": schema,
        "options": {"temperature": router_temperature()},
    }
    if not router_thinking_enabled():
        payload["think"] = False

    response = requests.post(
        f"{router_base_url()}/api/generate",
        json=payload,
        timeout=router_timeout_seconds(),
    )
    response.raise_for_status()
    body = response.json()
    raw_text = str(body.get("response") or "")
    if not raw_text.strip():
        raise ValueError("empty_planner_response")
    return _extract_json_object(raw_text)


def _call_semantic_rewrite(
    *,
    question: str,
    lab_name: Optional[str],
    query_signals: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    schema: Dict[str, Any] = {
        "type": "object",
        "required": ["rewritten_question", "changed"],
        "properties": {
            "rewritten_question": {"type": "string"},
            "changed": {"type": "boolean"},
            "reason": {"type": "string"},
        },
        "additionalProperties": False,
    }
    prompt = (
        "Rewrite the user question into a concise canonical form for routing.\n"
        "Preserve user intent exactly. Do not add facts, assumptions, or missing scope.\n"
        "If question is already clear, return it unchanged with changed=false.\n"
        "Return ONLY JSON with keys: rewritten_question, changed, reason.\n\n"
        f"Deterministic query signals: {json.dumps(query_signals or {}, ensure_ascii=True)}\n"
        f"Question: {question}\n"
        f"Lab hint: {lab_name or ''}\n"
    )
    payload: Dict[str, Any] = {
        "model": router_model(),
        "prompt": prompt,
        "stream": False,
        "format": schema,
        "options": {"temperature": 0.0},
        "think": False,
    }
    response = requests.post(
        f"{router_base_url()}/api/generate",
        json=payload,
        timeout=min(router_timeout_seconds(), router_semantic_rewrite_timeout_seconds()),
    )
    response.raise_for_status()
    body = response.json()
    raw_text = str(body.get("response") or "")
    if not raw_text.strip():
        raise ValueError("empty_rewrite_response")
    rewritten = _extract_json_object(raw_text)
    rewritten_q = str(rewritten.get("rewritten_question") or "").strip()
    if not rewritten_q:
        raise ValueError("invalid_rewrite_question")
    return {
        "rewritten_question": rewritten_q,
        "changed": bool(rewritten.get("changed")),
        "reason": str(rewritten.get("reason") or "").strip(),
    }


def _should_semantic_rewrite(question: str, query_signals: Dict[str, Any]) -> bool:
    if not router_semantic_rewrite_enabled():
        return False
    scope_class = str(query_signals.get("query_scope_class") or "").strip().lower()
    if scope_class != QueryScopeClass.AMBIGUOUS.value:
        return False
    asks_for_db_facts = bool(query_signals.get("asks_for_db_facts"))
    is_general_knowledge = bool(query_signals.get("is_general_knowledge_question"))
    metric_strength = float(query_signals.get("metric_signal_strength") or 0.0)
    scope_strength = float(query_signals.get("scope_signal_strength") or 0.0)
    diagnostic_strength = float(query_signals.get("diagnostic_signal_strength") or 0.0)
    q = str(question or "").lower()
    mixed_lexical_intents = (
        re.search(r"\b(compare|comparison|vs|versus)\b", q) is not None
        and re.search(r"\b(explain|meaning|what does|definition|interpret)\b", q) is not None
    )
    conflicting_scope = asks_for_db_facts and is_general_knowledge
    evidence_score = max(metric_strength, scope_strength, diagnostic_strength)
    return (mixed_lexical_intents or conflicting_scope) and evidence_score < _REWRITE_EVIDENCE_THRESHOLD


def _normalize_fallback_reason(exc: Optional[Exception]) -> str:
    if exc is None:
        return "planner_error:unknown"
    name = type(exc).__name__.lower()
    detail = str(exc or "").lower()
    if "timeout" in name or "timeout" in detail:
        kind = "timeout"
    elif "connection" in name or "http" in name or "transport" in detail:
        kind = "transport"
    elif "empty" in detail:
        kind = "empty_response"
    elif "json" in name or "json" in detail or "missing_json_object" in detail:
        kind = "parse_error"
    elif "invalid_" in detail or "mismatch" in detail:
        kind = "normalization_error"
    else:
        kind = "schema_error"
    return f"planner_error:{kind}:{type(exc).__name__}"


def _build_success_plan(
    *,
    model: str,
    raw_plan: Dict[str, Any],
    question: str,
    query_signals: Dict[str, Any],
) -> RoutePlan:
    category, decision, strategy, secondary_intents, template = normalize_plan(raw_plan)
    decision, category, strategy, secondary_intents, template = enforce_non_domain_block(
        decision=decision,
        category=category,
        strategy=strategy,
        secondary_intents=secondary_intents,
        template=template,
        query_signals=query_signals,
    )
    planner_parameters = normalize_planner_parameters(
        raw_plan=raw_plan,
        question=question,
        intent=decision.intent,
        query_signals=query_signals,
    )
    raw_action = str(raw_plan.get("action") or "").strip().lower()
    try:
        agent_action = AgentAction(raw_action) if raw_action else AgentAction.FINALIZE
    except ValueError:
        agent_action = AgentAction.FINALIZE
    if strategy == AnswerStrategy.CLARIFY:
        agent_action = AgentAction.CLARIFY
    tool_name = str(raw_plan.get("tool_name") or "").strip().lower() or None
    if agent_action == AgentAction.TOOL_CALL and tool_name not in _ALLOWED_AGENT_TOOLS:
        agent_action = AgentAction.FINALIZE
        tool_name = None
    tool_arguments = raw_plan.get("tool_arguments")
    if not isinstance(tool_arguments, dict):
        tool_arguments = {}
    expected_observation = str(raw_plan.get("expected_observation") or "").strip() or None
    enough_evidence_raw = raw_plan.get("enough_evidence")
    enough_evidence = bool(enough_evidence_raw) if isinstance(enough_evidence_raw, bool) else None
    goal_coverage = _normalize_goal_coverage(raw_plan)
    if template:
        planner_parameters["decomposition_template"] = template.value
    return RoutePlan(
        decision=decision,
        intent_category=category,
        route_source="llm_planner",
        planner_model=model,
        planner_fallback_used=False,
        planner_fallback_reason=None,
        planner_raw=raw_plan,
        planner_parameters=planner_parameters,
        answer_strategy=strategy,
        secondary_intents=secondary_intents,
        decomposition_template=template,
        agent_action=agent_action,
        tool_name=tool_name,
        tool_arguments=tool_arguments,
        expected_observation=expected_observation,
        enough_evidence=enough_evidence,
        goal_coverage=goal_coverage,
    )


def _should_fastpath_knowledge_route(query_signals: Dict[str, Any]) -> bool:
    scope_class = str(query_signals.get("query_scope_class") or "").strip().lower()
    if scope_class == QueryScopeClass.NON_DOMAIN.value:
        return True
    if bool(query_signals.get("is_hypothetical_conditional")) and not bool(
        query_signals.get("requests_current_measured_data")
    ):
        return True
    # Conceptual/general-knowledge asks without DB-facts requirement do not need
    # an extra planner LLM round-trip.
    if bool(query_signals.get("is_general_knowledge_question")) and not bool(query_signals.get("asks_for_db_facts")):
        return True
    return False


def _build_fastpath_knowledge_plan(
    *,
    model: str,
    question: str,
    query_signals: Dict[str, Any],
) -> RoutePlan:
    decision = RouteDecision(
        intent=IntentType.DEFINITION_EXPLANATION,
        confidence=0.9 if str(query_signals.get("query_scope_class") or "") == QueryScopeClass.NON_DOMAIN.value else 0.85,
        reason="signal_fastpath_knowledge",
    )
    planner_parameters = normalize_planner_parameters(
        raw_plan={
            "response_mode": "knowledge_only",
            "needs_cards": True,
            "card_topics": ["definitions", "metric_explanations"],
            "max_cards": 4,
        },
        question=question,
        intent=decision.intent,
        query_signals=query_signals,
    )
    return RoutePlan(
        decision=decision,
        intent_category=IntentCategory.SEMANTIC_EXPLANATORY,
        route_source="signal_fastpath",
        planner_model=model,
        planner_fallback_used=False,
        planner_fallback_reason=None,
        planner_raw={
            "intent_category": IntentCategory.SEMANTIC_EXPLANATORY.value,
            "intent": decision.intent.value,
            "confidence": decision.confidence,
            "reason": decision.reason,
            "response_mode": "knowledge_only",
            "needs_cards": True,
            "card_topics": ["definitions", "metric_explanations"],
            "max_cards": 4,
        },
        planner_parameters=planner_parameters,
        answer_strategy=AnswerStrategy.DIRECT,
        secondary_intents=tuple(),
        decomposition_template=None,
    )


def plan_route(question: str, lab_name: Optional[str] = None) -> RoutePlan:
    model = router_model()
    effective_question = question
    query_signals = extract_query_signals(question=effective_question, lab_name=lab_name)
    rewrite_metadata: Dict[str, Any] = {"enabled": router_semantic_rewrite_enabled(), "attempted": False, "applied": False}
    if _should_semantic_rewrite(question=effective_question, query_signals=query_signals):
        rewrite_metadata["attempted"] = True
        try:
            rewritten = _call_semantic_rewrite(
                question=effective_question,
                lab_name=lab_name,
                query_signals=query_signals,
            )
            candidate_question = str(rewritten.get("rewritten_question") or "").strip()
            rewrite_metadata["reason"] = str(rewritten.get("reason") or "").strip()
            if candidate_question and candidate_question.lower() != effective_question.strip().lower():
                effective_question = candidate_question
                query_signals = extract_query_signals(question=effective_question, lab_name=lab_name)
                rewrite_metadata["applied"] = True
                rewrite_metadata["rewritten_question"] = effective_question
        except Exception as exc:
            rewrite_metadata["error"] = type(exc).__name__
    if _should_fastpath_knowledge_route(query_signals):
        plan = _build_fastpath_knowledge_plan(
            model=model,
            question=effective_question,
            query_signals=query_signals,
        )
        plan.planner_parameters["semantic_rewrite"] = rewrite_metadata
        return plan
    last_error: Optional[Exception] = None
    for _ in range(router_max_retries()):
        try:
            raw_classification = _call_router_planner(
                question=effective_question,
                lab_name=lab_name,
                query_signals=query_signals,
            )
            normalized_classification = _normalize_classification(raw_classification)
            calibrated_classification = _calibrate_intent_confidence(
                raw_plan=normalized_classification,
                query_signals=query_signals,
            )
            deterministic_plan = _build_plan_defaults_from_classification(
                raw_classification=calibrated_classification,
                question=effective_question,
                query_signals=query_signals,
            )
            planning_fields: Dict[str, Any] = {}
            if _planning_needed(
                classification=normalized_classification,
                query_signals=query_signals,
            ):
                try:
                    planning_prompt = _build_planning_prompt(
                        question=effective_question,
                        lab_name=lab_name,
                        query_signals=query_signals,
                        classification=calibrated_classification,
                    )
                    raw_details = _call_router_planner(
                        question=effective_question,
                        lab_name=lab_name,
                        query_signals=query_signals,
                        prompt_override=planning_prompt,
                    )
                    planning_fields = _extract_planning_fields(raw_details)
                except Exception as plan_exc:
                    rewrite_metadata["planning_step_error"] = type(plan_exc).__name__
            raw_plan = dict(calibrated_classification)
            for key, value in deterministic_plan.items():
                raw_plan[key] = value
            raw_plan = _merge_planning_fields(
                base_plan=raw_plan,
                planning_fields=planning_fields,
                query_signals=query_signals,
                calibrated_confidence=float(calibrated_classification.get("confidence") or 0.0),
            )
            plan = _build_success_plan(
                model=model,
                raw_plan=raw_plan,
                question=effective_question,
                query_signals=query_signals,
            )
            plan.planner_parameters["semantic_rewrite"] = rewrite_metadata
            return plan
        except Exception as exc:
            last_error = exc
            should_repair = any(
                marker in str(exc).lower()
                for marker in (
                    "missing_json_object",
                    "empty_planner_response",
                    "json",
                    "invalid_intent",
                    "invalid_intent_category",
                    "intent_category_mismatch",
                )
            )
            if should_repair:
                try:
                    repaired_prompt = _build_repair_prompt(
                        question=effective_question,
                        lab_name=lab_name,
                        query_signals=query_signals,
                        validation_error=str(exc),
                        prior_response="invalid_or_unparseable",
                    )
                    repaired_plan = _call_router_planner(
                        question=effective_question,
                        lab_name=lab_name,
                        query_signals=query_signals,
                        prompt_override=repaired_prompt,
                    )
                    repaired_plan = _normalize_classification(repaired_plan)
                    repaired_plan = _calibrate_intent_confidence(
                        raw_plan=repaired_plan,
                        query_signals=query_signals,
                    )
                    repaired_defaults = _build_plan_defaults_from_classification(
                        raw_classification=repaired_plan,
                        question=effective_question,
                        query_signals=query_signals,
                    )
                    repaired_planning_fields: Dict[str, Any] = {}
                    if _planning_needed(
                        classification=repaired_plan,
                        query_signals=query_signals,
                    ):
                        try:
                            repaired_planning_prompt = _build_planning_prompt(
                                question=effective_question,
                                lab_name=lab_name,
                                query_signals=query_signals,
                                classification=repaired_plan,
                            )
                            repaired_details = _call_router_planner(
                                question=effective_question,
                                lab_name=lab_name,
                                query_signals=query_signals,
                                prompt_override=repaired_planning_prompt,
                            )
                            repaired_planning_fields = _extract_planning_fields(repaired_details)
                        except Exception as repaired_plan_exc:
                            rewrite_metadata["planning_step_error"] = type(repaired_plan_exc).__name__
                    merged_repaired_plan = dict(repaired_plan)
                    for key, value in repaired_defaults.items():
                        merged_repaired_plan[key] = value
                    merged_repaired_plan = _merge_planning_fields(
                        base_plan=merged_repaired_plan,
                        planning_fields=repaired_planning_fields,
                        query_signals=query_signals,
                        calibrated_confidence=float(repaired_plan.get("confidence") or 0.0),
                    )
                    plan = _build_success_plan(
                        model=model,
                        raw_plan=merged_repaired_plan,
                        question=effective_question,
                        query_signals=query_signals,
                    )
                    plan.planner_parameters["semantic_rewrite"] = rewrite_metadata
                    return plan
                except Exception as repair_exc:
                    last_error = repair_exc

            jitter_ms = router_retry_jitter_ms()
            if jitter_ms > 0:
                time.sleep(random.uniform(0, jitter_ms / 1000.0))

    plan = fallback_plan(
        question=effective_question,
        model=model,
        fallback_reason=_normalize_fallback_reason(last_error),
        query_signals=query_signals,
    )
    plan.planner_parameters["semantic_rewrite"] = rewrite_metadata
    return plan
