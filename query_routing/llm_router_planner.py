"""LLM-based single-tool query router with robust JSON guarantees."""

from __future__ import annotations

import json
import random
import re
import time
from typing import Any, Dict, Optional

import requests

try:
    from query_routing.intent_classifier import IntentType, RouteDecision
    from query_routing.router_policy import (
        ALLOWED_CARD_TOPICS,
        ALLOWED_PLANNER_METRICS,
        enforce_non_domain_block,
        fallback_plan,
        legacy_plan,
        normalize_plan,
        normalize_planner_parameters,
    )
    from query_routing.router_settings import (
        router_base_url,
        router_max_retries,
        router_mode,
        router_model,
        router_retry_jitter_ms,
        router_temperature,
        router_thinking_enabled,
        router_timeout_seconds,
    )
    from query_routing.router_signals import extract_query_signals
    from query_routing.router_types import (
        AnswerStrategy,
        DecompositionTemplate,
        IntentCategory,
        QueryScopeClass,
        RoutePlan,
    )
except ImportError:
    from .intent_classifier import IntentType, RouteDecision
    from .router_policy import (
        ALLOWED_CARD_TOPICS,
        ALLOWED_PLANNER_METRICS,
        enforce_non_domain_block,
        fallback_plan,
        legacy_plan,
        normalize_plan,
        normalize_planner_parameters,
    )
    from .router_settings import (
        router_base_url,
        router_max_retries,
        router_mode,
        router_model,
        router_retry_jitter_ms,
        router_temperature,
        router_thinking_enabled,
        router_timeout_seconds,
    )
    from .router_signals import extract_query_signals
    from .router_types import (
        AnswerStrategy,
        DecompositionTemplate,
        IntentCategory,
        QueryScopeClass,
        RoutePlan,
    )


def _build_router_prompt(
    question: str, lab_name: Optional[str], query_signals: Optional[Dict[str, Any]] = None
) -> str:
    allowed_intents = ", ".join(sorted([intent.value for intent in IntentType]))
    signals_json = json.dumps(query_signals or {}, ensure_ascii=True)
    return (
        "You route indoor air-quality questions.\n"
        "Focus on the user question first, then use deterministic hints as secondary guidance.\n"
        "Deterministic hints can be noisy and must never override clear user intent.\n"
        "Return ONLY JSON with at least: intent_category, intent, confidence.\n"
        "Optional fields: reason, strategy, secondary_intents.\n"
        "strategy values: [direct, decompose, clarify].\n"
        "confidence must be between 0 and 1.\n"
        "Do not include markdown or extra text.\n\n"
        f"Allowed intent values: [{allowed_intents}]\n"
        "Allowed intent_category values: [semantic_explanatory, structured_factual_db, analytical_visualization, prediction]\n"
        "Use category-intent pairs that are semantically consistent.\n\n"
        "Routing policy:\n"
        "- if query_scope_class=non_domain, do NOT choose any DB intent.\n"
        "- if query_scope_class=domain, DB intents are allowed.\n"
        "- if query_scope_class=ambiguous, choose best intent and lower confidence when uncertain.\n\n"
        f"Deterministic query signals: {signals_json}\n"
        f"Question: {question}\n"
        f"Lab hint: {lab_name or ''}\n"
    )


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
        "Optional keys: reason, strategy, secondary_intents.\n\n"
        f"Previous output: {prior_response}\n\n"
        + _build_router_prompt(question=question, lab_name=lab_name, query_signals=query_signals)
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
            "secondary_intents": {
                "type": "array",
                "maxItems": 2,
                "items": {"type": "string", "enum": [intent.value for intent in IntentType]},
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
        "prompt": prompt_override or _build_router_prompt(question=question, lab_name=lab_name, query_signals=query_signals),
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
    )


def _should_fastpath_knowledge_route(query_signals: Dict[str, Any]) -> bool:
    scope_class = str(query_signals.get("query_scope_class") or "").strip().lower()
    if scope_class == QueryScopeClass.NON_DOMAIN.value:
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
    query_signals = extract_query_signals(question=question, lab_name=lab_name)
    if _should_fastpath_knowledge_route(query_signals):
        return _build_fastpath_knowledge_plan(
            model=model,
            question=question,
            query_signals=query_signals,
        )
    if router_mode() == "legacy":
        return legacy_plan(question=question, model=model, reason="router_mode_legacy")

    last_error: Optional[Exception] = None
    for _ in range(router_max_retries()):
        try:
            raw_plan = _call_router_planner(question=question, lab_name=lab_name, query_signals=query_signals)
            return _build_success_plan(
                model=model,
                raw_plan=raw_plan,
                question=question,
                query_signals=query_signals,
            )
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
                        question=question,
                        lab_name=lab_name,
                        query_signals=query_signals,
                        validation_error=str(exc),
                        prior_response="invalid_or_unparseable",
                    )
                    repaired_plan = _call_router_planner(
                        question=question,
                        lab_name=lab_name,
                        query_signals=query_signals,
                        prompt_override=repaired_prompt,
                    )
                    return _build_success_plan(
                        model=model,
                        raw_plan=repaired_plan,
                        question=question,
                        query_signals=query_signals,
                    )
                except Exception as repair_exc:
                    last_error = repair_exc

            jitter_ms = router_retry_jitter_ms()
            if jitter_ms > 0:
                time.sleep(random.uniform(0, jitter_ms / 1000.0))

    return fallback_plan(
        question=question,
        model=model,
        fallback_reason=_normalize_fallback_reason(last_error),
        query_signals=query_signals,
    )
