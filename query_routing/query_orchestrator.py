"""Top-level query orchestration between routing and executors."""

from datetime import datetime, timedelta, timezone
import re
from typing import Any, AsyncIterator, Dict, Optional

try:
    from core_settings import load_settings
    from evidence.evidence_layer import build_repaired_evidence, normalize_evidence
    from executors.db_query_executor import run_db_query, stream_db_query
    from executors.env_query_langchain import (
        answer_env_question_with_metadata,
        stream_answer_env_question,
    )
    from query_routing.intent_classifier import IntentType
    from query_routing.llm_router_planner import (
        AnswerStrategy,
        DecompositionTemplate,
        RoutePlan,
        plan_route,
    )
    from query_routing.query_use_cases import (
        build_clarify_result,
        execute_db_use_case,
        execute_knowledge_use_case,
    )
    from query_routing.observability import (
        get_observability_snapshot,
        record_critic_outcome,
        record_route_plan,
    )
except ImportError:
    from ..core_settings import load_settings
    from ..evidence.evidence_layer import build_repaired_evidence, normalize_evidence
    from ..executors.db_query_executor import run_db_query, stream_db_query
    from ..executors.env_query_langchain import (
        answer_env_question_with_metadata,
        stream_answer_env_question,
    )
    from .intent_classifier import IntentType
    from .llm_router_planner import (
        AnswerStrategy,
        DecompositionTemplate,
        RoutePlan,
        plan_route,
    )
    from .query_use_cases import (
        build_clarify_result,
        execute_db_use_case,
        execute_knowledge_use_case,
    )
    from .observability import (
        get_observability_snapshot,
        record_critic_outcome,
        record_route_plan,
    )


def get_route_plan(question: str, lab_name: Optional[str]) -> RoutePlan:
    route_plan = plan_route(question=question, lab_name=lab_name)
    record_route_plan(route_plan)
    return route_plan


def query_scope_class(route_plan: RoutePlan) -> str:
    signals = (route_plan.planner_parameters or {}).get("query_signals") or {}
    return str(signals.get("query_scope_class") or "").strip().lower()


def build_non_domain_scope_message(question: str) -> str:
    normalized_question = str(question or "").strip()
    return (
        "I focus on Indoor Environmental Quality (IEQ) questions.\n\n"
        f"Your question ({normalized_question}) appears outside that scope. "
        "I can help with IEQ status, trends, comparisons, comfort checks, and recommendations "
        "using metrics like CO2, PM2.5, TVOC, temperature, humidity, noise, light, and IEQ index.\n\n"
        "Try asking something like:\n"
        "- What is today's IEQ in smart_lab?\n"
        "- Is smart_lab comfortable right now?\n"
        "- Compare CO2 in smart_lab vs concrete_lab this week."
    )


def should_use_knowledge_executor(route_plan: RoutePlan) -> bool:
    planner_parameters = route_plan.planner_parameters or {}
    mode = str(planner_parameters.get("response_mode") or "").strip().lower()
    signals = planner_parameters.get("query_signals") or {}
    scope_class = query_scope_class(route_plan)
    if scope_class == "non_domain":
        return True
    if mode != "knowledge_only":
        return False
    # If query asks for scoped/measured facts, keep DB execution even for QA phrasing.
    if bool(signals.get("asks_for_db_facts")):
        return False
    return True


def resolve_execution_intent(intent: IntentType) -> IntentType:
    """Use router task intent directly for DB execution."""
    return intent


def _clarify_threshold() -> float:
    """Read clarify threshold from centralized runtime settings."""
    return load_settings().router_clarify_threshold


def should_clarify(route_plan: RoutePlan, allow_clarify: bool) -> bool:
    if not allow_clarify:
        return False
    if route_plan.answer_strategy == AnswerStrategy.CLARIFY:
        return True
    scope_class = query_scope_class(route_plan)
    confidence = float(route_plan.decision.confidence)
    if scope_class == "ambiguous" and confidence < max(_clarify_threshold(), 0.65):
        return True
    return confidence < _clarify_threshold()


def build_clarify_prompt(route_plan: RoutePlan) -> str:
    intent = route_plan.decision.intent.value
    alternatives = [item.value for item in route_plan.secondary_intents]
    if alternatives:
        return (
            "I can answer this in a few ways. Confirm whether you want "
            f"{intent} or one of: {', '.join(alternatives)}."
        )
    return (
        "Quick clarification: do you want (1) a data-backed answer with exact values from the database, "
        "or (2) a high-level conceptual explanation?"
    )

_METRIC_KEYWORDS = (
    "ieq",
    "co2",
    "pm2.5",
    "pm25",
    "tvoc",
    "temperature",
    "humidity",
    "light",
    "sound",
)
_TIME_PHRASES = (
    "today",
    "yesterday",
    "this week",
    "last week",
    "this month",
    "last month",
)

_MONTH_TOKEN_TO_NUMBER = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}
_MONTH_MENTION_RE = re.compile(
    r"\b(" + "|".join(sorted(_MONTH_TOKEN_TO_NUMBER.keys(), key=len, reverse=True)) + r")\.?"
    r"(?:\s+(\d{1,2})(?:st|nd|rd|th)?)?\b",
    flags=re.IGNORECASE,
)


def _extract_requested_metrics(question: str) -> list[str]:
    q = (question or "").lower()
    metrics = []
    for token in _METRIC_KEYWORDS:
        if token in q and token not in metrics:
            metrics.append(token)
    return metrics


def _requested_time_phrase(question: str) -> Optional[str]:
    q = (question or "").lower()
    for phrase in _TIME_PHRASES:
        if phrase in q:
            return phrase
    m = re.search(r"\b(last|past)\s+\d+\s+(hour|hours|day|days|week|weeks|month|months)\b", q)
    if m:
        return m.group(0)
    return None


def _question_needs_recommendation(question: str) -> bool:
    q = (question or "").lower()
    return any(token in q for token in ("recommend", "advice", "improve", "what should", "next step"))


def _question_needs_explanation(question: str) -> bool:
    q = (question or "").lower()
    return any(token in q for token in ("why", "explain", "reason", "how come"))


def _parse_iso_datetime(raw: Any) -> Optional[datetime]:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _extract_answer_month_mentions(answer: str) -> list[tuple[int, Optional[int]]]:
    mentions: list[tuple[int, Optional[int]]] = []
    for match in _MONTH_MENTION_RE.finditer(str(answer or "")):
        token = str(match.group(1) or "").lower()
        month = _MONTH_TOKEN_TO_NUMBER.get(token)
        if not month:
            continue
        day_raw = match.group(2)
        day: Optional[int] = int(day_raw) if day_raw and day_raw.isdigit() else None
        mentions.append((month, day))
    return mentions


def _expected_month_day_sets(start: datetime, end: datetime) -> tuple[set[int], set[tuple[int, int]]]:
    start_utc = start.astimezone(timezone.utc)
    end_utc = end.astimezone(timezone.utc)
    if end_utc < start_utc:
        start_utc, end_utc = end_utc, start_utc

    adjusted_end = end_utc - timedelta(seconds=1)
    if adjusted_end < start_utc:
        adjusted_end = start_utc

    current = start_utc.date()
    end_date = adjusted_end.date()
    expected_months: set[int] = set()
    expected_pairs: set[tuple[int, int]] = set()
    max_days = 62
    steps = 0
    while current <= end_date and steps < max_days:
        expected_months.add(current.month)
        expected_pairs.add((current.month, current.day))
        current = current + timedelta(days=1)
        steps += 1
    if not expected_months:
        expected_months.add(start_utc.month)
        expected_months.add(end_utc.month)
        expected_pairs.add((start_utc.month, start_utc.day))
        expected_pairs.add((end_utc.month, end_utc.day))
    return expected_months, expected_pairs


def _has_date_consistency_mismatch(answer: str, metadata: Dict[str, Any]) -> bool:
    mentions = _extract_answer_month_mentions(answer)
    if not mentions:
        return False
    time_window = metadata.get("time_window") or {}
    start = _parse_iso_datetime(time_window.get("start"))
    end = _parse_iso_datetime(time_window.get("end"))
    if not start or not end:
        return False
    expected_months, expected_pairs = _expected_month_day_sets(start=start, end=end)
    for month, day in mentions:
        if month not in expected_months:
            return True
        if day is not None and (month, day) not in expected_pairs:
            return True
    return False


def _run_answer_critic(question: str, answer: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    issues: list[str] = []
    answer_text = (answer or "").lower()
    evidence = metadata.get("evidence") or {}
    requested_metrics = _extract_requested_metrics(question)
    if requested_metrics and not any(metric in answer_text for metric in requested_metrics):
        issues.append("missing_metric")

    requested_time = _requested_time_phrase(question)
    time_window_label = str(((metadata.get("time_window") or {}).get("label") or "")).lower()
    if requested_time and requested_time not in time_window_label:
        issues.append("wrong_time_window")
    if _has_date_consistency_mismatch(answer=answer, metadata=metadata):
        issues.append("date_consistency_mismatch")

    if _question_needs_recommendation(question):
        recommendation_markers = ("recommend", "should", "consider", "action", "improve")
        if not any(marker in answer_text for marker in recommendation_markers):
            issues.append("missing_recommendation")

    if _question_needs_explanation(question):
        explanation_markers = ("because", "due to", "likely", "indicates", "explains")
        if not any(marker in answer_text for marker in explanation_markers):
            issues.append("missing_explanation")

    executor = str(metadata.get("executor") or "")
    evidence_kind = str(evidence.get("evidence_kind") or "")
    allowed = {
        "db_query": {"db_query", "decomposed_composition"},
        "decompose": {"decomposed_composition"},
        "knowledge_qa": {"knowledge_qa"},
    }
    if executor in allowed and evidence_kind not in allowed[executor]:
        issues.append("evidence_answer_mismatch")

    status = "pass" if not issues else "warn"
    blocked = False
    if "evidence_answer_mismatch" in issues or "date_consistency_mismatch" in issues:
        status = "block"
        blocked = True
    return {
        "critic_status": status,
        "critic_blocked": blocked,
        "critic_issues": issues,
    }


def _apply_critic(result: Dict[str, Any], question: str) -> Dict[str, Any]:
    metadata = dict(result.get("metadata") or {})
    answer = str(result.get("answer") or "")
    critic = _run_answer_critic(question=question, answer=answer, metadata=metadata)
    record_critic_outcome(str(critic.get("critic_status") or ""))
    metadata.update(critic)
    if critic.get("critic_blocked"):
        issues = set(critic.get("critic_issues") or [])
        if "date_consistency_mismatch" in issues:
            time_window = metadata.get("time_window") or {}
            display_start = str(time_window.get("display_start") or time_window.get("start") or "").strip()
            display_end = str(time_window.get("display_end") or time_window.get("end") or "").strip()
            if display_start and display_end:
                answer = (
                    "I detected a date formatting mismatch and blocked this draft for safety. "
                    f"Verified window: {display_start} to {display_end}."
                )
            else:
                answer = "I detected a date formatting mismatch and blocked this draft for safety."
        else:
            answer = (
                "I need to re-check evidence alignment before giving a definitive answer. "
                "Please rephrase with explicit metric and time scope."
            )
    elif critic.get("critic_issues"):
        issue_list = ", ".join(critic.get("critic_issues") or [])
        answer = f"Note: This answer may be incomplete ({issue_list}).\n\n{answer}"
    result["answer"] = answer
    result["metadata"] = metadata
    return result


def _attach_observability_snapshot(result: Dict[str, Any]) -> Dict[str, Any]:
    metadata = dict(result.get("metadata") or {})
    signals = ((metadata.get("planner_parameters") or {}).get("query_signals") or {}) if isinstance(
        metadata.get("planner_parameters"), dict
    ) else {}
    if "query_scope_class" not in metadata:
        alt_signals = (metadata.get("query_signals") or {}) if isinstance(metadata.get("query_signals"), dict) else {}
        scope_class = str((signals or alt_signals).get("query_scope_class") or "").strip().lower()
        if scope_class:
            metadata["query_scope_class"] = scope_class
    metadata["observability"] = get_observability_snapshot()
    metadata["rollout_slo"] = {
        "planner_fallback_rate_target": 0.05,
        "planner_fallback_rate_max": 0.10,
        "critic_failure_rate_target": 0.02,
        "critic_failure_rate_max": 0.05,
    }
    result["metadata"] = metadata
    return result


def _is_decomposition_allowed(route_plan: RoutePlan) -> bool:
    if route_plan.answer_strategy != AnswerStrategy.DECOMPOSE:
        return False
    if route_plan.decomposition_template is None:
        return False
    # Hard cap: primary + at most one secondary.
    return len(route_plan.secondary_intents) <= 1


def _deterministic_decomposition_merge(
    template: DecompositionTemplate,
    primary_answer: str,
    secondary_answer: str,
) -> str:
    header = {
        DecompositionTemplate.STATE_RECOMMENDATION: "Recommendations",
        DecompositionTemplate.TREND_INTERPRETATION: "Interpretation",
        DecompositionTemplate.ANOMALY_EXPLANATION: "Explanation",
    }[template]
    if not secondary_answer.strip():
        return primary_answer
    return f"{primary_answer}\n\n{header}:\n{secondary_answer.strip()}"


def _build_decomposition_prompt(template: DecompositionTemplate, question: str) -> str:
    if template == DecompositionTemplate.STATE_RECOMMENDATION:
        return f"Provide concise practical recommendations for this state assessment: {question}"
    if template == DecompositionTemplate.TREND_INTERPRETATION:
        return f"Provide concise interpretation for this trend question: {question}"
    return f"Provide concise likely explanations for this anomaly question: {question}"


def _execute_decomposed_query(
    question: str,
    k: int,
    lab_name: Optional[str],
    route_plan: RoutePlan,
) -> Dict:
    decision = route_plan.decision
    execution_intent = resolve_execution_intent(decision.intent)
    template = route_plan.decomposition_template
    if template is None:
        route_plan = RoutePlan(
            decision=route_plan.decision,
            intent_category=route_plan.intent_category,
            route_source=route_plan.route_source,
            planner_model=route_plan.planner_model,
            planner_fallback_used=route_plan.planner_fallback_used,
            planner_fallback_reason=route_plan.planner_fallback_reason,
            planner_raw=route_plan.planner_raw,
            planner_parameters=route_plan.planner_parameters,
            answer_strategy=AnswerStrategy.DIRECT,
            secondary_intents=tuple(),
            decomposition_template=None,
        )
        db_result = run_db_query(
            question=question,
            intent=execution_intent,
            lab_name=lab_name,
            planner_hints=route_plan.planner_parameters,
        )
        evidence = normalize_evidence(
            raw=db_result.get("evidence"),
            executor="db_query",
            lab_name=lab_name,
        )
        return {
            "answer": db_result["answer"],
            "timescale": db_result["timescale"],
            "cards_retrieved": int(db_result.get("cards_retrieved") or 0),
            "recent_card": False,
            "metadata": {
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
                "clarification_required": False,
                "executor": "db_query",
            "query_signals": route_plan.planner_parameters.get("query_signals", {}),
            "query_scope_class": (
                str((route_plan.planner_parameters.get("query_signals") or {}).get("query_scope_class") or "")
                .strip()
                .lower()
                or None
            ),
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
            },
            "data": db_result.get("data"),
            "visualization_type": db_result.get("visualization_type", "none"),
            "chart": db_result.get("chart"),
        }

    primary_result = run_db_query(
        question=question,
        intent=execution_intent,
        lab_name=lab_name,
        planner_hints=route_plan.planner_parameters,
    )
    primary_evidence = normalize_evidence(
        raw=primary_result.get("evidence"),
        executor="db_query",
        lab_name=lab_name,
    )

    secondary_result = answer_env_question_with_metadata(
        user_question=_build_decomposition_prompt(template=template, question=question),
        k=max(1, min(k, 8)),
        space=lab_name,
    )
    secondary_text = str(secondary_result.get("answer") or "")
    secondary_evidence = normalize_evidence(
        raw=secondary_result.get("evidence"),
        executor="knowledge_qa",
        lab_name=lab_name,
    )

    merged_answer = _deterministic_decomposition_merge(
        template=template,
        primary_answer=str(primary_result.get("answer") or ""),
        secondary_answer=secondary_text,
    )

    composed_evidence = normalize_evidence(
        raw={
            "evidence_kind": "decomposed_composition",
            "intent": execution_intent.value,
            "strategy": "decompose",
            "metric_aliases": list(primary_evidence.get("metric_aliases") or []),
            "resolved_scope": primary_evidence.get("resolved_scope") or lab_name,
            "resolved_time_window": primary_evidence.get("resolved_time_window"),
            "provenance_sources": list(primary_evidence.get("provenance_sources") or [])
            + list(secondary_evidence.get("provenance_sources") or []),
            "confidence_notes": list(primary_evidence.get("confidence_notes") or [])
            + list(secondary_evidence.get("confidence_notes") or []),
            "recommendation_allowed": True,
        },
        executor="decomposed_composition",
        lab_name=lab_name,
    )

    metadata = {
        "route_source": route_plan.route_source,
        "route_type": decision.intent.value,
        "intent_category": route_plan.intent_category.value,
        "route_confidence": decision.confidence,
        "route_reason": decision.reason,
        "planner_model": route_plan.planner_model,
        "planner_fallback_used": route_plan.planner_fallback_used,
        "planner_fallback_reason": route_plan.planner_fallback_reason,
        "answer_strategy": route_plan.answer_strategy.value,
        "decomposition_template": template.value,
        "secondary_intents": [item.value for item in route_plan.secondary_intents],
        "decomposition_task_count": 2,
        "clarification_required": False,
        "executor": "decompose",
        "query_signals": route_plan.planner_parameters.get("query_signals", {}),
        "query_scope_class": (
            str((route_plan.planner_parameters.get("query_signals") or {}).get("query_scope_class") or "")
            .strip()
            .lower()
            or None
        ),
        "execution_intent": execution_intent.value,
        "intent_rerouted_to_db": execution_intent != decision.intent,
        "k_requested": k,
        "lab_name": lab_name,
        "resolved_lab_name": primary_result.get("resolved_lab_name"),
        "llm_used": bool(primary_result.get("llm_used", False)),
        "time_window": primary_result.get("time_window"),
        "sources": primary_result.get("sources", []),
        "forecast_model": (primary_result.get("forecast") or {}).get("model"),
        "forecast_confidence": (primary_result.get("forecast") or {}).get("confidence"),
        "forecast_confidence_score": (primary_result.get("forecast") or {}).get("confidence_score"),
        "forecast_horizon_hours": (primary_result.get("forecast") or {}).get("horizon_hours"),
        "correlation": primary_result.get("correlation"),
        "visualization_type": primary_result.get("visualization_type", "none"),
        "evidence": composed_evidence,
    }
    return {
        "answer": merged_answer,
        "timescale": primary_result.get("timescale", "1hour"),
        "cards_retrieved": int(primary_result.get("cards_retrieved") or 0)
        + int(secondary_result.get("cards_retrieved") or 0),
        "recent_card": False,
        "metadata": metadata,
        "data": primary_result.get("data"),
        "visualization_type": primary_result.get("visualization_type", "none"),
        "chart": primary_result.get("chart"),
    }


def execute_query(question: str, k: int, lab_name: Optional[str], allow_clarify: bool = True) -> Dict:
    route_plan = get_route_plan(question=question, lab_name=lab_name)
    decision = route_plan.decision
    if should_clarify(route_plan=route_plan, allow_clarify=allow_clarify):
        clarify_result = build_clarify_result(
            route_plan=route_plan,
            decision=decision,
            k=k,
            lab_name=lab_name,
            clarify_threshold=_clarify_threshold(),
            clarify_text=build_clarify_prompt(route_plan),
        )
        return _attach_observability_snapshot(clarify_result)

    if _is_decomposition_allowed(route_plan):
        decomposed = _execute_decomposed_query(
            question=question,
            k=k,
            lab_name=lab_name,
            route_plan=route_plan,
        )
        return _attach_observability_snapshot(_apply_critic(result=decomposed, question=question))

    if should_use_knowledge_executor(route_plan):
        knowledge_result = execute_knowledge_use_case(
            question=question,
            k=k,
            lab_name=lab_name,
            route_plan=route_plan,
            decision=decision,
            scope_guardrail_builder=build_non_domain_scope_message,
            answer_with_metadata_fn=answer_env_question_with_metadata,
        )
        return _attach_observability_snapshot(_apply_critic(result=knowledge_result, question=question))

    execution_intent = resolve_execution_intent(decision.intent)
    db_result = execute_db_use_case(
        question=question,
        k=k,
        lab_name=lab_name,
        route_plan=route_plan,
        decision=decision,
        execution_intent=execution_intent,
        run_db_query_fn=run_db_query,
    )
    return _attach_observability_snapshot(_apply_critic(result=db_result, question=question))


async def stream_query(question: str, k: int, lab_name: Optional[str]) -> AsyncIterator[str]:
    route_plan = get_route_plan(question=question, lab_name=lab_name)
    decision = route_plan.decision
    if should_use_knowledge_executor(route_plan):
        async for chunk in stream_answer_env_question(
            user_question=question, k=max(1, min(k, 8)), space=lab_name
        ):
            yield chunk
        return

    execution_intent = resolve_execution_intent(decision.intent)

    async for chunk in stream_db_query(
        question=question,
        intent=execution_intent,
        lab_name=lab_name,
        planner_hints=route_plan.planner_parameters,
    ):
        yield chunk

