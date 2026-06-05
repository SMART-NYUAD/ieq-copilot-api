"""LLM-based query router: single call, structured JSON output."""

from __future__ import annotations

import asyncio
import json
import random
import re
import time
from typing import List, Optional

import httpx
import requests

try:
    from core_settings import (
        router_base_url,
        router_max_retries,
        router_model,
        router_retry_jitter_ms,
        router_temperature,
        router_thinking_enabled,
        router_timeout_seconds,
    )
    from query_routing.intent_classifier import IntentType
    from query_routing.router_types import RoutePlan
except ImportError:
    from ..core_settings import (
        router_base_url,
        router_max_retries,
        router_model,
        router_retry_jitter_ms,
        router_temperature,
        router_thinking_enabled,
        router_timeout_seconds,
    )
    from .intent_classifier import IntentType
    from .router_types import RoutePlan


_INTENT_VALUES = {i.value for i in IntentType} - {IntentType.UNKNOWN_FALLBACK.value}
_METRIC_RE = re.compile(r"\b(co2|pm\s*2\.?\s*5|pm25|tvoc|voc|temperature|temp|humidity|light|lux|sound|noise|ieq)\b")
_METRIC_CANONICAL = {
    "pm 2.5": "pm25", "pm2.5": "pm25", "pm 25": "pm25",
    "voc": "tvoc", "temp": "temperature", "lux": "light", "noise": "sound",
}

_SYSTEM_PROMPT = (
    "You are an indoor air quality query router for a facility management system.\n"
    "Given a user question and optional lab hint, output ONLY a JSON object with these fields:\n"
    '  "intent": one of [definition_explanation, current_status_db, point_lookup_db, '
    "aggregation_db, comparison_db, anomaly_analysis_db]\n"
    '  "lab": the lab/space name if mentioned, else null\n'
    '  "second_lab": always null\n'
    '  "metrics": list of relevant metrics from [co2, pm25, tvoc, humidity, temperature, light, sound, ieq]\n'
    '  "time_phrase": exact time window phrase from question (e.g. "last 24 hours"), else null\n'
    '  "confidence": float 0-1\n\n'
    "Routing rules:\n"
    "- When Prior conversation is present, use it only to fill missing lab/time slots. "
    "The current Question sets metric/topic scope; do not route to a prior-turn metric when "
    "the user asked a different scope (e.g. 'air quality' after a temperature question → IEQ/IAQ, not temperature).\n\n"
    "Intent definitions:\n"
    "- definition_explanation: ONLY for conceptual/educational questions asking what a metric means, "
    "with no definite article before the metric name. "
    "Examples: 'what is CO2?', 'what does IEQ mean?', 'explain TVOC', 'define humidity', 'what is pm2.5?'.\n"
    "- current_status_db: The user wants the live/latest sensor reading right now, with no specific past "
    "timestamp. Covers informal, typo-variant, and assessment phrasing. The definite article 'the' before a "
    "metric always signals a value request, not a definition. Also use for opinion/assessment questions about "
    "a metric that require knowing the current value to answer. "
    "Examples: 'what is the CO2?', 'what\\'s the VOC?', 'whats the voc', 'whats co2', "
    "'show me the temperature', 'how is the humidity?', 'co2?', 'voc levels?', 'current readings', "
    "'is the CO2 ok?', 'is humidity too high?', 'do you think co2 is fine?', "
    "'should I worry about PM2.5?', 'is the air safe?'. "
    "Use this when in doubt — never fabricate sensor values.\n"
    "- point_lookup_db: The user asks for a reading at a specific past moment (not a range or average). "
    "Examples: 'what was the CO2 at 3pm?', 'what was the temperature at 9am this morning?', "
    "'what did humidity read an hour ago?'.\n"
    "- aggregation_db: Questions about trends, averages, or summaries across a time window. "
    "Examples: 'average CO2 last 24 hours', 'how has humidity trended this week?', "
    "'daily temperature summary', 'CO2 over the past 7 days'.\n"
    "- comparison_db: Comparing two or more metrics in the same lab, or the same metric across two time periods. "
    "Examples: 'compare CO2 and humidity', 'humidity vs CO2', 'is it better than yesterday?', "
    "'today vs last week'. Never use for cross-space comparisons.\n"
    "- anomaly_analysis_db: Detecting or explaining unusual readings, spikes, or outliers. "
    "Examples: 'why did CO2 spike?', 'any unusual readings today?', 'were there anomalies last night?', "
    "'what caused that PM2.5 outlier?'.\n\n"
    "Output only the JSON object, no markdown, no explanation."
)


def _extract_metrics_from_question(question: str) -> List[str]:
    q = question.lower()
    found = []
    for raw in _METRIC_RE.findall(q):
        canonical = _METRIC_CANONICAL.get(raw.strip(), raw.strip().replace(" ", ""))
        if canonical not in found:
            found.append(canonical)
    if not found:
        found = ["ieq", "co2"]
    return found


_FALLBACK_COMPARISON_RE = re.compile(r"\b(compare|comparison|versus|vs\.?|between)\b")
_FALLBACK_ANOMALY_RE = re.compile(r"\b(anomal|spike|outlier|unusual|abnormal|deviation)\b")
_FALLBACK_AGGREGATION_RE = re.compile(r"\b(trend|average|avg|mean|sum|over\s+the|last\s+\d|past\s+\d|history|historical|weekly|daily)\b")


def _fallback_plan(question: str, lab_name: Optional[str]) -> RoutePlan:
    """Emergency fallback when the LLM router is unreachable. Keeps only unambiguous structural keywords;
    defaults to current_status_db to avoid hallucination via the knowledge executor."""
    q = question.lower()
    if _FALLBACK_COMPARISON_RE.search(q):
        intent = IntentType.COMPARISON_DB
    elif _FALLBACK_ANOMALY_RE.search(q):
        intent = IntentType.ANOMALY_ANALYSIS_DB
    elif _FALLBACK_AGGREGATION_RE.search(q):
        intent = IntentType.AGGREGATION_DB
    else:
        intent = IntentType.CURRENT_STATUS_DB

    return RoutePlan(
        intent=intent,
        confidence=0.5,
        lab_name=lab_name,
        second_lab_name=None,
        metrics=_extract_metrics_from_question(question),
        time_phrase=None,
        model="fallback",
        fallback_used=True,
    )


def _parse_llm_response(raw: str, question: str, lab_name: Optional[str]) -> Optional[RoutePlan]:
    text = raw.strip()
    # Strip thinking tags if present
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Extract JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return None

    intent_str = str(data.get("intent") or "").strip().lower()
    if intent_str not in _INTENT_VALUES:
        return None

    intent = IntentType(intent_str)
    try:
        confidence = float(data.get("confidence") or 0.75)
        confidence = max(0.0, min(1.0, confidence))
    except (TypeError, ValueError):
        confidence = 0.75

    lab = data.get("lab") or lab_name
    if isinstance(lab, str):
        lab = lab.strip().lower() or None

    second_lab = data.get("second_lab")
    if isinstance(second_lab, str):
        second_lab = second_lab.strip().lower() or None

    raw_metrics = data.get("metrics")
    if isinstance(raw_metrics, list) and raw_metrics:
        metrics = [str(m).strip().lower() for m in raw_metrics if m]
        metrics = [_METRIC_CANONICAL.get(m, m) for m in metrics]
        metrics = list(dict.fromkeys(m for m in metrics if m))
    else:
        metrics = _extract_metrics_from_question(question)
    if not metrics:
        metrics = ["ieq", "co2"]

    time_phrase = data.get("time_phrase")
    if isinstance(time_phrase, str):
        time_phrase = time_phrase.strip() or None

    return RoutePlan(
        intent=intent,
        confidence=confidence,
        lab_name=lab,
        second_lab_name=second_lab,
        metrics=metrics,
        time_phrase=time_phrase,
        model=router_model(),
        fallback_used=False,
    )


def _build_router_user_message(question: str, lab_name: Optional[str], conversation_context: str) -> str:
    """Build the user message for the router, including a compact prior-context snippet."""
    base = f"Question: {question}\nLab hint: {lab_name or '(none)'}"
    if not conversation_context:
        return base
    ctx_lines = [
        line for line in conversation_context.strip().splitlines()
        if line.strip() and not line.startswith("Previous conversation context")
    ]
    # Keep last 4 lines (≈ 2 prior turns) to avoid ballooning the router prompt.
    snippet = "\n".join(ctx_lines[-4:])
    return f"Prior conversation:\n{snippet}\n\n{base}"


def plan_route(question: str, lab_name: Optional[str] = None, conversation_context: str = "") -> RoutePlan:
    base_url = router_base_url()
    model = router_model()
    timeout = router_timeout_seconds()
    temperature = router_temperature()
    thinking = router_thinking_enabled()
    max_retries = router_max_retries()
    jitter_ms = router_retry_jitter_ms()

    user_message = _build_router_user_message(question, lab_name, conversation_context)
    options: dict = {"temperature": temperature, "num_predict": 256, "think": thinking}

    for attempt in range(max_retries):
        if attempt > 0:
            time.sleep((jitter_ms / 1000.0) * (1 + random.random()))
        try:
            resp = requests.post(
                f"{base_url}/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": user_message},
                    ],
                    "stream": False,
                    "options": options,
                },
                timeout=timeout,
            )
            resp.raise_for_status()
            raw = resp.json().get("message", {}).get("content", "")
            plan = _parse_llm_response(raw, question, lab_name)
            if plan is not None:
                return plan
        except Exception:
            pass

    return _fallback_plan(question, lab_name)


async def plan_route_async(question: str, lab_name: Optional[str] = None, conversation_context: str = "") -> RoutePlan:
    """Async version of plan_route — uses httpx so the event loop is never blocked."""
    base_url = router_base_url()
    model = router_model()
    timeout = router_timeout_seconds()
    temperature = router_temperature()
    thinking = router_thinking_enabled()
    max_retries = router_max_retries()
    jitter_ms = router_retry_jitter_ms()

    user_message = _build_router_user_message(question, lab_name, conversation_context)
    options: dict = {"temperature": temperature, "num_predict": 256, "think": thinking}

    for attempt in range(max_retries):
        if attempt > 0:
            await asyncio.sleep((jitter_ms / 1000.0) * (1 + random.random()))
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(
                    f"{base_url}/api/chat",
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": _SYSTEM_PROMPT},
                            {"role": "user", "content": user_message},
                        ],
                        "stream": False,
                        "options": options,
                    },
                )
                resp.raise_for_status()
                raw = resp.json().get("message", {}).get("content", "")
                plan = _parse_llm_response(raw, question, lab_name)
                if plan is not None:
                    return plan
        except Exception:
            pass

    return _fallback_plan(question, lab_name)
