"""LLM-based query router: single call, structured JSON output."""

from __future__ import annotations

import json
import random
import re
import time
from typing import List, Optional

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
_FORECAST_RE = re.compile(r"\b(forecast|predict|prediction|project|tomorrow|next\s+(hour|day|week))\b")
_COMPARISON_RE = re.compile(r"\b(compare|comparison|versus|vs\.?|between)\b")
_ANOMALY_RE = re.compile(r"\b(anomal|spike|outlier|unusual|abnormal|deviation)\b")
_AGGREGATION_RE = re.compile(r"\b(trend|average|avg|mean|sum|over\s+the|last\s+\d|past\s+\d|history|historical|weekly|daily)\b")
_CURRENT_RE = re.compile(r"\b(current|now|right now|latest|at this moment|most recent)\b")
_DEFINITION_RE = re.compile(r"\b(what\s+is|what\s+does|meaning\s+of|definition|define|explain)\b")

_SYSTEM_PROMPT = (
    "You are an indoor air quality query router for a facility management system.\n"
    "Given a user question and optional lab hint, output ONLY a JSON object with these fields:\n"
    '  "intent": one of [definition_explanation, current_status_db, aggregation_db, '
    "comparison_db, anomaly_analysis_db, forecast_db]\n"
    '  "lab": the lab/space name if mentioned, else null\n'
    '  "second_lab": second lab name for comparisons, else null\n'
    '  "metrics": list of relevant metrics from [co2, pm25, tvoc, humidity, temperature, light, sound, ieq]\n'
    '  "time_phrase": exact time window phrase from question (e.g. "last 24 hours"), else null\n'
    '  "confidence": float 0-1\n\n'
    "Routing rules:\n"
    "- definition_explanation: conceptual questions, what is X, how does X work, explain\n"
    "- current_status_db: current/latest readings without a time range\n"
    "- aggregation_db: trends, averages, summaries over a time window\n"
    "- comparison_db: comparing two or more labs/spaces\n"
    "- anomaly_analysis_db: anomalies, spikes, outliers, unusual readings\n"
    "- forecast_db: predictions, future values, tomorrow, next N hours/days\n\n"
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


def _fallback_plan(question: str, lab_name: Optional[str]) -> RoutePlan:
    q = question.lower()
    if _FORECAST_RE.search(q):
        intent = IntentType.FORECAST_DB
    elif _COMPARISON_RE.search(q):
        intent = IntentType.COMPARISON_DB
    elif _ANOMALY_RE.search(q):
        intent = IntentType.ANOMALY_ANALYSIS_DB
    elif _AGGREGATION_RE.search(q):
        intent = IntentType.AGGREGATION_DB
    elif _CURRENT_RE.search(q):
        intent = IntentType.CURRENT_STATUS_DB
    elif _DEFINITION_RE.search(q) and not lab_name:
        intent = IntentType.DEFINITION_EXPLANATION
    elif lab_name:
        intent = IntentType.CURRENT_STATUS_DB
    else:
        intent = IntentType.DEFINITION_EXPLANATION

    return RoutePlan(
        intent=intent,
        confidence=0.65,
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


def plan_route(question: str, lab_name: Optional[str] = None) -> RoutePlan:
    base_url = router_base_url()
    model = router_model()
    timeout = router_timeout_seconds()
    temperature = router_temperature()
    thinking = router_thinking_enabled()
    max_retries = router_max_retries()
    jitter_ms = router_retry_jitter_ms()

    user_message = f"Question: {question}\nLab hint: {lab_name or '(none)'}"
    options: dict = {"temperature": temperature, "num_predict": 256}
    if thinking:
        options["think"] = True

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
