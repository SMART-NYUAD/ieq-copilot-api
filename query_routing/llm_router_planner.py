"""LLM-based query router: single call, structured JSON output."""

from __future__ import annotations

import asyncio
import json
import random
import re
import time
from typing import Dict, List, Optional

import httpx
import requests

from core_settings import (
    router_base_url,
    router_max_retries,
    router_model,
    router_retry_jitter_ms,
    router_temperature,
    router_thinking,
    router_timeout_seconds,
)
from ollama_helpers import extract_chat_content
from query_routing.intent_classifier import IntentType
from query_routing.router_types import RoutePlan


_INTENT_VALUES = {i.value for i in IntentType}
_METRIC_RE = re.compile(r"\b(co2|pm\s*2\.?\s*5|pm25|tvoc|voc|temperature|temp|humidity|light|lux|sound|noise|ieq)\b")
_METRIC_CANONICAL = {
    "pm 2.5": "pm25", "pm2.5": "pm25", "pm 25": "pm25",
    "tvoc": "voc", "temp": "temperature", "lux": "light", "noise": "sound",
}

_SYSTEM_PROMPT = (
    "You are an indoor air quality query router for a facility management system.\n"
    "Given a user question and optional lab hint, output ONLY a JSON object with these fields:\n"
    '  "intent": one of [definition_explanation, current_status_db, point_lookup_db, '
    "forecast_db, aggregation_db, comparison_db, anomaly_analysis_db, viewer_control, "
    "heatmap_control, download_data, ifc_model_qa, sensor_inspection, unknown_fallback]\n"
    '  "lab": the lab/space name if mentioned, else null\n'
    '  "second_lab": always null\n'
    '  "metrics": list of relevant metrics from [co2, pm25, voc, humidity, temperature, light, sound, ieq]\n'
    '  "time_phrase": exact time window phrase from question (e.g. "last 24 hours"), else null\n'
    '  "confidence": float 0-1\n'
    '  "viewer_type": one of ["splat", "ifc", "pc", "pano"] when intent is viewer_control, else null\n'
    '  "heatmap_action": one of ["on", "off"] when intent is heatmap_control, else null\n'
    '  "heatmap_metric": one of ["temperature", "humidity", "voc", "pm25"] when intent is '
    "heatmap_control and a metric is named, else null\n"
    '  "download_format": one of ["csv", "json"] when intent is download_data (default "csv"), else null\n'
    '  "download_metric": the single metric to export when intent is download_data, one of '
    '["temperature", "humidity", "co2", "voc", "pm25"] — null if the user did not name one\n'
    '  "download_interval": aggregation bucket size for the export when intent is download_data '
    '(e.g. "1m", "15m", "1h", "1d") — the granularity, NOT the time range; else null\n\n'
    "Routing rules:\n"
    "- DOMAIN GUARDRAIL: This assistant only handles indoor environmental quality, sensor data, "
    "building/BIM/IFC model questions, viewer-control, and heatmap-overlay requests. If the question is unrelated "
    "(sports, politics, travel, cooking, coding, weather outside the monitored spaces, personal chat), "
    "nonsensical/gibberish, or impossible to map to the supported domain, use `unknown_fallback` "
    "with high confidence. Do not force out-of-domain or nonsense questions into DB, IFC, viewer, "
    "or definition routes.\n"
    "- CRITICAL OVERRIDE: If the question contains the words 'forecast', 'predict', 'prediction', "
    "'will be', 'going to be', or asks about FUTURE sensor values, ALWAYS use `forecast_db` — "
    "with NO exceptions. This rule beats every other rule, including the definite-article "
    "heuristic for current_status_db. 'Predict the X' means forecast_db, NOT current_status_db. "
    "Never route these to `aggregation_db`, `current_status_db`, or any other intent. "
    "Prior conversation showing historical/aggregation data does NOT change this. "
    "`forecast_db` = future data only. `aggregation_db` = past data only.\n"
    "- When Prior conversation is present, use it only to fill missing lab/time slots. "
    "The current Question sets metric/topic scope; do not route to a prior-turn metric when "
    "the user asked a different scope (e.g. 'air quality' after a temperature question → IEQ/IAQ, not temperature). "
    "Prior context NEVER overrides the CRITICAL OVERRIDE rule above.\n\n"
    "Intent definitions:\n"
    "- definition_explanation: ONLY for conceptual/educational questions asking what a metric means, "
    "with no definite article before the metric name. "
    "Examples: 'what is CO2?', 'what does IEQ mean?', 'explain VOC', 'define humidity', 'what is pm2.5?'.\n"
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
    "- forecast_db: FUTURE/PREDICTED sensor values only — the system provides 6-hour ahead predictions. "
    "Use whenever the user says 'forecast', 'predict', 'prediction', 'will', 'going to be', or asks about "
    "what values will be in the future. NEVER use for historical/past data. "
    "IMPORTANT: 'predict the X' constructions always map here even though they contain 'the' — "
    "the predict/forecast keyword overrides the definite-article rule. "
    "Examples: 'can you forecast the PM2.5?', 'forecast pm2.5', 'predict CO2 levels', "
    "'predict the pm2.5', 'predict the CO2', 'predict the temperature', "
    "'what will the temperature be?', 'give me a forecast', 'PM2.5 prediction', "
    "'what is the expected air quality?', 'will humidity rise?', 'CO2 forecast'.\n"
    "- aggregation_db: PAST data only — trends, historical averages, or summaries across a time window. "
    "NEVER use for future/predicted values (that is `forecast_db`). "
    "Examples: 'average CO2 last 24 hours', 'how has humidity trended this week?', "
    "'daily temperature summary', 'CO2 over the past 7 days'.\n"
    "- comparison_db: Comparing two or more metrics in the same lab, or the same metric across two time periods. "
    "Examples: 'compare CO2 and humidity', 'humidity vs CO2', 'is it better than yesterday?', "
    "'today vs last week'. Never use for cross-space comparisons.\n"
    "- anomaly_analysis_db: Detecting or explaining unusual readings, spikes, or outliers. "
    "Examples: 'why did CO2 spike?', 'any unusual readings today?', 'were there anomalies last night?', "
    "'what caused that PM2.5 outlier?'.\n"
    "- viewer_control: The user wants to switch or open a specific 3D visualization mode in the viewer. "
    "Set viewer_type to the matching value: "
    "'splat' for Gaussian splat / splat view; "
    "'ifc' for IFC, floor plan, BIM, or 3D model; "
    "'pc' for point cloud or PC view; "
    "'pano' for panorama or 360-degree view. "
    "Examples: 'show me the splat', 'open the gaussian splat', 'switch to IFC', 'open the floor plan', "
    "'show me the point cloud', 'open pc view', 'show the panorama', 'open pano', "
    "'can you open the 3D model', 'switch to point cloud mode'.\n"
    "- heatmap_control: The user wants to toggle the heatmap/overlay on or off, or change which metric "
    "the heatmap colors the model by. Set heatmap_action to 'on' when enabling, switching, or selecting a "
    "metric (selecting a metric implies on); set it to 'off' when disabling, hiding, or removing the heatmap. "
    "Set heatmap_metric to the named metric only if it is one of [temperature, humidity, voc, pm25], else null. "
    "A single request may both enable the heatmap AND pick a metric — fill both fields then. "
    "This is a UI overlay control, NOT a data question: it never returns sensor values. "
    "Examples: 'turn on the heatmap' (on, null), 'enable the heatmap' (on, null), "
    "'turn off the heatmap' (off, null), 'hide the heatmap' (off, null), "
    "'change the heatmap to temperature' (on, temperature), 'color the model by humidity' (on, humidity), "
    "'switch the overlay to voc' (on, voc), 'show the pm2.5 heatmap' (on, pm25), "
    "'turn on the heatmap and use the metric temperature' (on, temperature), "
    "'set the heatmap metric to humidity' (on, humidity).\n"
    "- download_data: The user EXPLICITLY wants to DOWNLOAD, EXPORT, or SAVE the sensor readings as a "
    "file (CSV/JSON), or get a data dump / spreadsheet / report FILE of the measurements. There must be "
    "an explicit file/export verb — 'download', 'export', 'save to a file', 'CSV', 'JSON', 'spreadsheet', "
    "'data dump'. This is a UI action that hands the frontend the download parameters — it never returns "
    "sensor values inline, and it is NOT a question about the data. "
    "IMPORTANT — do NOT route here for a bare data REQUEST. 'give me the CO2 data', 'show me the "
    "temperature', 'get me last week's readings', 'what was the humidity', 'pull the data for May' are "
    "data QUESTIONS — route them to the matching DB intent (aggregation_db / point_lookup_db / "
    "current_status_db), NOT download_data. The word 'data' alone is NOT a download cue. When it is "
    "AMBIGUOUS whether the user wants a file or an answer, PREFER the DB data path; only choose "
    "download_data when a file/export verb is explicitly present. Likewise, do NOT inherit download_data "
    "from a previous turn: if the prior question was a download but THIS question merely asks for data "
    "(no explicit export verb), route it to the DB path. "
    "Carry the time window in `time_phrase` exactly as stated "
    "(e.g. 'last 7 days', 'March'); when no window is given the system defaults to the last 24 hours. "
    "Set download_format to 'json' only if the user explicitly asks for JSON, else 'csv'. "
    "Set download_metric to the single metric the user wants to export (temperature, humidity, co2, "
    "voc, pm25); leave it null if the user did not name a metric — a metric is REQUIRED, so the system "
    "will ask the user which metric when it is missing. Set download_interval to the aggregation "
    "granularity the user asks for — both named ('hourly'->'1h', 'daily'->'1d') and explicit numeric "
    "('a 1 hour interval'->'1h', 'every 15 minutes'->'15m') forms; else null. The interval is the "
    "bucket SIZE, not the time range — 'set the interval to 1 hour' / 'make it more granular, hourly' "
    "changes download_interval (->'1h'); it does NOT set time_phrase. "
    "Distinguish from aggregation_db: 'what was the average CO2 last week?' is a "
    "data QUESTION (aggregation_db); 'download last week's data' / 'export the readings' is download_data. "
    "Examples: 'download the data', 'export the sensor readings', 'export last 7 days as CSV', "
    "'can I get a CSV of the readings?', 'download March data', 'save the measurements to a file', "
    "'give me a spreadsheet of the data', 'export the raw readings as JSON', "
    "'download the last 30 days', 'I want to download the sensor data'.\n"
    "- ifc_model_qa: The user asks a QUESTION ABOUT the building/BIM/IFC model itself — its geometry, "
    "dimensions, measurements, number of elements, levels/floors, rooms, materials, or element "
    "properties. This answers from the building model, it does NOT open a viewer. "
    "Distinguish carefully from viewer_control: 'open the IFC view' / 'switch to the floor plan' is "
    "viewer_control (a UI action); 'how many doors does the building have?' / 'what is the building "
    "made of?' is ifc_model_qa (a question about the model). When the user wants a FACT about the "
    "structure, walls, doors, windows, columns, slabs, furniture, lights, storeys, floor heights, "
    "materials, or sizes of the building, use ifc_model_qa. These are about the physical building "
    "structure, NOT about air-quality sensors. "
    "Examples: 'how many columns are in the building?', 'how many floors does it have?', "
    "'what are the dimensions of the door?', 'how tall is the building?', 'what materials are used?', "
    "'how many desks are there?', 'what is the wall thickness?', 'list the rooms', "
    "'how big are the columns?', 'what's the elevation of level 1?', 'how many lights are installed?', "
    "'tell me about the building model', 'what does the IFC contain?', 'how many windows?', "
    "'what is the size of the model?', 'how big is the building?', 'what are the overall dimensions?', "
    "'what is the floor area?', 'what is the footprint of the building?', "
    "'what is the GIA?', 'what is the gross internal area?', 'what is the gross floor area?', "
    "'what is the net internal area?', 'what is the building volume?', "
    "'what is the floor-to-floor height?', 'what is the wall thickness?', 'what is the perimeter?'.\n\n"
    "- sensor_inspection: The question is about INDIVIDUAL sensors/devices rather than the "
    "space as a whole — comparing sensors against each other (which sensor reads the "
    "highest/lowest of a metric), or asking about a sensor's HEALTH / online status "
    "(faulty, offline, dead, stale, hasn't reported, last seen, is it working). This is "
    "answered from a per-device snapshot of each sensor's latest reading and timestamp. "
    "Distinguish from current_status_db: 'what is the temperature?' asks for the SPACE "
    "value (current_status_db); 'which SENSOR has the highest temperature?' / 'which sensor "
    "is hottest?' compares sensors (sensor_inspection). Any question about a specific named "
    "sensor/device, a ranking of sensors, or sensor faults/offline status goes here. "
    "Examples: 'which sensor has the highest temperature?', 'which sensor reads the lowest "
    "humidity?', 'which sensor is hottest?', 'are any sensors faulty?', 'which sensors are "
    "offline?', 'find a dead sensor', 'which sensor hasn\\'t reported today?', 'list sensors "
    "with stale data', 'is sensor 8 working?', 'when did sensor 67 last report?', "
    "'show me broken sensors'.\n\n"
    "- unknown_fallback: The question is outside supported topics or makes no coherent request. "
    "Examples: 'who won the football match?', 'write me a poem about dragons', 'asdf qwer banana?', "
    "'what is the stock price of Apple?', 'tell me a joke', 'what is the weather in Paris?'.\n\n"
    "Output only the JSON object, no markdown, no explanation."
)


_VALID_VIEWER_TYPES = {"splat", "ifc", "pc", "pano"}

# Ordered longest-first so multi-word phrases match before single words.
_VIEWER_TYPE_ALIASES: Dict[str, str] = {
    "gaussian splat": "splat",
    "point cloud": "pc",
    "floor plan": "ifc",
    "floorplan": "ifc",
    "panorama": "pano",
    "gaussian": "splat",
    "splat": "splat",
    "pano": "pano",
    "ifc": "ifc",
    "bim": "ifc",
}


def _infer_viewer_type(question: str) -> str:
    """Derive viewer_type from question text without regex when the LLM omits the field."""
    q = question.lower()
    for alias in sorted(_VIEWER_TYPE_ALIASES, key=len, reverse=True):
        if alias in q:
            return _VIEWER_TYPE_ALIASES[alias]
    return "splat"


_VALID_HEATMAP_METRICS = {"temperature", "humidity", "voc", "pm25"}
_VALID_HEATMAP_ACTIONS = {"on", "off"}

# Ordered longest-first so multi-word phrases match before single words.
_HEATMAP_METRIC_ALIASES: Dict[str, str] = {
    "temperature": "temperature",
    "temp": "temperature",
    "humidity": "humidity",
    "tvoc": "voc",
    "voc": "voc",
    "pm 2.5": "pm25",
    "pm2.5": "pm25",
    "pm 25": "pm25",
    "pm25": "pm25",
}
# Substrings that indicate the user wants the heatmap turned off rather than on.
_HEATMAP_OFF_HINTS = (
    "turn off",
    "switch off",
    "shut off",
    "disable",
    "hide",
    "remove",
    "clear",
    "stop",
    "no heatmap",
    "without the heatmap",
)


_VALID_DOWNLOAD_FORMATS = {"csv", "json"}
_VALID_DOWNLOAD_METRICS = {"temperature", "humidity", "co2", "voc", "pm25"}

# Canonical metric → metric_type path segment expected by the download-agg-summary endpoint.
_DOWNLOAD_METRIC_TYPES: Dict[str, str] = {
    "temperature": "temperature",
    "humidity": "humidity",
    "co2": "co2",
    "voc": "voc",
    "pm25": "pm2.5",
}

# Aliases for the downloadable metrics (superset of the heatmap aliases — also covers co2).
_DOWNLOAD_METRIC_ALIASES: Dict[str, str] = {
    "temperature": "temperature",
    "temp": "temperature",
    "humidity": "humidity",
    "tvoc": "voc",
    "voc": "voc",
    "co2": "co2",
    "carbon dioxide": "co2",
    "pm 2.5": "pm25",
    "pm2.5": "pm25",
    "pm 25": "pm25",
    "pm25": "pm25",
}


def _infer_download_format(question: str) -> str:
    """Pick the download format from question text when the LLM omits it. Defaults to csv."""
    return "json" if "json" in question.lower() else "csv"


def _infer_download_metric(question: str) -> Optional[str]:
    """Derive the export metric from question text without regex when the LLM omits it.

    Returns None when no metric is named — the caller treats that as "must ask the user".
    """
    q = question.lower()
    for alias in sorted(_DOWNLOAD_METRIC_ALIASES, key=len, reverse=True):
        if alias in q:
            return _DOWNLOAD_METRIC_ALIASES[alias]
    return None


def _canon_interval_unit(unit: str) -> Optional[str]:
    """Map a spelled-out/abbreviated time unit to the export interval suffix (m/h/d)."""
    u = unit.rstrip("s")
    if u in ("minute", "min"):
        return "m"
    if u in ("hour", "hr"):
        return "h"
    if u == "day":
        return "d"
    return None


def _infer_download_interval(question: str) -> Optional[str]:
    """Derive the aggregation interval (e.g. '1h', '15m', '1d') from the question, else None.

    Recognises both named granularities ('hourly', 'daily') and explicit numeric
    intervals ('1 hour interval', 'every 15 minutes', '1h'). A numeric value is only
    read as an interval when it sits next to interval/cadence wording (or a compact
    form like '1h'), so a plain time window such as 'last 1 hour' is never mistaken
    for an interval. The caller defaults the interval when this returns None.
    """
    q = question.lower()

    # Compact forms: "1h", "15m", "1d" (optionally hyphenated, e.g. "1-h").
    compact = re.search(r"\b(\d+)\s*-?\s*(m|h|d)\b", q)
    if compact:
        return f"{int(compact.group(1))}{compact.group(2)}"

    # Numeric interval tied to cadence/granularity wording: "every 15 minutes",
    # "per 2 hours", "interval of 1 day", "1 hour interval", "30 minute buckets".
    numeric = re.search(
        r"(?:every|per|each|interval of)\s+(\d+)\s*(minutes?|mins?|hours?|hrs?|days?)\b"
        r"|(?:intervals?|granularity|resolution)\s+(?:to|of|at|=|:)?\s*(\d+)\s*"
        r"(minutes?|mins?|hours?|hrs?|days?)\b"
        r"|(\d+)\s*(minutes?|mins?|hours?|hrs?|days?)\s+"
        r"(?:intervals?|granularity|resolution|buckets?)",
        q,
    )
    if numeric:
        num = numeric.group(1) or numeric.group(3) or numeric.group(5)
        unit = _canon_interval_unit(numeric.group(2) or numeric.group(4) or numeric.group(6))
        if unit:
            return f"{int(num)}{unit}"

    # Named granularities.
    if "daily" in q or "per day" in q or "by day" in q or "every day" in q:
        return "1d"
    if "hourly" in q or "per hour" in q or "by hour" in q or "every hour" in q:
        return "1h"
    if "per minute" in q or "by minute" in q or "every minute" in q or "minute" in q:
        return "1m"
    return None


def _infer_heatmap_metric(question: str) -> Optional[str]:
    """Derive heatmap_metric from question text without regex when the LLM omits the field."""
    q = question.lower()
    for alias in sorted(_HEATMAP_METRIC_ALIASES, key=len, reverse=True):
        if alias in q:
            return _HEATMAP_METRIC_ALIASES[alias]
    return None


def _infer_heatmap_action(question: str) -> str:
    """Derive heatmap_action from question text without regex when the LLM omits the field.

    Defaults to "on" — selecting a metric or a bare "heatmap" request enables it; only an
    explicit off-hint disables it.
    """
    q = question.lower()
    if any(hint in q for hint in _HEATMAP_OFF_HINTS):
        return "off"
    return "on"


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
# A per-sensor question pairs a device noun with a superlative or a health/offline keyword.
_FALLBACK_SENSOR_RE = re.compile(r"\b(sensor|sensors|device|devices)\b")
_FALLBACK_SENSOR_SIGNAL_RE = re.compile(
    r"\b(highest|lowest|hottest|coldest|max|maximum|min|minimum|faulty|fault|offline|"
    r"dead|broken|stale|working|down|not\s+reporting|hasn'?t\s+reported|last\s+(?:seen|report))\b"
)


def _fallback_plan(question: str, lab_name: Optional[str]) -> RoutePlan:
    """Emergency fallback when the LLM router is unreachable. Keeps only unambiguous structural keywords;
    defaults to current_status_db to avoid hallucination via the knowledge executor."""
    q = question.lower()
    if _FALLBACK_SENSOR_RE.search(q) and _FALLBACK_SENSOR_SIGNAL_RE.search(q):
        intent = IntentType.SENSOR_INSPECTION
    elif _FALLBACK_COMPARISON_RE.search(q):
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

    viewer_type: Optional[str] = None
    if intent == IntentType.VIEWER_CONTROL:
        raw_vt = str(data.get("viewer_type") or "").strip().lower()
        viewer_type = raw_vt if raw_vt in _VALID_VIEWER_TYPES else _infer_viewer_type(question)

    heatmap_action: Optional[str] = None
    heatmap_metric: Optional[str] = None
    if intent == IntentType.HEATMAP_CONTROL:
        raw_action = str(data.get("heatmap_action") or "").strip().lower()
        heatmap_action = raw_action if raw_action in _VALID_HEATMAP_ACTIONS else _infer_heatmap_action(question)
        raw_metric = str(data.get("heatmap_metric") or "").strip().lower()
        raw_metric = _METRIC_CANONICAL.get(raw_metric, raw_metric)
        heatmap_metric = raw_metric if raw_metric in _VALID_HEATMAP_METRICS else _infer_heatmap_metric(question)

    download_format: Optional[str] = None
    download_metric: Optional[str] = None
    download_interval: Optional[str] = None
    if intent == IntentType.DOWNLOAD_DATA:
        raw_format = str(data.get("download_format") or "").strip().lower()
        download_format = raw_format if raw_format in _VALID_DOWNLOAD_FORMATS else _infer_download_format(question)
        raw_metric = str(data.get("download_metric") or "").strip().lower()
        raw_metric = _METRIC_CANONICAL.get(raw_metric, raw_metric)
        download_metric = raw_metric if raw_metric in _VALID_DOWNLOAD_METRICS else _infer_download_metric(question)
        raw_interval = str(data.get("download_interval") or "").strip().lower()
        download_interval = raw_interval or _infer_download_interval(question)

    return RoutePlan(
        intent=intent,
        confidence=confidence,
        lab_name=lab,
        second_lab_name=second_lab,
        metrics=metrics,
        time_phrase=time_phrase,
        model=router_model(),
        fallback_used=False,
        viewer_type=viewer_type,
        heatmap_action=heatmap_action,
        heatmap_metric=heatmap_metric,
        download_format=download_format,
        download_metric=download_metric,
        download_interval=download_interval,
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
    max_retries = router_max_retries()
    jitter_ms = router_retry_jitter_ms()

    user_message = _build_router_user_message(question, lab_name, conversation_context)
    options: dict = {"temperature": temperature, "num_predict": 256}

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
                    "think": router_thinking(),
                    "options": options,
                },
                timeout=timeout,
            )
            resp.raise_for_status()
            raw = extract_chat_content(resp.json().get("message", {}))
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
    max_retries = router_max_retries()
    jitter_ms = router_retry_jitter_ms()

    user_message = _build_router_user_message(question, lab_name, conversation_context)
    options: dict = {"temperature": temperature, "num_predict": 256}

    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(max_retries):
            if attempt > 0:
                await asyncio.sleep((jitter_ms / 1000.0) * (1 + random.random()))
            try:
                resp = await client.post(
                    f"{base_url}/api/chat",
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": _SYSTEM_PROMPT},
                            {"role": "user", "content": user_message},
                        ],
                        "stream": False,
                        "think": router_thinking(),
                        "options": options,
                    },
                )
                resp.raise_for_status()
                raw = extract_chat_content(resp.json().get("message", {}))
                plan = _parse_llm_response(raw, question, lab_name)
                if plan is not None:
                    return plan
            except Exception:
                pass

    return _fallback_plan(question, lab_name)
