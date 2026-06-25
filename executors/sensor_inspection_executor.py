"""Per-sensor inspection executor.

Answers questions about INDIVIDUAL sensors/devices — ranking them by a metric
("which sensor has the highest temperature?") and reporting sensor health
("which sensors are offline / faulty?") — from the per-device snapshot returned
by ``GET /spaces/{slug}/heatmap/metrics``.

Mirrors the grounding/streaming pattern of ``ifc_executor``: pre-compute the
device facts, feed a strictly-grounded textual snapshot to the answer LLM, and
fall back to a deterministic answer (computed from the same facts, never
fabricated) when the LLM is unreachable. A reading is treated as stale/offline
when its ``latest_timestamp`` is older than ``SENSOR_STALE_HOURS`` (default 24h).
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

from core_settings import (
    download_space_slug,
    ollama_base_url,
    ollama_model,
    ollama_temperature,
    ollama_thinking,
    ollama_timeout_seconds,
    sensor_stale_hours,
)
from executors.db_support import api_client
from ollama_helpers import extract_generate_chunk, extract_generate_text
from prompting.shared_prompts import PRESENTATION_STYLE_PROMPT


SENSOR_SYSTEM_PROMPT = (
    "You are a facility-monitoring assistant answering questions about the individual "
    "sensors (devices) in a monitored space. You are given a factual snapshot of every "
    "sensor's latest reading per metric, each with its value, unit, and the timestamp it "
    "was last reported.\n\n"
    "Rules:\n"
    "- Answer ONLY from the provided sensor snapshot. It is the single source of truth.\n"
    "- For 'highest/lowest/hottest' style questions, rank the sensors by the relevant "
    "metric's latest value and name the winning sensor with its value and unit.\n"
    "- A sensor's reading is FAULTY / OFFLINE / STALE when its last-reported timestamp is "
    "older than the stale threshold stated in the snapshot. For health questions, list the "
    "affected sensors with the metric and when they were last seen. If none are stale, say "
    "all sensors have reported recently.\n"
    "- Refer to sensors by their alias (or name) so the user can identify them.\n"
    "- NEVER invent sensors, values, or timestamps. If the snapshot does not contain the "
    "asked-for metric or sensor, say so plainly.\n"
    "\nPresentation rules:\n"
    f"{PRESENTATION_STYLE_PROMPT}\n"
    "- Do NOT wrap responses in triple backticks or output raw JSON."
)


# Question-word → canonical heatmap metric type (the API uses these `type` values).
_METRIC_WORDS = {
    "temperature": "temperature",
    "temp": "temperature",
    "hottest": "temperature",
    "coldest": "temperature",
    "warmest": "temperature",
    "coolest": "temperature",
    "humidity": "humidity",
    "humid": "humidity",
    "tvoc": "voc",
    "voc": "voc",
    "pm 2.5": "pm25",
    "pm2.5": "pm25",
    "pm 25": "pm25",
    "pm25": "pm25",
    "co2": "co2",
    "carbon dioxide": "co2",
}

_HIGH_RE = re.compile(r"\b(highest|hottest|warmest|max|maximum|most|top|largest|greatest)\b")
_LOW_RE = re.compile(r"\b(lowest|coldest|coolest|min|minimum|least|bottom|smallest)\b")
_HEALTH_RE = re.compile(
    r"\b(faulty|fault|offline|dead|broken|stale|down|not\s+reporting|hasn'?t\s+reported|"
    r"last\s+(?:seen|report(?:ed)?)|working|online|alive|disconnected)\b"
)


def _resolve_slug(lab: Optional[str]) -> str:
    """Turn a lab/space name into the API {slug}, falling back to the configured default."""
    if not lab:
        return download_space_slug()
    slug = re.sub(r"[^a-z0-9]+", "_", lab.strip().lower()).strip("_")
    return slug or download_space_slug()


def _parse_ts(raw: Any) -> Optional[datetime]:
    """Parse an ISO-8601 timestamp (with offset like +04:00 or Z) to a UTC datetime."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = raw.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _build_device_facts(
    devices: List[Dict[str, Any]], now: datetime, threshold_hours: int
) -> List[Dict[str, Any]]:
    """Normalise the raw devices payload into per-device, per-metric facts with staleness."""
    facts: List[Dict[str, Any]] = []
    for d in devices or []:
        metrics: List[Dict[str, Any]] = []
        for m in d.get("metrics") or []:
            ts_raw = m.get("latest_timestamp")
            ts = _parse_ts(ts_raw)
            age_hours = (now - ts).total_seconds() / 3600.0 if ts else None
            metrics.append({
                "type": m.get("type"),
                "name": m.get("metric_name"),
                "value": m.get("latest_value"),
                "unit": m.get("unit"),
                "timestamp": ts_raw,
                "age_hours": age_hours,
                "stale": age_hours is not None and age_hours > threshold_hours,
            })
        facts.append({
            "device_id": d.get("device_id"),
            "name": d.get("device_name"),
            "alias": d.get("device_alias") or d.get("device_name"),
            "metrics": metrics,
        })
    return facts


def _fmt_age(age_hours: Optional[float]) -> str:
    if age_hours is None:
        return "no timestamp"
    if age_hours < 1:
        return f"{age_hours * 60:.0f} min ago"
    if age_hours < 48:
        return f"{age_hours:.1f} h ago"
    return f"{age_hours / 24:.1f} days ago"


def _fmt_value(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:g}"
    return str(value)


def _build_context_text(slug: str, facts: List[Dict[str, Any]], now: datetime, threshold_hours: int) -> str:
    lines = [
        f"Sensor snapshot for space '{slug}' as of {now.isoformat()}.",
        f"A reading is STALE/OFFLINE when older than {threshold_hours} hours.",
        "",
        f"Devices ({len(facts)}):",
    ]
    for d in facts:
        lines.append(f"- {d['alias']} (name {d['name']}, id {d['device_id']}):")
        for m in d["metrics"]:
            flag = "STALE" if m["stale"] else "OK"
            lines.append(
                f"    {m['type']} = {_fmt_value(m['value'])} {m['unit']}, "
                f"last reported {m['timestamp']} ({_fmt_age(m['age_hours'])}) [{flag}]"
            )

    # Derived: offline/stale readings.
    stale = [
        (d["alias"], m) for d in facts for m in d["metrics"] if m["stale"]
    ]
    lines.append("")
    if stale:
        lines.append("Offline / stale readings:")
        for alias, m in stale:
            lines.append(f"- {alias} {m['type']}: last reported {m['timestamp']} ({_fmt_age(m['age_hours'])})")
    else:
        lines.append("Offline / stale readings: none — all sensors reported within the threshold.")

    # Derived: per-metric ranking (highest → lowest by latest value).
    lines.append("")
    lines.append("Ranking by latest value (highest → lowest):")
    for metric_type in _metric_types_present(facts):
        ranked = _ranked_readings(facts, metric_type)
        if not ranked:
            continue
        parts = [
            f"{alias} {_fmt_value(m['value'])} {m['unit']}{' (STALE)' if m['stale'] else ''}"
            for alias, m in ranked
        ]
        lines.append(f"  {metric_type}: " + ", ".join(parts))
    return "\n".join(lines)


def _metric_types_present(facts: List[Dict[str, Any]]) -> List[str]:
    seen: List[str] = []
    for d in facts:
        for m in d["metrics"]:
            t = m.get("type")
            if t and t not in seen:
                seen.append(t)
    return seen


def _ranked_readings(facts: List[Dict[str, Any]], metric_type: str) -> List[tuple]:
    """Return [(alias, metric_dict), …] with a numeric value, sorted highest → lowest."""
    readings = [
        (d["alias"], m)
        for d in facts
        for m in d["metrics"]
        if m.get("type") == metric_type and isinstance(m.get("value"), (int, float))
    ]
    readings.sort(key=lambda pair: pair[1]["value"], reverse=True)
    return readings


def _target_metric(question: str) -> Optional[str]:
    q = question.lower()
    for word in sorted(_METRIC_WORDS, key=len, reverse=True):
        if word in q:
            return _METRIC_WORDS[word]
    return None


def _evidence_sources(slug: str, facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "source_kind": "api",
            "table": "heatmap/metrics",
            "operation": "device_snapshot",
            "metric": None,
            "source_label": f"/spaces/{slug}/heatmap/metrics",
            "topic": "sensor_status",
            "title": "Per-sensor latest readings",
            "details": {"device_count": len(facts)},
        }
    ]


def _deterministic_fallback(question: str, facts: List[Dict[str, Any]], threshold_hours: int) -> str:
    """Plain-text answer when the answer LLM is unreachable — computed, never fabricated."""
    if not facts:
        return "No sensors reported data for this space, so I can't inspect them right now."

    q = question.lower()
    is_health = bool(_HEALTH_RE.search(q))
    metric_type = _target_metric(q)
    wants_low = bool(_LOW_RE.search(q))

    # Health / offline questions take priority when no ranking metric is implied.
    if is_health and not (metric_type and (wants_low or _HIGH_RE.search(q))):
        stale = [(d["alias"], m) for d in facts for m in d["metrics"] if m["stale"]]
        if not stale:
            return (
                f"All {len(facts)} sensors have reported within the last {threshold_hours} hours — "
                "none appear faulty or offline."
            )
        items = "; ".join(
            f"{alias} ({m['type']}, last seen {m['timestamp']}, {_fmt_age(m['age_hours'])})"
            for alias, m in stale
        )
        return (
            f"{len(stale)} sensor reading(s) look stale/offline (older than {threshold_hours}h): {items}."
        )

    # Ranking questions.
    if metric_type:
        ranked = _ranked_readings(facts, metric_type)
        if not ranked:
            return f"No sensor is currently reporting {metric_type}."
        alias, m = ranked[-1] if wants_low else ranked[0]
        direction = "lowest" if wants_low else "highest"
        note = " (note: this reading is stale)" if m["stale"] else ""
        return (
            f"{alias} has the {direction} {metric_type} at {_fmt_value(m['value'])} {m['unit']}, "
            f"last reported {m['timestamp']}{note}."
        )

    # Generic snapshot summary.
    stale_count = sum(1 for d in facts for m in d["metrics"] if m["stale"])
    return (
        f"There are {len(facts)} sensors in this space. "
        + (f"{stale_count} reading(s) appear stale/offline." if stale_count else "All readings are current.")
    )


def _build_prompt(question: str, context_text: str) -> str:
    return (
        f"{SENSOR_SYSTEM_PROMPT}\n\n"
        f"=== Sensor Snapshot ===\n{context_text}\n=== End Snapshot ===\n\n"
        f"Question: {question}\n\n"
        "Answer the question using only the sensor snapshot above."
    )


def _no_data_answer() -> str:
    return (
        "I couldn't retrieve the per-sensor readings for this space right now, so I can't "
        "inspect individual sensors. Please try again shortly."
    )


def answer_sensor_question_with_metadata(user_question: str, space: Optional[str] = None) -> Dict[str, Any]:
    """Synchronously answer a per-sensor question with grounding metadata."""
    slug = _resolve_slug(space)
    threshold = sensor_stale_hours()
    now = datetime.now(tz=timezone.utc)
    devices = api_client.fetch_heatmap_metrics(slug)
    facts = _build_device_facts(devices, now, threshold)

    if not facts:
        return {
            "answer": _no_data_answer(),
            "footnotes": [],
            "indexed_sources": _evidence_sources(slug, facts),
            "llm_used": False,
        }

    context_text = _build_context_text(slug, facts, now, threshold)
    prompt = _build_prompt(user_question, context_text)
    payload = {
        "model": ollama_model(),
        "prompt": prompt,
        "stream": False,
        "think": ollama_thinking(),
        "temperature": ollama_temperature(),
    }
    answer = ""
    llm_used = False
    try:
        with httpx.Client(timeout=ollama_timeout_seconds()) as client:
            response = client.post(f"{ollama_base_url()}/api/generate", json=payload)
            response.raise_for_status()
            answer = extract_generate_text(response.json()).strip()
            llm_used = bool(answer)
    except Exception:
        answer = ""

    if not answer:
        answer = _deterministic_fallback(user_question, facts, threshold)
        llm_used = False

    return {
        "answer": answer.strip(),
        "footnotes": [],
        "indexed_sources": _evidence_sources(slug, facts),
        "llm_used": llm_used,
    }


async def stream_sensor_tokens(user_question: str, space: Optional[str] = None) -> AsyncIterator[str]:
    """Stream a per-sensor answer as SSE token events."""
    slug = _resolve_slug(space)
    threshold = sensor_stale_hours()
    now = datetime.now(tz=timezone.utc)
    devices = api_client.fetch_heatmap_metrics(slug)
    facts = _build_device_facts(devices, now, threshold)

    if not facts:
        yield f"data: {json.dumps({'event': 'token', 'text': _no_data_answer()})}\n\n"
        yield f"data: {json.dumps({'event': 'done'})}\n\n"
        return

    context_text = _build_context_text(slug, facts, now, threshold)
    prompt = _build_prompt(user_question, context_text)
    payload = {
        "model": ollama_model(),
        "prompt": prompt,
        "stream": True,
        "think": ollama_thinking(),
        "temperature": ollama_temperature(),
    }
    produced = False
    try:
        async with httpx.AsyncClient(timeout=ollama_timeout_seconds()) as client:
            async with client.stream("POST", f"{ollama_base_url()}/api/generate", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = extract_generate_chunk(event)
                    if text:
                        produced = True
                        yield f"data: {json.dumps({'event': 'token', 'text': text})}\n\n"
    except Exception:
        pass

    if not produced:
        # LLM unreachable — emit the deterministic, grounded answer so the stream is never empty.
        fallback = _deterministic_fallback(user_question, facts, threshold)
        yield f"data: {json.dumps({'event': 'token', 'text': fallback})}\n\n"

    yield f"data: {json.dumps({'event': 'done'})}\n\n"
