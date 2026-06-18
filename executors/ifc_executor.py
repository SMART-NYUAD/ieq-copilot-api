"""IFC building-model QA executor.

Answers natural-language questions about the BIM/IFC model (geometry counts,
dimensions, levels, materials, properties) by feeding a fully-grounded textual
description of the model — parsed from the IFC file — to the answer LLM. Mirrors
the grounding/streaming pattern of ``knowledge_executor`` so sync and stream
paths share behavior, but the context is the building model rather than env
knowledge cards.

All facts come straight from the IFC file; the model is instructed to answer only
from that context and never to invent dimensions or counts.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, List

import httpx

from core_settings import (
    ifc_model_path,
    ollama_base_url,
    ollama_model,
    ollama_temperature,
    ollama_timeout_seconds,
)
from ifc_model.ifc_store import build_ifc_context_text, get_ifc_summary


IFC_SYSTEM_PROMPT = (
    "You are a BIM (Building Information Modeling) assistant answering questions about a "
    "specific building described by an IFC model. You are given a structured, factual "
    "description of that model parsed directly from the IFC file.\n\n"
    "Rules:\n"
    "- Answer ONLY from the provided building-model context. It is the single source of truth.\n"
    "- If a measurement, count, material, or property is in the context, report it exactly, "
    "with its unit (mm, m², etc.).\n"
    "- If the asked-for detail is not present in the context, say plainly that the model does not "
    "contain that information. NEVER guess or fabricate dimensions, counts, or properties.\n"
    "- When the user asks 'how many', count the relevant elements from the inventory and answer "
    "with the number.\n"
    "- For questions about the building's overall size, dimensions, footprint, or how big/tall it is, "
    "use the 'Overall Model Dimensions' section (world-space bounding box). Report footprint and "
    "height with units. Note it reflects the modeled geometry's extents.\n"
    "- For architectural / area / quantity-surveying metrics — GIA (Gross Internal Area), gross floor "
    "area, footprint area, floor-to-floor height, gross internal volume, perimeter, wall thickness, "
    "number of storeys, window-to-wall ratio — use the 'Architectural Metrics' section. Report the "
    "value with its unit (m², m, m³). If asked for GIA specifically, give the figure and briefly state "
    "its basis (sum of floor-plate areas, gross). If asked for NIA (Net Internal Area) and only GIA is "
    "available, say NIA is not computed and explain it would deduct internal walls/columns/circulation "
    "— do not invent an NIA number.\n"
    "- Element 'name' fields come from the authoring tool (e.g. Revit) and often encode the family, "
    "type, and size (e.g. '800x800mm', 'Generic - 200mm'); you may read sizes from these names.\n"
    "- Be concise and direct. Lead with a one-sentence answer, then add brief supporting detail "
    "(a few bullets) only when helpful.\n"
    "- Use plain language; expand IFC jargon where it helps a non-expert.\n"
    "- Do NOT wrap responses in triple backticks or output raw JSON."
)


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(_coerce_text(v) for v in value)
    if isinstance(value, dict):
        return str(value.get("text", ""))
    return str(value)


def _build_prompt(user_question: str, context_text: str) -> str:
    return (
        f"{IFC_SYSTEM_PROMPT}\n\n"
        f"=== Building Model Context ===\n{context_text}\n=== End Context ===\n\n"
        f"Question: {user_question}\n\n"
        "Answer the question using only the building model context above."
    )


def _evidence_sources(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {
            "source_kind": "ifc_model",
            "table": "ifc_file",
            "operation": "model_parse",
            "metric": None,
            "source_label": summary.get("path"),
            "topic": "building_model",
            "title": summary.get("project") or summary.get("building") or "IFC model",
            "details": {
                "schema": summary.get("schema"),
                "total_elements": summary.get("total_elements"),
            },
        }
    ]


def answer_ifc_question_with_metadata(user_question: str) -> Dict[str, Any]:
    """Synchronously answer an IFC-model question with grounding metadata."""
    path = ifc_model_path()
    try:
        context_text = build_ifc_context_text(path)
        summary = get_ifc_summary(path)
    except FileNotFoundError:
        return {
            "answer": "The building (IFC) model is not available on the server, so I can't "
            "answer questions about it right now.",
            "footnotes": [],
            "indexed_sources": [],
            "model_available": False,
        }

    prompt = _build_prompt(user_question, context_text)
    payload = {
        "model": ollama_model(),
        "prompt": prompt,
        "stream": False,
        "temperature": ollama_temperature(),
    }
    answer = ""
    llm_used = False
    try:
        with httpx.Client(timeout=ollama_timeout_seconds()) as client:
            response = client.post(f"{ollama_base_url()}/api/generate", json=payload)
            response.raise_for_status()
            answer = _coerce_text(response.json().get("response")).strip()
            llm_used = bool(answer)
    except Exception:
        answer = ""

    if not answer:
        answer = _deterministic_fallback(summary, user_question)
        llm_used = False

    return {
        "answer": answer.strip(),
        "footnotes": [],
        "indexed_sources": _evidence_sources(summary),
        "model_available": True,
        "summary": summary,
        "llm_used": llm_used,
    }


def _deterministic_fallback(summary: Dict[str, Any], user_question: str) -> str:
    """Plain text answer when the answer LLM is unreachable — never fabricates."""
    counts = summary.get("element_counts") or {}
    parts = [
        f"This is the '{summary.get('project') or summary.get('building') or 'building'}' "
        f"model ({summary.get('schema')}).",
    ]
    if counts:
        inv = ", ".join(f"{n} {label}" for label, n in counts.items())
        parts.append(f"It contains {summary.get('total_elements')} physical elements: {inv}.")
    storeys = summary.get("storeys") or []
    if storeys:
        names = ", ".join(str(s.get("name")) for s in storeys)
        parts.append(f"Levels: {names}.")
    return " ".join(parts)


async def stream_ifc_tokens(user_question: str) -> AsyncIterator[str]:
    """Stream an IFC-model answer as SSE token events."""
    path = ifc_model_path()
    try:
        context_text = build_ifc_context_text(path)
    except FileNotFoundError:
        msg = (
            "The building (IFC) model is not available on the server, so I can't "
            "answer questions about it right now."
        )
        yield f"data: {json.dumps({'event': 'token', 'text': msg})}\n\n"
        yield f"data: {json.dumps({'event': 'done'})}\n\n"
        return

    prompt = _build_prompt(user_question, context_text)
    payload = {
        "model": ollama_model(),
        "prompt": prompt,
        "stream": True,
        "temperature": ollama_temperature(),
    }
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
                    text = _coerce_text(event.get("response"))
                    if text:
                        yield f"data: {json.dumps({'event': 'token', 'text': text})}\n\n"
    except Exception:
        pass

    yield f"data: {json.dumps({'event': 'done'})}\n\n"
