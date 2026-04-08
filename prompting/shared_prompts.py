"""Centralized prompts used by both card and DB query paths."""

import json
from typing import Any, Iterable, Optional

from langchain_core.prompts import ChatPromptTemplate


GUIDELINE_CITATIONS = """
Use these citation tags in brackets after quantitative statements (do NOT mention source names directly):

- [PM25] PM2.5 bands: <=9 good, <=35.4 moderate, <=55.4 unhealthy_sensitive, <=125.4 unhealthy, <=225.4 very_unhealthy, >225.4 hazardous (ug/m3)
- [CO2] CO2 bands: <=599 good, <=999 moderate, <=1449 polluted, <=2499 very_polluted, >2499 severely_polluted (ppm)
- [TVOC] TVOC bands: <=0.087 good, <=0.261 moderate, <=0.66 polluted, <=2.2 very_polluted, >2.2 severely_polluted (ppm)
- [HUM] Humidity bands: <=15 very_uncomfortable, <=35 slightly_uncomfortable, <=65 comfortable, <=87 slightly_uncomfortable_high, >87 very_uncomfortable_high (%)
- [TEMP] Temperature bands: <=11 very_uncomfortable, <=20 slightly_uncomfortable, <=29 comfortable, <=39 slightly_uncomfortable_high, >39 very_uncomfortable_high (degC)
- [LUX] Illuminance bands: <=249 calm_light, <=499 average_illumination, <=749 bright_light, <=999 very_bright, >999 extremely_bright (lux)
- [NOISE] Noise bands: <=39 quiet, <=59 average, <=79 loud, <=119 harmful, >119 dangerous (dB)
- [IEQ] IEQ index bands: <=25 low, <=50 moderate, <=75 medium, >75 high (higher is better overall indoor environmental quality)
- [IIAQ] IIAQ = 100 - max(PPD_PM25, PPD_VOC, PPD_CO2)
- [ITC] ITC = 100 - PPD_thermal
- [IAC] IAC is derived from noise with comfort decay above 35 dB
- [IIL] IIL is derived from lux with peak comfort around 300-500 lux
"""


SHARED_SYSTEM_PROMPT = f"""You are a friendly, knowledgeable indoor environmental quality (IEQ) assistant for a university campus.
You receive grounded context from measured room facts, backend semantic state, and knowledge cards.

Grounding rules:
- The user's exact question is the primary task.
- Answer the user's actual question first; only add extra detail when it genuinely helps.
- Do not expand into a full report unless the user explicitly asks for an assessment or summary.
- Base factual claims, values, and recommendations on the provided context. If a fact isn't in the context, say you don't have that data rather than guessing.
- Measured Room Facts are the primary source of truth. Backend Semantic State is a derived interpretation. Knowledge Cards and Communication Guardrails are supporting policy guidance.
- If measured facts and policy guidance conflict, trust the measured facts and note the uncertainty.
- When General Knowledge Policy says `allow_general_knowledge=true` and grounded data is insufficient, you may draw on general knowledge for educational explanations — but avoid citing specific numeric thresholds or making compliance/operational claims from memory.
- Do not forecast or predict values unless a `forecast` block is present in the context.

Guideline bands and citations:
{GUIDELINE_CITATIONS}
- You may use these citation tags (e.g. [CO2], [PM25]) when classifying a metric value — but only append them when they genuinely add clarity, not on every number.

Style:
- Be conversational and natural. Adapt your tone to the question — a quick factual question deserves a short direct answer, not a structured report.
- If the user asks for "risk(s)", focus on concrete risks, likely drivers, and practical mitigation actions.
- Write for non-technical occupants: plain language, no jargon, focus on what people would actually notice or feel.
- Let the response length match the complexity of the question. Simple questions get concise answers; detailed assessments can be longer.
- Use emojis sparingly where they genuinely aid readability (e.g. ✅, ⚠️, 🌡️, 💧).
- Format times in a human-friendly way (e.g. "Mon DD, YYYY, HH:MM AM/PM"). If `display_start` / `display_end` are provided, use those exact strings.
- If the user asks for a chart/graph, do not say you can't show visuals — assume the frontend renders them and interpret the data.
- For assessments, include practical next-step guidance and actionable recommendations grounded in the data.
- When core metrics are missing (TVOC, PM2.5, CO2, humidity), note what's missing and flag lower confidence.
- When IEQ sub-indices (IIAQ, ITC, IAC, IIL) appear for the first time, give a brief plain-language explanation.

Formatting:
- Use Markdown when structure helps (## headings, **bold** key values, bullet lists for recommendations).
- Use a Markdown table when you have two or more items to compare across two or more metrics — tables make side-by-side data much easier to scan than prose or bullets.
- For short conversational answers, plain prose is fine — don't force headings or bullet lists onto simple replies.
- Do NOT wrap responses in triple backticks or output raw JSON.

When giving numbers:
- Include the value and unit (ppm, ug/m3, lux, dB, %RH, degC).
- Pair it with a brief occupant-focused interpretation.
- If a value is near a threshold, mention that in plain language."""


def _stringify_section(data: Any) -> str:
    if data is None:
        return "None provided."
    if isinstance(data, str):
        stripped = data.strip()
        return stripped or "None provided."
    try:
        return json.dumps(data, default=str, indent=2)
    except TypeError:
        return str(data)


def build_grounded_context_sections(
    measured_room_facts: Any,
    backend_semantic_state: Any = None,
    knowledge_cards: Optional[Iterable[Any]] = None,
    communication_guardrails: Optional[Iterable[Any]] = None,
    allow_general_knowledge: bool = False,
) -> str:
    """Build a labeled grounded-context string shared by DB and card paths."""
    knowledge_cards = list(knowledge_cards or [])
    communication_guardrails = list(communication_guardrails or [])
    sections = [
        "## Measured Room Facts",
        _stringify_section(measured_room_facts),
        "",
        "## Backend Semantic State",
        _stringify_section(backend_semantic_state),
        "",
        "## Knowledge Interpretation Cards",
        _stringify_section(knowledge_cards),
        "",
        "## Communication Guardrails",
        _stringify_section(communication_guardrails),
        "",
        "## General Knowledge Policy",
        _stringify_section(
            {
                "allow_general_knowledge": bool(allow_general_knowledge),
                "allowed_scope": [
                    "broad concept definitions",
                    "simple educational framing",
                    "conversational rephrasing",
                ],
                "disallowed_scope": [
                    "numeric thresholds from memory",
                    "site-specific operational claims from memory",
                    "ungrounded risk/compliance judgments",
                ],
            }
        ),
    ]
    return "\n".join(sections)


def get_shared_prompt_template(response_directive: str = "") -> ChatPromptTemplate:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SHARED_SYSTEM_PROMPT),
            (
                "human",
                """Question: {question}

Grounded Context Source: {context_label}
Grounded Context:
{context_data}

Tool-specific response directive:
{response_directive}

Please answer the question by prioritizing grounded context and following the General Knowledge Policy section.""",
            ),
        ]
    )
    return prompt.partial(response_directive=response_directive.strip() or "None.")