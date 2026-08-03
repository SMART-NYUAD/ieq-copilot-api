"""Per-role audience blocks — the only place a role changes what the model is told.

Each block replaces the two audience bullets that used to be hardcoded under ``Domain
style:`` in ``SHARED_SYSTEM_PROMPT``. It is spliced into the *system prompt* and nowhere
else: the DB response directives, the presentation style block and the metric-completeness
rules are all untouched by role. That single-point rule is deliberate — the advisory bug
recorded in CLAUDE.md was caused by four separate prompts repeating the same instruction
until the model followed none of them, and a role block appended to both the system prompt
and the directive would be the start of the same pattern.

The occupant block is the previous wording **verbatim**, with no invariant clause appended,
so ``shared_system_prompt(ROLE_OCCUPANT)`` is byte-identical to the old constant. That
makes the default path a provable no-op. The clause is carried by the three new roles,
which are the actual new risk: each of them pushes on length or vocabulary, and a model
told to be brief for an executive must still not drop the pollutant that failed.
"""

from __future__ import annotations

from prompting.roles import (
    ROLE_EXECUTIVE,
    ROLE_FACILITY_MANAGER,
    ROLE_OCCUPANT,
    ROLE_RESEARCHER,
    coerce_role,
)


# Attached to every role block except the default. Role is allowed to change how something
# is said and how much of it is elaborated; it is never a licence to leave a metric out or
# to restate a verdict the threshold assessment already computed.
_INVARIANT = (
    "- These audience rules govern wording, emphasis and length only. They never permit "
    "dropping a metric that was fetched, softening or restating a verdict from the "
    "Threshold Assessment section, or omitting a citation. If audience brevity and metric "
    "completeness conflict, completeness wins."
)


_OCCUPANT = """- Prefer natural, compassionate phrasing over clinical/policy-heavy wording unless the user explicitly asks for formal compliance language.
- Write for non-technical occupants: plain language, no jargon, focus on what people would actually notice or feel."""


_FACILITY_MANAGER = f"""- You are answering a facility manager who operates this building. Lead with the operational fact: which metric is out of range, in which space, and since when.
- Use operational vocabulary freely (ventilation rate, setpoint, air changes, filtration, occupancy load). Do not expand common acronyms — this reader knows them.
- When a metric is over a limit, name the limit and the standard that publishes it, exactly as the Threshold Assessment section gives them.
- Prefer specifics over reassurance: a named space, a device, a time of day, or a measured trend beats a general statement.
- You may use up to 4 bullets and about 150 words, which supersedes the shorter default cap in the presentation style rules.
- State the operational implication of what you found in one clause. Do not write a full recommendations section unless the user asked for advice, actions, or next steps.
{_INVARIANT}"""


_RESEARCHER = f"""- You are answering an analyst who will work with these numbers. Do not simplify, round away precision, or substitute a qualitative word for a measured value.
- Give every value with its unit, and state the time window and aggregation interval the values came from. Where a reading's age is known, state it.
- Report every applicable threshold with its source and its unit, including where several standards disagree. When a metric is unrated, say so and give the reason (typically: no threshold published in the unit the sensor reports).
- Name the limits of the data rather than smoothing over them: missing metrics, stale readings, single-sensor coverage, derived rather than measured figures.
- Do not use emoji. Keep bold for the headline verdict only.
- You may use up to about 200 words and as many bullets or table rows as the data needs, which supersedes the shorter default cap in the presentation style rules.
{_INVARIANT}"""


_EXECUTIVE = f"""- You are answering a decision maker who needs the conclusion, not the readings. Open with one headline verdict and frame it as compliance and risk.
- Expand or avoid acronyms and index names on first use (say "air quality index (IAQ)", not "IAQ"). Never name individual sensors or devices.
- Still say which metrics failed and by how much — brevity here means fewer words about the same facts, not fewer facts. A metric that is over its limit must appear in the answer even if nothing else does.
- Keep the whole answer under 60 words, with at most 2 bullets or a short 2-column table.
- Do not include operational instructions unless the user asked what to do.
{_INVARIANT}"""


_ROLE_BLOCKS = {
    ROLE_OCCUPANT: _OCCUPANT,
    ROLE_FACILITY_MANAGER: _FACILITY_MANAGER,
    ROLE_RESEARCHER: _RESEARCHER,
    ROLE_EXECUTIVE: _EXECUTIVE,
}


def role_style_block(role: str = ROLE_OCCUPANT) -> str:
    """The audience bullets for ``role``. Unknown roles fall back to the default block."""
    return _ROLE_BLOCKS[coerce_role(role)]


def role_addendum(role: str = ROLE_OCCUPANT) -> str:
    """The audience bullets to *append* to a prompt that already reads as occupant-facing.

    ``IFC_SYSTEM_PROMPT`` and ``SENSOR_SYSTEM_PROMPT`` are hand-written and already carry
    plain-language audience lines, so unlike the shared system prompt there is nothing to
    replace. Returning ``""`` for the default role keeps those two prompts byte-identical
    to their previous form — the default has nothing to add that they do not already say.
    """
    resolved = coerce_role(role)
    if resolved == ROLE_OCCUPANT:
        return ""
    return "\nAudience:\n" + _ROLE_BLOCKS[resolved] + "\n"
