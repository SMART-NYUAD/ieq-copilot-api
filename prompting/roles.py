"""Stakeholder roles — who the answer is being written for.

A role is a property of the *person asking*, not of the question. That is what makes it
different from every other signal in this system: ``intent``, ``analysis_mode`` and
``metric_scope`` are all derived from the question text by the router, and they must be,
because only the wording can reveal them. A role cannot be read off the wording — the same
sentence ("how is the air quality?") comes from an occupant, an operator, an analyst and a
director, and each wants a different answer. So the role is declared by the caller and the
router is never asked to guess it.

The vocabulary is closed and lives in code, for the same reason the viewer/heatmap/download
alias maps do (see ``llm_router_planner``): a closed set maps deterministically from a
client's selection, stays testable, and cannot be invented at runtime.

``ROLE_OCCUPANT`` is the default because it is the audience this system was already aimed
at: ``SHARED_SYSTEM_PROMPT`` hardcoded "Write for non-technical occupants: plain language,
no jargon", so the assistant had exactly one persona and it just was not selectable.

The occupant *block* is no longer that wording, though. "No jargon" constrained vocabulary
and said nothing about volume, so an occupant still received every fetched pollutant with
its value and unit — a rundown indistinguishable from the researcher's. Fixing that meant
changing the default, which changes behavior for callers who send no role. That is
intended; see ``prompting/role_prompts.py``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


ROLE_OCCUPANT = "occupant"
ROLE_FACILITY_MANAGER = "facility_manager"
ROLE_RESEARCHER = "researcher"
ROLE_EXECUTIVE = "executive"

ROLE_DEFAULT = ROLE_OCCUPANT

VALID_ROLES: Tuple[str, ...] = (
    ROLE_OCCUPANT,
    ROLE_FACILITY_MANAGER,
    ROLE_RESEARCHER,
    ROLE_EXECUTIVE,
)

# Human-facing labels and blurbs, served by ``GET /roles`` so a client does not hardcode a
# vocabulary that will drift away from this file.
ROLE_LABELS: Dict[str, str] = {
    ROLE_OCCUPANT: "Occupant",
    ROLE_FACILITY_MANAGER: "Facility Manager",
    ROLE_RESEARCHER: "Researcher",
    ROLE_EXECUTIVE: "Executive",
}

ROLE_DESCRIPTIONS: Dict[str, str] = {
    ROLE_OCCUPANT: "Plain language, focused on what you would actually notice or feel.",
    ROLE_FACILITY_MANAGER: "Operational detail: what is out of range, where, and what to do about it.",
    ROLE_RESEARCHER: "Exact values, units, reading age, and every applicable threshold with its source.",
    ROLE_EXECUTIVE: "A short headline verdict framed as compliance and risk.",
}

# Aliases a client might plausibly send. Unknown values are not rejected — see
# :func:`normalize_role` — so this map only needs to cover the likely spellings.
_ALIASES: Dict[str, str] = {
    "fm": ROLE_FACILITY_MANAGER,
    "facility": ROLE_FACILITY_MANAGER,
    "facilities": ROLE_FACILITY_MANAGER,
    "facility-manager": ROLE_FACILITY_MANAGER,
    "facilities_manager": ROLE_FACILITY_MANAGER,
    "operator": ROLE_FACILITY_MANAGER,
    "operations": ROLE_FACILITY_MANAGER,
    "ops": ROLE_FACILITY_MANAGER,
    "building_manager": ROLE_FACILITY_MANAGER,
    "analyst": ROLE_RESEARCHER,
    "research": ROLE_RESEARCHER,
    "scientist": ROLE_RESEARCHER,
    "data_analyst": ROLE_RESEARCHER,
    "exec": ROLE_EXECUTIVE,
    "executive_summary": ROLE_EXECUTIVE,
    "director": ROLE_EXECUTIVE,
    "leadership": ROLE_EXECUTIVE,
    "general": ROLE_OCCUPANT,
    "default": ROLE_OCCUPANT,
    "user": ROLE_OCCUPANT,
    "resident": ROLE_OCCUPANT,
    "employee": ROLE_OCCUPANT,
}


def normalize_role(value: Optional[str]) -> Tuple[Optional[str], bool]:
    """Resolve a caller-supplied role to the canonical vocabulary.

    Returns ``(role, was_fallback)``. ``role`` is ``None`` when the caller supplied
    nothing at all, which is how the resolution order distinguishes "not sent" (fall
    through to the conversation's stored role, then the configured default) from "sent
    something we did not understand".

    An unrecognised non-empty value resolves to ``None`` with ``was_fallback=True`` rather
    than raising. A wrong role should degrade to the default voice, not fail the query —
    but it is echoed as ``role_fallback_used`` in the response metadata so a client bug
    stays visible instead of silently looking like a preference.
    """
    raw = str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
    if not raw:
        return None, False
    if raw in VALID_ROLES:
        return raw, False
    alias = _ALIASES.get(raw)
    if alias:
        return alias, False
    return None, True


def is_valid_role(value: Optional[str]) -> bool:
    return str(value or "") in VALID_ROLES


def coerce_role(value: Optional[str]) -> str:
    """A guaranteed-valid role, for prompt builders that must not fail on bad input."""
    return value if is_valid_role(value) else ROLE_DEFAULT


def role_catalog() -> List[Dict[str, Any]]:
    """The vocabulary as the ``GET /roles`` payload."""
    return [
        {
            "id": role,
            "label": ROLE_LABELS[role],
            "description": ROLE_DESCRIPTIONS[role],
            "default": role == ROLE_DEFAULT,
        }
        for role in VALID_ROLES
    ]


# Roles that want compliance detail — threshold figures, standards bodies, index acronyms —
# in the computed Threshold Assessment they are shown. The others get the same verdicts
# rendered in plain language with the metrics that are fine collapsed into one sentence.
# This is a rendering choice, never a verdict change: what a metric's status IS does not
# depend on who is reading.
_COMPLIANCE_DETAIL_ROLES = frozenset({ROLE_RESEARCHER, ROLE_FACILITY_MANAGER})


def role_wants_compliance_detail(role: Optional[str]) -> bool:
    """Whether this reader should see threshold numbers and standards names."""
    return coerce_role(role) in _COMPLIANCE_DETAIL_ROLES
