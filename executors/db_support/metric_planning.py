"""Which metrics a question needs, and how many of them may be fetched.

One question resolves to one :class:`MetricPlan`, and every handler consumes it the same
way (``plan.selected``). Previously the pack was chosen here and then re-decided by each
handler with its own hardcoded slice — thirteen call sites applying ``[:4]``/``[:5]``/
``[:8]``, several of which cut a pack below its own length. A comfort comparison, for
example, was capped at 8 and silently lost ``sound`` and ``light``: the two metrics that
make it a *comfort* assessment rather than an air-quality one.

Two rules make that unrepresentable:

* the limit travels with the pack, so a pack can never be truncated below itself;
* ``metrics`` is priority-ordered — the limit decides by that order, so the ordering of a
  pack is a deliberate statement about what matters most, not incidental.

The scope is chosen by the router (``RoutePlan.metric_scope``) when it is available;
:func:`classify_metric_scope` reproduces the decision from question text for the emergency
path where the router LLM was unreachable. Same arrangement as ``analysis_mode``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from executors import metric_registry
from executors.db_support import response_helpers as db_helpers
from query_routing.intent_classifier import IntentType


# Scope names. These are the router's closed vocabulary — it never returns metric names
# directly, so it cannot invent a metric, and scope→metrics stays a tested decision here.
SCOPE_NAMED = "named"                # only what the user (or planner) named
SCOPE_AIR_QUALITY = "air_quality"    # pollutant pack + IEQ
SCOPE_IEQ_INDEX = "ieq_index"        # IEQ composite + its four sub-indices
SCOPE_COMFORT = "comfort"            # everything an occupant would *feel*
SCOPE_DIAGNOSTIC = "diagnostic"      # root-cause: every contributing metric
SCOPE_FULL = "full"                  # explicit "give me everything" asks

VALID_METRIC_SCOPES = frozenset(
    {
        SCOPE_NAMED,
        SCOPE_AIR_QUALITY,
        SCOPE_IEQ_INDEX,
        SCOPE_COMFORT,
        SCOPE_DIAGNOSTIC,
        SCOPE_FULL,
    }
)

# Priority-ordered metric packs. Order matters: it is what a limit cuts by.
_PACKS: Dict[str, List[str]] = {
    # Pollutants first, then humidity and the composite that summarises them.
    SCOPE_AIR_QUALITY: ["co2", "pm25", "voc", "humidity", "ieq"],
    # The composite and the four sub-indices that explain it. Never the pollutant pack —
    # "what is the IEQ?" wants the score breakdown, not a list of gas concentrations.
    SCOPE_IEQ_INDEX: ["ieq", "iaq", "itc", "iac", "iil"],
    # Air, thermal, acoustic and visual comfort. `sound` and `light` are the point of this
    # pack; a limit below its length would remove exactly what distinguishes it.
    SCOPE_COMFORT: [
        "ieq", "itc", "iaq", "temperature", "humidity", "co2", "pm25", "voc", "sound", "light",
    ],
    # Root-cause decomposition: every metric that can contribute to the index.
    SCOPE_DIAGNOSTIC: ["co2", "pm25", "voc", "humidity", "temperature", "ieq", "sound", "light"],
    SCOPE_FULL: ["ieq", "co2", "pm25", "voc", "humidity", "temperature", "sound", "light"],
}

# Ceiling for a user-named metric list. Packs carry their own limit; this only stops a
# planner hint list from fanning out unboundedly.
_NAMED_SCOPE_LIMIT = 6

_POLLUTANTS = {"co2", "pm25", "voc"}
_IEQ_SUB_INDICES = ("iaq", "itc", "iac", "iil")

_ANALYTICAL_INTENTS = {
    IntentType.AGGREGATION_DB,
    IntentType.COMPARISON_DB,
    IntentType.ANOMALY_ANALYSIS_DB,
}

_TREND_PHRASES = (
    "trend",
    "trended",
    "over time",
    "this week",
    "last week",
    "this month",
    "last month",
    "past ",
    "last ",
)

_COMPARISON_PHRASES = ("compare", "vs", "versus")

_FULL_ASSESSMENT_PHRASES = (
    "complete assessment",
    "full assessment",
    "full picture",
    "everything you have",
    "environmental assessment",
)


@dataclass(frozen=True)
class MetricPlan:
    """The metrics a question resolves to, in priority order, plus how many to fetch."""

    scope: str
    metrics: List[str]
    limit: int
    # True when the scope came from question text because the router did not supply one.
    scope_inferred: bool = False

    @property
    def selected(self) -> List[str]:
        """The metrics a handler should actually fetch."""
        return self.metrics[: self.limit]


def is_full_assessment_query(question: str) -> bool:
    """Explicit "tell me everything" phrasing."""
    q = str(question or "").lower()
    return any(phrase in q for phrase in _FULL_ASSESSMENT_PHRASES)


def with_ieq_sub_indices(metrics: List[str]) -> List[str]:
    """Append the IEQ sub-indices when the composite is present.

    A snapshot of the IEQ score is a bare number; the sub-indices are what let the answer
    say *why* it is what it is. Only the point-lookup path does this — an aggregate row
    reports the composite over a window, where the sub-index breakdown does not apply the
    same way.
    """
    expanded = list(metrics)
    if "ieq" in expanded:
        for sub in _IEQ_SUB_INDICES:
            if sub not in expanded:
                expanded.append(sub)
    return expanded


def _named_metrics(
    explicit_metrics: List[str],
    hinted_metrics: List[str],
    intent: IntentType,
) -> List[str]:
    """Merge the metrics the user named with the planner's, keeping a stable order."""
    explicit = list(explicit_metrics or [])
    hinted = list(hinted_metrics or [])
    # An analytical question whose planner hints are broader than what the user spelled out
    # is being deliberately widened by the planner, so the hinted order leads.
    if explicit and hinted and len(hinted) > len(explicit) and intent in _ANALYTICAL_INTENTS:
        return hinted + [m for m in explicit if m not in hinted]
    return explicit + [m for m in hinted if m not in explicit]


def _explicit_scope_is_respected(
    question: str,
    explicit_metrics: List[str],
    hinted_metrics: List[str],
    intent: IntentType,
    is_diagnostic: bool,
    is_comfort: bool,
) -> bool:
    """Whether metrics the user named should be answered as-is, without a pack.

    Naming a metric usually means wanting that metric. The exceptions are questions whose
    phrasing asks for a comparison or trend *around* a pollutant, where answering with the
    single named metric would omit the context that makes the answer meaningful.
    """
    if not explicit_metrics:
        return False
    q = str(question or "").lower()
    names_pollutant = any(m in _POLLUTANTS for m in explicit_metrics)
    planner_widened = (
        bool(hinted_metrics)
        and len(hinted_metrics) > len(explicit_metrics)
        and intent in _ANALYTICAL_INTENTS
    )
    widening_case = (
        is_diagnostic
        or planner_widened
        or (intent == IntentType.COMPARISON_DB and is_comfort)
        or (
            intent == IntentType.COMPARISON_DB
            and names_pollutant
            and any(token in q for token in _COMPARISON_PHRASES)
        )
        or (
            intent == IntentType.AGGREGATION_DB
            and names_pollutant
            and any(token in q for token in _TREND_PHRASES)
        )
    )
    return not widening_case


def _reads_as_air_quality(
    question: str,
    explicit_metrics: List[str],
    named_metrics: List[str],
    intent: IntentType,
) -> bool:
    """Air-quality phrasing, plus the pollutant comparisons/trends that imply it."""
    if db_helpers.is_air_quality_query_text(question):
        return True
    q = str(question or "").lower()
    if (
        intent == IntentType.COMPARISON_DB
        and any(m in _POLLUTANTS for m in named_metrics)
        and any(token in q for token in _COMPARISON_PHRASES)
    ):
        return True
    return (
        intent == IntentType.AGGREGATION_DB
        and len(explicit_metrics) == 1
        and any(m in _POLLUTANTS for m in explicit_metrics)
        and any(token in q for token in _TREND_PHRASES)
    )


def classify_metric_scope(
    question: str,
    explicit_metrics: List[str],
    hinted_metrics: List[str],
    intent: IntentType,
    is_diagnostic: bool = False,
) -> str:
    """Infer the metric scope from question text.

    The emergency path: used when the router did not supply a scope (LLM unreachable, or a
    direct caller). Keyword matching handles the phrasings it was written for and nothing
    else, which is exactly why the router owns this decision when it is available.
    """
    if is_full_assessment_query(question):
        return SCOPE_FULL

    is_comfort = db_helpers.is_comfort_assessment_query_text(question)
    is_air_quality_text = db_helpers.is_air_quality_query_text(question)

    # An IEQ-index ask reports the score family, not the pollutant pack.
    if (
        db_helpers.is_ieq_index_query_text(question)
        and not is_air_quality_text
        and not is_comfort
        and not is_diagnostic
    ):
        return SCOPE_IEQ_INDEX

    if _explicit_scope_is_respected(
        question=question,
        explicit_metrics=explicit_metrics,
        hinted_metrics=hinted_metrics,
        intent=intent,
        is_diagnostic=is_diagnostic,
        is_comfort=is_comfort,
    ):
        return SCOPE_NAMED

    if is_diagnostic:
        return SCOPE_DIAGNOSTIC

    named = _named_metrics(explicit_metrics, hinted_metrics, intent)
    if _reads_as_air_quality(question, explicit_metrics, named, intent):
        return SCOPE_COMFORT if is_comfort else SCOPE_AIR_QUALITY
    if is_comfort:
        return SCOPE_COMFORT
    return SCOPE_NAMED


def plan_metrics(
    question: str,
    explicit_metrics: List[str],
    hinted_metrics: List[str],
    intent: IntentType,
    is_diagnostic: bool = False,
    metric_scope: Optional[str] = None,
) -> MetricPlan:
    """Resolve one question to the metrics to fetch and the limit that applies.

    ``metric_scope`` is the router's decision; anything absent or unrecognised falls back
    to :func:`classify_metric_scope`.
    """
    supplied_scope = str(metric_scope or "").strip().lower()
    scope_inferred = supplied_scope not in VALID_METRIC_SCOPES
    scope = (
        classify_metric_scope(
            question=question,
            explicit_metrics=explicit_metrics,
            hinted_metrics=hinted_metrics,
            intent=intent,
            is_diagnostic=is_diagnostic,
        )
        if scope_inferred
        else supplied_scope
    )
    if is_diagnostic:
        # A root-cause question needs every contributing metric, so the diagnostic pack
        # outranks a narrower scope. The router routinely returns both signals together —
        # "why is the IEQ low?" is `ieq_index` *and* diagnostic — and answering it from the
        # index family alone would report the score without the pollutants driving it.
        scope = SCOPE_DIAGNOSTIC

    named = _named_metrics(explicit_metrics, hinted_metrics, intent)
    pack = _PACKS.get(scope)
    if pack is None:
        metrics, limit = named, _NAMED_SCOPE_LIMIT
    else:
        # Named metrics trail the pack so they are visible to callers, but the limit is the
        # pack's own length: a pack is a complete answer, not a starting point.
        metrics, limit = pack + [m for m in named if m not in pack], len(pack)

    # Drop anything the metric registry cannot resolve to a column, before the limit is
    # applied, so an unknown alias costs the answer a slot rather than a metric.
    metrics = [m for m in metrics if metric_registry.metric_column(m) is not None]
    return MetricPlan(scope=scope, metrics=metrics, limit=limit, scope_inferred=scope_inferred)
