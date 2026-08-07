"""Compare readings against their guideline thresholds, in code.

The answer model cannot be trusted to do this comparison in prose. Measured on real
answers it got the *direction* roughly right and the *attribution* wrong, and when
pushed harder it invented numbers outright:

  * PM2.5 17.2 ug/m3 reported as "above the EPA daily standard" -- EPA's is 35; it is
    WHO's 15 that is exceeded.
  * VOC 1.25 ppm reported as "0.64 ppm, below RESET Air Grade A threshold of 0.8 ppm".
    RESET Grade A is 0.102 ppm in this unit, so the reading is over twelve times the
    threshold, and neither the value nor the limit in that sentence was real.
  * IAQ 0.0 -- the worst value on a 0-100 higher-is-better scale -- described as
    "consistent with low pollutant levels", inverting the scale it was given.

So the comparison moves out of the prompt: this module resolves, per metric, which
threshold applies, whether the reading exceeds it, and which citation index carries
it. The model receives verdicts and spends its judgement on wording instead of
arithmetic. Same move ``metric_planning`` made when metric selection stopped being a
paragraph and became a table.

Two rules matter for correctness:

* A threshold only applies if it is expressed in the unit the reading uses. VOC reads
  in ppm and most published TVOC limits are in ug/m3; comparing across them needs a
  molar-mass assumption, so a metric with no threshold in its own unit is reported
  ``unrated`` rather than guessed at.
* When several sources cover one metric, the strictest applicable one governs, so a
  clean verdict cannot be bought by quoting the most permissive standard.
* An **indoor** reading is graded against an indoor standard when one exists. WHO's Global
  Air Quality Guidelines and EPA's NAAQS are *ambient* (outdoor) standards — WHO's own
  record says so in its caveat — and grading a lab against them reports the room as failing
  an obligation nobody placed on it. RESET Air and WELL publish limits for the occupied
  interior; those govern here, and the ambient figures are used only when nothing indoor
  covers the metric, with the substitution stated on the verdict line.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from executors import metric_registry


# Internal 0-100 index bands, higher = better. These mirror the bands stated in
# SHARED_SYSTEM_PROMPT; they live here so the classification is computed once rather
# than re-derived in prose by a model that has already been observed inverting it.
_INDEX_BANDS = (
    (75.0, "high", "high quality"),
    (50.0, "medium", "medium quality"),
    (25.0, "moderate", "moderate quality"),
    (0.0, "low", "low quality"),
)

# Metrics reported on the 0-100 index scale rather than as a concentration.
_INDEX_METRICS = {"ieq", "iaq", "itc", "iac", "iil"}

# A reading within this fraction of its threshold is called out as approaching it,
# so "fine" and "about to not be fine" do not read identically.
_NEAR_FRACTION = 0.9

STATUS_EXCEEDS = "EXCEEDS"
STATUS_NEAR = "NEAR"
STATUS_WITHIN = "within"
STATUS_UNRATED = "unrated"

# Index metrics get their own vocabulary: nothing is "exceeded" when a score is low,
# and calling it EXCEEDS invites the model to describe a 0/100 as a breach of a limit
# rather than as the worst possible score.
STATUS_POOR = "POOR"
STATUS_FAIR = "FAIR"
STATUS_GOOD = "GOOD"

# Beyond the edge of an "optimal" band (threshold_type range_max/range_min) rather than
# past a hard limit. 54.7 %RH is above EPA's 50 % comfort-range top but well under
# ASHRAE's 65 % limit, and reporting that as an exceedance would flag a normal room.
STATUS_OUTSIDE_BAND = "outside optimal range"

# threshold_type values that denote a hard limit rather than the edge of a comfort band.
_HARD_LIMIT_TYPES = {"max", "min"}

# Sources that publish limits for OUTDOOR ambient air. Every reading this system takes is
# indoors, so these only decide a metric that no indoor standard covers — and when they do,
# the verdict line says the figure is an ambient one.
#
# This is a property of the guideline record, not of the reading, so it is keyed by
# source_key rather than inferred from wording: both of these records already say "outdoor
# ambient" in their own caveat text, which the assessment never sees. Adding an indoor
# standard for a metric is therefore enough to demote the ambient one automatically.
_AMBIENT_SOURCE_KEYS = frozenset(
    {
        "WHO_AQG_2021",          # WHO Global Air Quality Guidelines 2021 — ambient PM2.5
        "EPA_PM25_NAAQS_2024",   # EPA NAAQS — ambient PM2.5, explicitly outdoor
    }
)


def _is_ambient(source: Dict[str, Any]) -> bool:
    return str(source.get("source_key") or "").strip().upper() in _AMBIENT_SOURCE_KEYS


# citation_tier precedence. `regulatory` is a published standard a building is assessed
# against; `research` is a study finding or a guide value; `internal` is this system's own
# composite. An unknown tier sorts with research rather than being promoted.
_TIER_RANK = {"regulatory": 0, "research": 1, "internal": 1}


def _tier_rank(source: Dict[str, Any]) -> int:
    return _TIER_RANK.get(str(source.get("citation_tier") or "").strip().lower(), 1)


# Ranked worst-first, for picking the headline status.
_STATUS_RANK = {
    STATUS_EXCEEDS: 0,
    STATUS_POOR: 0,
    STATUS_NEAR: 1,
    STATUS_FAIR: 1,
    STATUS_OUTSIDE_BAND: 1,
    STATUS_WITHIN: 2,
    STATUS_GOOD: 2,
    STATUS_UNRATED: 3,
}


def _normalize_unit(unit: Any) -> str:
    """Fold the unit spellings that differ only by codepoint or casing.

    The registry writes PM2.5 as ``μg/m³`` (U+03BC, Greek mu) while the guideline seed
    writes ``µg/m³`` (U+00B5, micro sign). They render identically and a naive equality
    check silently decides the metric has no applicable threshold.
    """
    text = str(unit or "").strip().lower()
    text = text.replace("μ", "µ")           # greek mu -> micro sign
    text = text.replace("³", "3").replace("^3", "3")
    text = text.replace("ug/m3", "µg/m3")
    text = text.replace(" ", "")
    # Humidity is "%" in the registry and "percent RH" in the guideline seed.
    if text in ("%", "%rh", "percent", "percentrh", "percent_rh", "rh"):
        return "%"
    return text


@dataclass(frozen=True)
class MetricAssessment:
    metric: str
    display: str
    value: float
    unit: str
    status: str
    threshold_value: Optional[float] = None
    threshold_unit: Optional[str] = None
    source_index: Optional[int] = None
    source_label: Optional[str] = None
    band: Optional[str] = None
    band_label: Optional[str] = None
    note: Optional[str] = None
    # The averaging basis / qualifying condition the governing threshold is published under
    # ("24-hour mean guideline", "Grade A, occupied hours"). Carried through to the rendered
    # line because a limit quoted under the wrong basis is a wrong limit: a "today" answer
    # citing WHO's ANNUAL mean is comparing one day against a year.
    threshold_condition: Optional[str] = None
    # True when no indoor standard covered this metric and an outdoor ambient one was used.
    ambient_basis: bool = False

    @property
    def is_index(self) -> bool:
        return self.metric in _INDEX_METRICS


def _classify_index(value: float) -> tuple[str, str]:
    for floor, band, label in _INDEX_BANDS:
        if value > floor:
            return band, label
    return "low", "low quality"


def _applicable_sources(
    metric: str, reading_unit: str, indexed_sources: Iterable[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Sources for this metric carrying a threshold in the reading's own unit."""
    target = _normalize_unit(reading_unit)
    out: List[Dict[str, Any]] = []
    for source in indexed_sources or []:
        if str(source.get("metric") or "").strip().lower() != metric:
            continue
        if source.get("threshold_value") is None:
            continue
        if _normalize_unit(source.get("threshold_unit")) != target:
            continue
        out.append(source)
    return out


def _has_any_threshold(metric: str, indexed_sources: Iterable[Dict[str, Any]]) -> bool:
    return any(
        str(s.get("metric") or "").strip().lower() == metric
        and s.get("threshold_value") is not None
        for s in indexed_sources or []
    )


def assess_metric(
    metric: str,
    value: Any,
    indexed_sources: Iterable[Dict[str, Any]],
) -> Optional[MetricAssessment]:
    canonical = metric_registry.resolve_metric(metric)
    spec = metric_registry.METRICS.get(canonical)
    if spec is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None

    display = str(spec.get("display") or canonical.upper())
    unit = str(spec.get("unit") or "")

    if canonical in _INDEX_METRICS:
        band, band_label = _classify_index(numeric)
        # A sub-index is not "within a threshold"; it is a score. Anything below the
        # high band is surfaced so it cannot be waved through as fine.
        status = {
            "high": STATUS_GOOD,
            "medium": STATUS_FAIR,
            "moderate": STATUS_POOR,
            "low": STATUS_POOR,
        }[band]
        return MetricAssessment(
            metric=canonical, display=display, value=numeric, unit="/100",
            status=status, band=band, band_label=band_label,
            note="0-100 scale, higher is better",
        )

    sources = _applicable_sources(canonical, unit, indexed_sources)
    if not sources:
        note = None
        if _has_any_threshold(canonical, indexed_sources):
            note = (
                f"thresholds available for this metric are not expressed in {unit}, "
                "so no direct comparison is possible"
            )
        else:
            note = "no published threshold provided in the citation sources"
        return MetricAssessment(
            metric=canonical, display=display, value=numeric, unit=unit,
            status=STATUS_UNRATED, note=note,
        )

    # Indoor standards first. Ambient limits only grade a metric no indoor source covers,
    # and the fact that they did is recorded so the answer can say which world the figure
    # comes from.
    indoor = [s for s in sources if not _is_ambient(s)]
    ambient_basis = not indoor
    sources = indoor or sources

    # A published standard outranks a research guide value. RESET Air Grade A and WELL v2
    # set 0.102 ppm for TVOC; Seifert's hygienic band edge is 0.061 ppm, so strictest-wins
    # alone made a *hygienic* figure — explicitly not a health-based limit, and not what a
    # building is held to — govern every VOC verdict ahead of the two standards the space
    # is actually assessed against. Research records stay in the pool and still govern a
    # metric no standard covers; they just stop outranking one that does.
    regulatory = [s for s in sources if _tier_rank(s) == 0]
    sources = regulatory or sources

    # A hard limit outranks a comfort-band edge: exceeding ASHRAE's 65 %RH limit is a
    # finding, drifting past EPA's 50 % "optimal range" top is not. Only when a metric
    # has no hard limit at all does the band edge decide, and then it is reported as
    # being outside the optimal range rather than over a limit.
    hard = [
        s for s in sources
        if str(s.get("threshold_type") or "").strip().lower() in _HARD_LIMIT_TYPES
    ]
    graded = hard or sources
    strictest = min(graded, key=lambda s: float(s["threshold_value"]))
    threshold = float(strictest["threshold_value"])
    if not hard:
        status = STATUS_OUTSIDE_BAND if numeric > threshold else STATUS_WITHIN
    elif numeric > threshold:
        status = STATUS_EXCEEDS
    elif threshold > 0 and numeric >= threshold * _NEAR_FRACTION:
        status = STATUS_NEAR
    else:
        status = STATUS_WITHIN

    return MetricAssessment(
        metric=canonical, display=display, value=numeric, unit=unit, status=status,
        threshold_value=threshold, threshold_unit=strictest.get("threshold_unit"),
        source_index=strictest.get("index"), source_label=strictest.get("source_label"),
        threshold_condition=(str(strictest.get("threshold_condition") or "").strip() or None),
        ambient_basis=ambient_basis,
    )


def assess_readings(
    readings: Dict[str, Any],
    indexed_sources: Iterable[Dict[str, Any]],
) -> List[MetricAssessment]:
    """Assess every readable metric, worst status first."""
    sources = list(indexed_sources or [])
    out: List[MetricAssessment] = []
    for metric, value in (readings or {}).items():
        assessment = assess_metric(metric, value, sources)
        if assessment is not None:
            out.append(assessment)
    out.sort(key=lambda a: (_STATUS_RANK.get(a.status, 9), a.metric))
    return out


def _format_value(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 1:
        return f"{value:.1f}"
    return f"{value:.3f}".rstrip("0").rstrip(".")


# Plain-language labels for readers who should never see an index acronym. The section is
# labelled authoritative and says "state them as given", so whatever appears here is what
# the model writes — an occupant asked "how is the air?" and got "IAQ Sub-index is 94.0"
# because that is the string it was handed.
_PLAIN_LABELS = {
    "ieq": "overall comfort score",
    "iaq": "air quality score",
    "itc": "thermal comfort score",
    "iac": "noise comfort score",
    "iil": "lighting score",
}

# Verdict phrasing without the threshold figure or the publishing body. The citation marker
# still carries the source, so nothing is lost for a reader who wants it — the frontend
# renders [N].
_PLAIN_VERBS = {
    STATUS_EXCEEDS: "is above what is recommended",
    STATUS_NEAR: "is close to the recommended limit",
    STATUS_WITHIN: "is within the recommended range",
    STATUS_OUTSIDE_BAND: "is outside the ideal range, though not over a hard limit",
}


def _plain_display(a: "MetricAssessment") -> str:
    return _PLAIN_LABELS.get(a.metric, a.display)


def _render_plain(assessments: List[MetricAssessment]) -> List[str]:
    """One line per metric that needs attention; everything else in a single sentence.

    This is what makes an audience-scoped answer possible at all. The full renderer emits a
    line per metric carrying a threshold number and a standards body, and the model is
    instructed to state those as given — so no prompt wording could stop an occupant answer
    reciting "0.061 ppm (WHO Indoor Air Quality Guidelines)" for six metrics. Selecting and
    phrasing here removes the material rather than asking the model to ignore it.

    Metrics that are FINE are collapsed. Metrics that are NOT fine keep their own line, with
    value and unit, because that is the boundary the completeness rules protect.
    """
    lines: List[str] = []
    fine: List[str] = []
    for a in assessments:
        value = f"{_format_value(a.value)} {a.unit}".strip()
        citation = f" [{a.source_index}]" if a.source_index else ""
        if a.is_index:
            if a.status in (STATUS_GOOD,):
                fine.append(_plain_display(a))
            else:
                lines.append(
                    f"- {_plain_display(a)} = {value} — {a.band_label}. Status: {a.status}."
                )
            continue
        if a.status == STATUS_UNRATED:
            lines.append(
                f"- {_plain_display(a)} = {value} — no comparable limit was published for "
                f"this reading, so it could not be checked. Status: {a.status}."
            )
            continue
        if a.status == STATUS_WITHIN:
            fine.append(_plain_display(a))
            continue
        lines.append(
            f"- {_plain_display(a)} = {value} — {_PLAIN_VERBS[a.status]}{citation}. "
            f"Status: {a.status}."
        )
    if fine:
        # Stated as a fact the answer may repeat, and true of every metric it names.
        lines.append(
            f"- Everything else measured is within its recommended range: {', '.join(fine)}."
        )
    return lines


def render_assessment_block(
    assessments: List[MetricAssessment], compliance_detail: bool = True
) -> str:
    """Render the verdicts as the labelled context section the prompt refers to.

    ``compliance_detail=False`` drops the threshold figures, the standards bodies and the
    index acronyms, and collapses the metrics that are within range into one sentence. The
    verdicts themselves are unchanged — the same computation, described for a reader who
    does not want compliance language.
    """
    if not assessments:
        return ""
    lines = [
        "These verdicts are COMPUTED from the measured values and the citation "
        "sources. They are authoritative: state them as given, do not recompute or "
        "second-guess them, and do not introduce a threshold number that does not "
        "appear here.",
        "",
    ]
    if not compliance_detail:
        lines += _render_plain(assessments)
    for a in assessments if compliance_detail else []:
        value = f"{_format_value(a.value)} {a.unit}".strip()
        if a.is_index:
            lines.append(
                f"- {a.display} = {value} — {a.band_label.upper()} band "
                f"({a.note}). Status: {a.status}."
            )
        elif a.status == STATUS_UNRATED:
            lines.append(f"- {a.display} = {value} — not rated: {a.note}.")
        else:
            threshold = f"{_format_value(a.threshold_value)} {a.threshold_unit}".strip()
            citation = f" [{a.source_index}]" if a.source_index else ""
            verb = {
                STATUS_EXCEEDS: "EXCEEDS the strictest applicable limit",
                STATUS_NEAR: "is approaching the strictest applicable limit",
                STATUS_WITHIN: "is within the strictest applicable limit",
                STATUS_OUTSIDE_BAND: "is outside the optimal range (not a hard limit)",
            }[a.status]
            # The basis travels with the figure. Without it the model supplied one from
            # memory and picked the wrong one — a "today" answer quoting WHO's ANNUAL mean.
            basis = f", {a.threshold_condition}" if a.threshold_condition else ""
            ambient = (
                " — NOTE: no indoor standard covers this metric, so this is an OUTDOOR "
                "ambient limit applied for reference"
                if a.ambient_basis
                else ""
            )
            lines.append(
                f"- {a.display} = {value} — {verb} of {threshold} "
                f"({a.source_label}{basis}){citation}{ambient}. Status: {a.status}."
            )

    worst = min(assessments, key=lambda a: _STATUS_RANK.get(a.status, 9))
    over = [a.display for a in assessments if a.status == STATUS_EXCEEDS]
    poor = [a.display for a in assessments if a.status == STATUS_POOR]
    if over or poor:
        parts = []
        if over:
            parts.append(f"{', '.join(over)} above the applicable threshold")
        if poor:
            parts.append(f"{', '.join(poor)} scoring in a poor band")
        lines += [
            "",
            f"OVERALL: {'; '.join(parts)} — the overall verdict must reflect this and "
            "cannot be 'good', 'healthy' or 'no concerns'.",
        ]
    elif worst.status in (STATUS_NEAR, STATUS_FAIR):
        lines += ["", f"OVERALL: nothing over its limit; {worst.display} is the weakest point."]
    else:
        lines += ["", "OVERALL: nothing measured exceeds its applicable threshold."]
    return "\n".join(lines)


# Row keys that are not metrics. A reading row carries these alongside the values.
_NON_METRIC_ROW_KEYS = frozenset({"lab_space", "bucket", "space", "timestamp"})

# Generic value columns produced when ONE metric was queried and named on the payload
# instead of in the row. Priority order: the average is the headline figure of an
# aggregation, and the extremes only stand in when there is no average.
_GENERIC_VALUE_KEYS = ("value", "avg_value", "mean_value", "max_value", "min_value")


def readings_from_rows(
    rows: Optional[Iterable[Dict[str, Any]]],
    fallback_metric: Optional[str] = None,
) -> Dict[str, Any]:
    """Metric -> latest value, from either row shape the query layer produces.

    A metric *pack* produces one column per metric (``{"co2": 452, "voc": 0.06, ...}``),
    while a single named metric produces a generic ``value`` column and names the metric
    elsewhere on the payload. Both shapes reach the assessment, and only the first used to
    work: a point lookup like "what is the VOC?" arrived as ``{"value": 0.06}``, matched no
    metric, and produced ZERO verdict lines — silently handing the model a number with no
    computed comparison, which is the exact situation this module exists to prevent.

    ``fallback_metric`` is the metric name to attach when the row is ``value``-shaped.

    An AGGREGATE row is the same shape under a different name — ``avg_value`` rather than
    ``value`` — and had the same defect for the same reason: "what was the average CO2 last
    week?" normalised to ``{"avg_value": 600}``, matched no metric, and produced no verdict
    at all. Every generic value column is handled here so the next one added does not
    silently reopen the hole.
    """
    latest: Dict[str, Any] = {}
    for row in reversed(list(rows or [])):
        if isinstance(row, dict):
            latest = row
            break
    if not latest:
        return {}
    readings = {k: v for k, v in latest.items() if k not in _NON_METRIC_ROW_KEYS}
    if readings and set(readings).issubset(_GENERIC_VALUE_KEYS):
        metric = str(fallback_metric or "").strip()
        if not metric:
            return {}
        for key in _GENERIC_VALUE_KEYS:
            if readings.get(key) is not None:
                return {metric: readings[key]}
        return {}
    return readings


def governing_records(
    readings: Dict[str, Any],
    records: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """The guideline records the computed assessment actually graded against.

    ``get_thresholds_for_metrics`` returns EVERY active record for the metrics on screen,
    and that whole list was offered to the model as numbered citation sources — seventeen
    of them for a six-metric air answer, of which the answer used five. Worse than noisy:
    the same standard appears once per metric AND once per unit, so "RESET Air Standard
    v2.1 — Commercial Interiors, Section 4: Performance Thresholds (2021)" was listed four
    times with four different thresholds, and the reader's Sources panel showed eleven
    entries for an answer that cited a handful.

    The set that can honestly be cited is smaller than the set that was fetched, because
    the assessment has already chosen: one governing threshold per metric. The permissive
    twin, the wrong-unit twin and the standards that lost the strictest-wins comparison are
    not things the answer may quote — the directives forbid it — so offering them as
    citation options only invites the model to reach for one.

    The full list still reaches :func:`assess_readings`, which needs every candidate to
    pick the strictest; only the *citable* list is narrowed. With no readings (a standards
    question with nothing measured) there is nothing to govern and the list passes through.
    """
    all_records = list(records or [])
    if not readings or not all_records:
        return all_records
    # Local 1-based numbering purely to bind an assessment back to its record; the caller
    # renumbers when it builds the citation block.
    indexed = [{**record, "index": i} for i, record in enumerate(all_records, start=1)]
    used = {
        a.source_index for a in assess_readings(readings, indexed) if a.source_index
    }
    kept = [record for i, record in enumerate(all_records, start=1) if i in used]
    # A metric whose only records are unrated (ASHRAE 62.1 on CO2, ASHRAE 55 on
    # temperature) governs nothing, so it drops out here — which is correct: an answer may
    # not cite it as a threshold, and the assessment already reports the metric as unrated.
    return kept


def build_assessment_section(
    readings: Dict[str, Any],
    indexed_sources: Iterable[Dict[str, Any]],
    compliance_detail: bool = True,
) -> str:
    return render_assessment_block(
        assess_readings(readings, indexed_sources), compliance_detail=compliance_detail
    )
