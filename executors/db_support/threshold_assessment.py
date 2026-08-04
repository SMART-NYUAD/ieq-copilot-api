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

# Statuses that must force the overall verdict away from "good".
_FLAGGED_STATUSES = {STATUS_EXCEEDS, STATUS_POOR}


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
            lines.append(
                f"- {a.display} = {value} — {verb} of {threshold} "
                f"({a.source_label}){citation}. Status: {a.status}."
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
    """
    latest: Dict[str, Any] = {}
    for row in reversed(list(rows or [])):
        if isinstance(row, dict):
            latest = row
            break
    if not latest:
        return {}
    readings = {k: v for k, v in latest.items() if k not in _NON_METRIC_ROW_KEYS}
    if set(readings) == {"value"}:
        metric = str(fallback_metric or "").strip()
        return {metric: readings["value"]} if metric else {}
    return readings


def build_assessment_section(
    readings: Dict[str, Any],
    indexed_sources: Iterable[Dict[str, Any]],
    compliance_detail: bool = True,
) -> str:
    return render_assessment_block(
        assess_readings(readings, indexed_sources), compliance_detail=compliance_detail
    )
