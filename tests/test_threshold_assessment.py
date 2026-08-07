"""Threshold comparison is arithmetic, so it is done in code and pinned here.

Three real answers motivated moving it out of the prompt:

  * PM2.5 17.2 ug/m3 "above the EPA daily standard" -- EPA's is 35; WHO's 15 is the
    one exceeded. Right direction, wrong source.
  * VOC 1.25 ppm reported as "0.64 ppm, below RESET Air Grade A threshold of 0.8 ppm".
    RESET Grade A is 0.102 ppm here, so the reading was over twelve times the limit and
    neither number in that sentence existed.
  * IAQ 0.0 -- the worst value on a 0-100 higher-is-better scale -- described as
    "consistent with low pollutant levels".

The fixtures below are the libra_lab readings from that last report.
"""

import os
import sys
import unittest

TEST_DIR = os.path.dirname(__file__)
SERVER_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

from executors.db_support import threshold_assessment as ta


def _source(index, metric, value, unit, ttype="max", label=None, key=None, condition=None):
    return {
        "index": index,
        "metric": metric,
        "threshold_value": value,
        "threshold_type": ttype,
        "threshold_unit": unit,
        "source_label": label or f"{metric} source {index}",
        "source_key": key,
        "threshold_condition": condition,
    }


# Mirrors what the guideline table actually holds for these metrics.
SOURCES = [
    _source(1, "co2", 1000, "ppm", "max", "RESET Air Standard v2.1", "RESET_AIR_V2",
            "Grade A, occupied hours"),
    _source(2, "pm25", 35, "µg/m³", "max", "EPA NAAQS 2024", "EPA_PM25_NAAQS_2024",
            "24-hour average"),
    _source(3, "pm25", 15, "µg/m³", "max", "WHO Global Air Quality Guidelines 2021",
            "WHO_AQG_2021", "24-hour mean guideline"),
    _source(4, "voc", 500, "µg/m³", "max", "RESET Air Standard v2.1", "RESET_AIR_V2_VOC"),
    _source(5, "voc", 0.102, "ppm", "max", "RESET Air Standard v2.1 (ppm equivalent)",
            "RESET_AIR_V2_VOC_PPM", "Grade A, occupied hours (derived from 500 µg/m³)"),
    _source(6, "voc", 0.061, "ppm", "max", "WHO IAQ Guidelines 2010 (ppm equivalent)",
            "WHO_IAQ_VOC_2010_PPM", "comfort range upper boundary"),
    _source(7, "humidity", 65, "percent RH", "max", "ANSI/ASHRAE 62.1-2022",
            "ASHRAE_62_1_2022_HUM", "occupied spaces, IAQ requirement"),
    _source(8, "humidity", 50, "percent RH", "range_max", "EPA: The Inside Story",
            "EPA_INDOOR_HUMIDITY", "optimal comfort and health range"),
]

# The same set with the two indoor PM2.5 standards present, which is what the seeded table
# now holds. Kept separate so the ambient-fallback path above stays exercised.
SOURCES_WITH_INDOOR_PM25 = SOURCES + [
    _source(9, "pm25", 12, "µg/m³", "max", "RESET Air Standard v2.1 — Commercial Interiors",
            "RESET_AIR_V2_PM25", "Grade A, occupied hours, indoor"),
    _source(10, "pm25", 15, "µg/m³", "max", "WELL Building Standard v2, Feature A01",
            "WELL_V2_A01", "regularly occupied spaces, indoor"),
]

# The reported libra_lab turn.
LIBRA_LAB = {
    "co2": 480.0,
    "pm25": 17.1,
    "voc": 1.25,
    "humidity": 54.97,
    "ieq": 42.58,
    "iaq": 0.0,
    "itc": 82.52,
    "iac": 63.35,
    "iil": 39.47,
}


def _by_metric(assessments):
    return {a.metric: a for a in assessments}


class StrictestThresholdTests(unittest.TestCase):
    def test_strictest_hard_limit_wins(self):
        a = ta.assess_metric("pm25", 17.1, SOURCES)
        self.assertEqual(a.status, ta.STATUS_EXCEEDS)
        # WHO 15, not EPA 35 — the failure that started this.
        self.assertEqual(a.threshold_value, 15)
        self.assertEqual(a.source_index, 3)
        self.assertIn("WHO", a.source_label)


class IndoorApplicabilityTests(unittest.TestCase):
    """An indoor reading is graded against an indoor standard when one exists.

    WHO AQG and EPA NAAQS are ambient (outdoor) standards — both records say so in their own
    caveat text. Grading a lab against them reported the room as breaching an obligation
    nobody had placed on it, and strictest-wins meant WHO won every PM2.5 verdict.
    """

    def test_indoor_standard_governs_over_a_stricter_ambient_one(self):
        a = ta.assess_metric("pm25", 16.3, SOURCES_WITH_INDOOR_PM25)
        self.assertEqual(a.status, ta.STATUS_EXCEEDS)
        self.assertEqual(a.threshold_value, 12)
        self.assertIn("RESET", a.source_label)
        self.assertFalse(a.ambient_basis)

    def test_strictest_still_wins_among_the_indoor_standards(self):
        # RESET 12 is stricter than WELL 15; a clean verdict cannot be bought with WELL.
        a = ta.assess_metric("pm25", 13.0, SOURCES_WITH_INDOOR_PM25)
        self.assertEqual(a.status, ta.STATUS_EXCEEDS)
        self.assertEqual(a.threshold_value, 12)

    def test_ambient_is_used_only_when_nothing_indoor_covers_the_metric(self):
        a = ta.assess_metric("pm25", 17.1, SOURCES)
        self.assertTrue(a.ambient_basis)
        self.assertEqual(a.threshold_value, 15)

    def test_the_ambient_substitution_is_stated_on_the_line(self):
        block = ta.render_assessment_block([ta.assess_metric("pm25", 17.1, SOURCES)])
        self.assertIn("OUTDOOR", block)
        indoor = ta.render_assessment_block(
            [ta.assess_metric("pm25", 16.3, SOURCES_WITH_INDOOR_PM25)]
        )
        self.assertNotIn("OUTDOOR", indoor)

    def test_an_indoor_metric_is_never_marked_ambient(self):
        for metric, value in (("co2", 443.0), ("voc", 0.06), ("humidity", 55.0)):
            self.assertFalse(
                ta.assess_metric(metric, value, SOURCES_WITH_INDOOR_PM25).ambient_basis,
                metric,
            )


class AveragingBasisTests(unittest.TestCase):
    """A limit quoted under the wrong averaging period is a wrong limit.

    Asked "how is the air quality today?", the answer cited "the WHO annual mean limit of
    5 µg/m³" — a real number from a real standard, compared against one day's readings, and
    not the figure the assessment had computed against (the 24-hour guideline). The basis
    now travels with the figure instead of being supplied from the model's memory.
    """

    def test_condition_is_carried_onto_the_assessment(self):
        a = ta.assess_metric("pm25", 16.3, SOURCES_WITH_INDOOR_PM25)
        self.assertEqual(a.threshold_condition, "Grade A, occupied hours, indoor")

    def test_condition_is_rendered_beside_the_threshold(self):
        block = ta.render_assessment_block(
            [ta.assess_metric("pm25", 16.3, SOURCES_WITH_INDOOR_PM25)]
        )
        self.assertIn("12.0 µg/m³", block)
        self.assertIn("Grade A, occupied hours, indoor", block)

    def test_a_source_without_a_condition_renders_cleanly(self):
        bare = [_source(20, "co2", 800, "ppm", "max", "Test Source", "TEST_KEY", None)]
        block = ta.render_assessment_block([ta.assess_metric("co2", 900, bare)])
        self.assertIn("Test Source", block)
        self.assertNotIn("None", block)

    def test_the_numbered_sources_block_shows_the_published_figure(self):
        from evidence.citation_processor import build_numbered_sources_block

        block, _ = build_numbered_sources_block([
            {
                "source_key": "WHO_AQG_2021",
                "source_label": "WHO Global Air Quality Guidelines 2021",
                "metric": "pm25", "citation_tier": "regulatory",
                "threshold_value": 15, "threshold_type": "max",
                "threshold_unit": "µg/m³",
                "threshold_condition": "24-hour mean guideline",
            },
        ])
        # The number the source actually publishes, on the line that names it. Without this
        # the model was asked to cite a figure it was never shown, and filled the gap from
        # memory with WHO's annual 5 µg/m³.
        self.assertIn("maximum 15 µg/m³", block)
        self.assertIn("24-hour mean guideline", block)

    def test_a_source_with_no_threshold_says_so_rather_than_going_silent(self):
        from evidence.citation_processor import build_numbered_sources_block

        block, _ = build_numbered_sources_block([
            {
                "source_key": "ASHRAE_55_2023_COMFORT",
                "source_label": "ANSI/ASHRAE Standard 55-2023",
                "metric": "temperature", "citation_tier": "regulatory",
                "threshold_value": None, "threshold_type": None,
                "threshold_unit": "degC",
            },
        ])
        self.assertIn("no numeric threshold", block)

    def test_a_reading_under_the_strict_limit_is_within(self):
        a = ta.assess_metric("co2", 480.0, SOURCES)
        self.assertEqual(a.status, ta.STATUS_WITHIN)
        self.assertEqual(a.source_index, 1)

    def test_near_is_distinguished_from_within(self):
        a = ta.assess_metric("pm25", 14.0, SOURCES)   # 93% of 15
        self.assertEqual(a.status, ta.STATUS_NEAR)

    def test_hard_limit_outranks_a_comfort_band_edge(self):
        # 54.97 %RH is over EPA's 50 % optimal-range top but well under ASHRAE's 65 %
        # limit. Reporting that as an exceedance would flag an ordinary room.
        a = ta.assess_metric("humidity", 54.97, SOURCES)
        self.assertEqual(a.status, ta.STATUS_WITHIN)
        self.assertEqual(a.threshold_value, 65)

    def test_band_edge_is_used_only_when_no_hard_limit_exists(self):
        band_only = [_source(1, "humidity", 50, "percent RH", "range_max")]
        a = ta.assess_metric("humidity", 54.97, band_only)
        self.assertEqual(a.status, ta.STATUS_OUTSIDE_BAND)
        self.assertNotEqual(a.status, ta.STATUS_EXCEEDS)


class UnitMatchingTests(unittest.TestCase):
    def test_threshold_must_be_in_the_readings_unit(self):
        # VOC reads in ppm; a µg/m³-only source cannot rate it.
        ugm3_only = [s for s in SOURCES if s["metric"] == "voc" and s["threshold_unit"] == "µg/m³"]
        a = ta.assess_metric("voc", 1.25, ugm3_only)
        self.assertEqual(a.status, ta.STATUS_UNRATED)
        self.assertIn("not expressed in ppm", a.note)

    def test_no_source_at_all_is_reported_as_such(self):
        a = ta.assess_metric("voc", 1.25, [])
        self.assertEqual(a.status, ta.STATUS_UNRATED)
        self.assertIn("no published threshold", a.note)

    def test_micro_sign_and_greek_mu_are_the_same_unit(self):
        # The registry writes μg/m³ (U+03BC), the seed writes µg/m³ (U+00B5). They render
        # identically; a naive comparison silently decides there is no threshold.
        self.assertEqual(ta._normalize_unit("μg/m³"), ta._normalize_unit("µg/m³"))
        self.assertEqual(ta._normalize_unit("ug/m3"), ta._normalize_unit("µg/m³"))
        self.assertEqual(ta._normalize_unit("percent RH"), ta._normalize_unit("%"))

    def test_ppm_source_rates_a_ppm_reading(self):
        a = ta.assess_metric("voc", 1.25, SOURCES)
        self.assertEqual(a.status, ta.STATUS_EXCEEDS)
        self.assertEqual(a.threshold_value, 0.061)
        self.assertEqual(a.threshold_unit, "ppm")


class IndexMetricTests(unittest.TestCase):
    def test_zero_is_the_worst_band_not_a_good_sign(self):
        a = ta.assess_metric("iaq", 0.0, SOURCES)
        self.assertEqual(a.band, "low")
        self.assertEqual(a.status, ta.STATUS_POOR)
        self.assertIn("higher is better", a.note)

    def test_index_bands(self):
        for value, band in ((90.0, "high"), (60.0, "medium"), (40.0, "moderate"), (10.0, "low")):
            with self.subTest(value=value):
                self.assertEqual(ta.assess_metric("iaq", value, SOURCES).band, band)

    def test_index_metrics_do_not_borrow_a_concentration_threshold(self):
        a = ta.assess_metric("ieq", 42.58, SOURCES)
        self.assertIsNone(a.threshold_value)
        self.assertEqual(a.unit, "/100")


class LibraLabRegressionTests(unittest.TestCase):
    """The exact readings behind 'good, all pollutants well within healthy ranges'."""

    def setUp(self):
        self.assessments = ta.assess_readings(LIBRA_LAB, SOURCES)
        self.by_metric = _by_metric(self.assessments)
        self.block = ta.render_assessment_block(self.assessments)

    def test_the_three_misreported_metrics_are_flagged(self):
        self.assertEqual(self.by_metric["voc"].status, ta.STATUS_EXCEEDS)
        self.assertEqual(self.by_metric["pm25"].status, ta.STATUS_EXCEEDS)
        self.assertEqual(self.by_metric["iaq"].status, ta.STATUS_POOR)

    def test_worst_first_ordering(self):
        self.assertIn(self.assessments[0].status, (ta.STATUS_EXCEEDS, ta.STATUS_POOR))

    def test_overall_line_forbids_a_clean_verdict(self):
        self.assertIn("cannot be 'good'", self.block)
        for metric in ("PM2.5", "VOC", "IAQ"):
            self.assertIn(metric, self.block)

    def test_block_never_offers_a_threshold_it_did_not_compute(self):
        # The fabricated "0.8 ppm" must not be constructible from this block: every
        # number in it comes from a source record.
        self.assertNotIn("0.8 ppm", self.block)
        self.assertIn("0.061 ppm", self.block)

    def test_block_is_empty_without_readings(self):
        self.assertEqual(ta.build_assessment_section({}, SOURCES), "")


class DirectiveWiringTests(unittest.TestCase):
    def test_directives_tell_the_model_the_verdicts_are_authoritative(self):
        from prompting.db_prompts import (
            DB_TOOL_RESPONSE_DIRECTIVE,
            DB_TOOL_RESPONSE_DIRECTIVE_AIR_QUALITY_POINT_LOOKUP,
        )

        for directive in (
            DB_TOOL_RESPONSE_DIRECTIVE,
            DB_TOOL_RESPONSE_DIRECTIVE_AIR_QUALITY_POINT_LOOKUP,
        ):
            text = directive.lower()
            self.assertIn("threshold assessment", text)
            self.assertIn("never recompute a comparison", text)
            self.assertIn("higher is better", text)
            # The style call: numbers only where magnitude matters, citations carry
            # the rest.
            self.assertIn("only for metrics flagged exceeds or near", text)

    def test_context_section_is_rendered_when_present(self):
        from prompting.shared_prompts import build_grounded_context_sections

        context = build_grounded_context_sections(
            measured_room_facts={"co2": 480},
            threshold_assessment="- CO2 = 480 ppm — within.",
        )
        self.assertIn("## Threshold Assessment (computed — authoritative)", context)
        self.assertIn("- CO2 = 480 ppm — within.", context)

    def test_context_section_is_omitted_when_absent(self):
        from prompting.shared_prompts import build_grounded_context_sections

        context = build_grounded_context_sections(measured_room_facts={"co2": 480})
        self.assertNotIn("Threshold Assessment", context)

    def test_indexed_sources_carry_what_the_assessment_needs(self):
        from evidence.citation_processor import build_numbered_sources_block

        _, indexed = build_numbered_sources_block([
            {
                "source_key": "WHO_AQG_2021", "source_label": "WHO 2021", "metric": "pm25",
                "citation_tier": "regulatory", "threshold_value": 15,
                "threshold_type": "max", "threshold_unit": "µg/m³",
            }
        ])
        self.assertEqual(indexed[0]["metric"], "pm25")
        self.assertEqual(indexed[0]["threshold_type"], "max")
        self.assertEqual(indexed[0]["index"], 1)


class CitableSourcePruningTests(unittest.TestCase):
    """The list a reader may cite is smaller than the list that was fetched.

    A six-metric air answer was offered seventeen numbered sources and used five. The same
    standard appeared once per metric AND once per unit, so "RESET Air Standard v2.1 —
    Commercial Interiors, Section 4: Performance Thresholds (2021)" was listed four times
    with four different thresholds, and the reader's Sources panel showed eleven entries.
    """

    def _records(self):
        # source_key/metric pairs as get_thresholds_for_metrics returns them.
        return [
            {"source_key": "ASHRAE_62_1_2022", "metric": "co2", "threshold_value": None,
             "threshold_type": None, "threshold_unit": "ppm", "source_label": "ASHRAE 62.1"},
            {"source_key": "RESET_AIR_V2", "metric": "co2", "threshold_value": 1000,
             "threshold_type": "max", "threshold_unit": "ppm", "source_label": "RESET Air"},
            {"source_key": "ALLEN_ET_AL_2016", "metric": "co2", "threshold_value": 1000,
             "threshold_type": "max", "threshold_unit": "ppm", "source_label": "Allen 2016"},
            {"source_key": "RESET_AIR_V2_VOC", "metric": "voc", "threshold_value": 500,
             "threshold_type": "max", "threshold_unit": "µg/m³", "source_label": "RESET Air"},
            {"source_key": "RESET_AIR_V2_VOC_PPM", "metric": "voc", "threshold_value": 0.102,
             "threshold_type": "max", "threshold_unit": "ppm", "source_label": "RESET Air ppm"},
            {"source_key": "WHO_IAQ_VOC_2010_PPM", "metric": "voc", "threshold_value": 0.061,
             "threshold_type": "max", "threshold_unit": "ppm", "source_label": "WHO ppm"},
        ]

    def test_only_the_governing_record_per_metric_survives(self):
        kept = ta.governing_records({"co2": 443.0, "voc": 0.06}, self._records())
        self.assertEqual(
            {r["source_key"] for r in kept}, {"RESET_AIR_V2", "WHO_IAQ_VOC_2010_PPM"}
        )

    def test_the_wrong_unit_twin_of_the_same_standard_is_dropped(self):
        kept = ta.governing_records({"voc": 0.06}, self._records())
        keys = {r["source_key"] for r in kept}
        self.assertNotIn("RESET_AIR_V2_VOC", keys)   # µg/m³ against a ppm reading

    def test_a_record_with_no_threshold_is_never_citable(self):
        kept = ta.governing_records({"co2": 443.0}, self._records())
        self.assertNotIn("ASHRAE_62_1_2022", {r["source_key"] for r in kept})

    def test_no_readings_leaves_the_list_untouched(self):
        # A standards question with nothing measured: there is nothing to govern, and the
        # records ARE the answer material.
        records = self._records()
        self.assertEqual(ta.governing_records({}, records), records)

    def test_original_order_is_preserved(self):
        kept = ta.governing_records({"co2": 443.0, "voc": 0.06}, self._records())
        self.assertEqual([r["source_key"] for r in kept],
                         ["RESET_AIR_V2", "WHO_IAQ_VOC_2010_PPM"])


class AggregateRowShapeTests(unittest.TestCase):
    """An aggregate row is the point-lookup bug under a different column name.

    ``{"value": 453}`` was fixed; ``{"avg_value": 600}`` was not, so "what was the average
    CO2 last week?" produced zero verdict lines — the computed machinery bypassed for a
    whole intent.
    """

    def test_avg_value_normalises_to_the_named_metric(self):
        self.assertEqual(
            ta.readings_from_rows([{"avg_value": 600}], "co2"), {"co2": 600}
        )

    def test_value_still_wins_when_both_are_present(self):
        self.assertEqual(
            ta.readings_from_rows([{"value": 450, "avg_value": 600}], "co2"), {"co2": 450}
        )

    def test_an_extreme_stands_in_when_there_is_no_average(self):
        self.assertEqual(
            ta.readings_from_rows([{"max_value": 900}], "co2"), {"co2": 900}
        )

    def test_a_metric_pack_row_is_untouched(self):
        rows = [{"co2": 450, "pm25": 16.3, "bucket": "2026-08-07T10:00"}]
        self.assertEqual(ta.readings_from_rows(rows), {"co2": 450, "pm25": 16.3})

    def test_a_generic_row_with_no_metric_name_yields_nothing(self):
        self.assertEqual(ta.readings_from_rows([{"avg_value": 600}], None), {})

    def test_an_aggregate_reading_gets_a_verdict(self):
        block = ta.build_assessment_section(
            ta.readings_from_rows([{"avg_value": 1400}], "co2"), SOURCES
        )
        self.assertIn("EXCEEDS", block)



if __name__ == "__main__":
    unittest.main()
