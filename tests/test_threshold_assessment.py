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


def _source(index, metric, value, unit, ttype="max", label=None):
    return {
        "index": index,
        "metric": metric,
        "threshold_value": value,
        "threshold_type": ttype,
        "threshold_unit": unit,
        "source_label": label or f"{metric} source {index}",
    }


# Mirrors what the guideline table actually holds for these metrics.
SOURCES = [
    _source(1, "co2", 1000, "ppm", "max", "RESET Air Standard v2.1"),
    _source(2, "pm25", 35, "µg/m³", "max", "EPA NAAQS 2024"),
    _source(3, "pm25", 15, "µg/m³", "max", "WHO Global Air Quality Guidelines 2021"),
    _source(4, "voc", 500, "µg/m³", "max", "RESET Air Standard v2.1"),
    _source(5, "voc", 0.102, "ppm", "max", "RESET Air Standard v2.1 (ppm equivalent)"),
    _source(6, "voc", 0.061, "ppm", "max", "WHO IAQ Guidelines 2010 (ppm equivalent)"),
    _source(7, "humidity", 65, "percent RH", "max", "ANSI/ASHRAE 62.1-2022"),
    _source(8, "humidity", 50, "percent RH", "range_max", "EPA: The Inside Story"),
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


if __name__ == "__main__":
    unittest.main()
