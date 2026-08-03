"""An air-quality assessment may not drop the metrics that spoil its verdict.

Reported symptom: "How is the air quality today?" answered with CO2, IAQ, humidity
and the comfort sub-indices, and called the air "good", "no signs of pollutant
buildup", "excellent". PM2.5 was 17.86 ug/m3 at the time -- above the WHO 15 ug/m3
24-hour guideline that was in the model's own Citation Sources -- and VOC was
0.222 ppm. Both were fetched, both were in the payload (`has_full_coverage: true`),
and both were left out of the answer.

Tracing it: the router, the metric pack and the executor were all correct. The
omission happened in the answer prompt, which gave the model discretion ("only the
most important metric-by-metric interpretation", "only when needed") under a hard
brevity cap ("at most 2 short bullets", "under 90 words") and never said a metric
may not be dropped or that the verdict follows the worst metric. Given that, it kept
the flattering metrics.

A second, independent defect made VOC the easiest one to drop: its guideline records
were stored under the legacy metric key `tvoc` while `get_thresholds_for_metrics`
looks up `voc`, so VOC genuinely had no retrievable threshold and an uncited numeric
claim is discouraged elsewhere in the prompt. Migration 003 normalises the key; the
test below stops the seed file drifting from the schema that way again.
"""

import os
import re
import sys
import unittest

TEST_DIR = os.path.dirname(__file__)
SERVER_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

from prompting.db_prompts import (
    DB_TOOL_RESPONSE_DIRECTIVE,
    DB_TOOL_RESPONSE_DIRECTIVE_ADVISORY,
    DB_TOOL_RESPONSE_DIRECTIVE_AIR_QUALITY_POINT_LOOKUP,
    DB_TOOL_RESPONSE_DIRECTIVE_IEQ,
    DB_TOOL_RESPONSE_DIRECTIVE_POINT_LOOKUP,
)
from prompting.shared_prompts import PRESENTATION_STYLE_PROMPT

ASSESSMENT_DIRECTIVES = {
    "base": DB_TOOL_RESPONSE_DIRECTIVE,
    "air_quality": DB_TOOL_RESPONSE_DIRECTIVE_AIR_QUALITY_POINT_LOOKUP,
}


class MetricCompletenessRuleTests(unittest.TestCase):
    def test_assessment_directives_forbid_dropping_a_pollutant(self):
        for name, directive in ASSESSMENT_DIRECTIVES.items():
            with self.subTest(directive=name):
                text = directive.lower()
                self.assertIn("metric completeness", text)
                self.assertIn("must appear in the answer with", text)
                self.assertIn("never omit one", text)

    def test_assessment_directives_set_the_verdict_by_the_worst_metric(self):
        for name, directive in ASSESSMENT_DIRECTIVES.items():
            with self.subTest(directive=name):
                text = directive.lower()
                self.assertIn("worst metric, not the best", text)
                # The exact reassurance words the reported answer used.
                for word in ("good", "excellent", "no concerns"):
                    self.assertIn(word, text)

    def test_assessment_directives_report_metrics_that_have_no_threshold(self):
        for name, directive in ASSESSMENT_DIRECTIVES.items():
            with self.subTest(directive=name):
                text = directive.lower()
                self.assertIn("no usable threshold is still reported", text)
                # VOC reads in ppm while its only thresholds are in ug/m3. Converting
                # between them needs a molar-mass assumption the model must not invent.
                self.assertIn("cannot be compared directly", text)
                self.assertIn("never convert between units yourself", text)

    def test_completeness_is_not_applied_where_it_would_be_wrong(self):
        # An IEQ ask wants the score family; a point lookup asked for one named
        # metric. Forcing the pollutant pack into either is the same failure inverted.
        for name, directive in (
            ("ieq", DB_TOOL_RESPONSE_DIRECTIVE_IEQ),
            ("point_lookup", DB_TOOL_RESPONSE_DIRECTIVE_POINT_LOOKUP),
        ):
            with self.subTest(directive=name):
                self.assertNotIn("METRIC COMPLETENESS", directive)

    def test_advisory_answer_may_not_reassure_about_an_out_of_range_metric(self):
        text = DB_TOOL_RESPONSE_DIRECTIVE_ADVISORY.lower()
        self.assertIn("may not be reassuring about a metric that is out", text)

    def test_old_omission_licences_are_gone(self):
        for name, directive in ASSESSMENT_DIRECTIVES.items():
            with self.subTest(directive=name):
                text = directive.lower()
                self.assertNotIn("only the most important metric-by-metric", text)
                self.assertNotIn("core metrics only when needed", text)

    def test_brevity_cap_yields_to_completeness(self):
        text = PRESENTATION_STYLE_PROMPT.lower()
        self.assertIn("completeness wins over the word cap", text)
        self.assertIn("multi-metric air-quality assessment is not a simple answer", text)


class GuidelineMetricKeyTests(unittest.TestCase):
    """The seed file's metric keys must satisfy the schema's CHECK constraint.

    VOC records were stored as `tvoc` while the lookup asks for `voc`, so VOC had no
    retrievable threshold. The deployed CHECK had drifted too -- it permitted `tvoc`
    and forbade `voc`, the reverse of the checked-in migration. Comparing the seed
    against the migration catches that class of drift without touching a database.
    """

    def _allowed_metrics(self):
        path = os.path.join(SERVER_DIR, "storage/migrations/002_guideline_records.sql")
        with open(path, encoding="utf-8") as fh:
            sql = fh.read()
        block = re.search(r"metric\s+TEXT NOT NULL\s*CHECK \(metric IN \((.*?)\)\)", sql, re.S)
        self.assertIsNotNone(block, "could not locate the metric CHECK in migration 002")
        return set(re.findall(r"'([a-z0-9_]+)'", block.group(1)))

    def test_seed_metrics_are_all_permitted_by_the_schema(self):
        from storage.seed_guidelines import GUIDELINE_RECORDS

        allowed = self._allowed_metrics()
        self.assertIn("voc", allowed)
        offenders = sorted({
            r["metric"] for r in GUIDELINE_RECORDS if r["metric"] not in allowed
        })
        self.assertEqual(offenders, [], f"seed metrics rejected by the schema: {offenders}")

    def test_seed_uses_the_canonical_voc_key_not_the_legacy_alias(self):
        from storage.seed_guidelines import GUIDELINE_RECORDS
        from executors.metric_registry import resolve_metric

        metrics = {r["metric"] for r in GUIDELINE_RECORDS}
        self.assertIn("voc", metrics)
        self.assertNotIn("tvoc", metrics)
        # `tvoc` stays a recognised alias for reading data; it is just never a stored key.
        self.assertEqual(resolve_metric("tvoc"), "voc")

    def test_voc_has_at_least_one_thresholded_source(self):
        from storage.seed_guidelines import GUIDELINE_RECORDS

        voc = [r for r in GUIDELINE_RECORDS if r["metric"] == "voc"]
        self.assertTrue(voc, "no VOC guideline records in the seed")
        thresholded = [r for r in voc if r.get("threshold_value") is not None]
        self.assertTrue(thresholded, "VOC has no record carrying a threshold_value")


class VocThresholdUnitTests(unittest.TestCase):
    """A threshold is only usable if it is in the unit the sensor reports.

    The Atmocube (Sensirion SGP41) reports TVOC in ppm, range 0-3, while every
    published TVOC threshold is in ug/m3. With nothing in ppm to compare against,
    a VOC reading could not be classified at all -- so the answer either went silent
    on VOC or leaned on another metric's threshold. The ppm records restate the same
    standards using the published TVOC conversion of 4.9 ug/m3 per ppb.
    """

    # 4.9 ug/m3 per ppb, i.e. ppm = (ug/m3 / 4.9) / 1000.
    UGM3_PER_PPB = 4.9
    EXPECTED = {
        "RESET_AIR_V2_VOC_PPM": (500, 0.102),
        "WELL_V2_A04_PPM": (500, 0.102),
        "WHO_IAQ_VOC_2010_PPM": (300, 0.061),
        "UBA_TVOC_PRECAUTIONARY_PPM": (950, 0.194),
    }

    def _voc_records(self):
        from storage.seed_guidelines import GUIDELINE_RECORDS

        return {
            r["source_key"]: r for r in GUIDELINE_RECORDS if r["metric"] == "voc"
        }

    def test_voc_has_a_threshold_in_the_unit_the_sensor_reports(self):
        from executors.metric_registry import METRICS

        sensor_unit = METRICS["voc"]["unit"]
        self.assertEqual(sensor_unit, "ppm")
        units = {r.get("threshold_unit") for r in self._voc_records().values()}
        self.assertIn(sensor_unit, units, f"no VOC threshold in {sensor_unit}: {units}")

    def test_ppm_values_match_the_published_conversion(self):
        records = self._voc_records()
        for key, (ugm3, expected_ppm) in self.EXPECTED.items():
            with self.subTest(source=key):
                self.assertIn(key, records, f"{key} missing from the seed")
                derived = (ugm3 / self.UGM3_PER_PPB) / 1000.0
                self.assertAlmostEqual(derived, expected_ppm, places=3)
                self.assertAlmostEqual(
                    float(records[key]["threshold_value"]), expected_ppm, places=3
                )
                self.assertEqual(records[key]["threshold_unit"], "ppm")

    def test_derived_records_declare_that_they_are_derived(self):
        # These ppm figures are not printed in the standards. Every one of them must
        # say so, or the answer model will attribute a number to a source that never
        # published it -- the failure mode GUIDELINE_CITATIONS exists to prevent.
        records = self._voc_records()
        for key in self.EXPECTED:
            with self.subTest(source=key):
                record = records[key]
                blob = f"{record['claim_text']} {record.get('caveat_text') or ''}".lower()
                self.assertIn("derived", blob)
                self.assertIn("4.9", record["claim_text"])
                self.assertIn("µg/m³", record["claim_text"])

    def test_original_mass_based_records_are_kept(self):
        # The ppm records are companions, not replacements: the µg/m³ figures are what
        # the standards actually publish and remain the citable ones.
        units = [r.get("threshold_unit") for r in self._voc_records().values()]
        self.assertIn("µg/m³", units)
        self.assertGreaterEqual(units.count("µg/m³"), 3)


if __name__ == "__main__":
    unittest.main()
