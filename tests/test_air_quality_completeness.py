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


if __name__ == "__main__":
    unittest.main()
