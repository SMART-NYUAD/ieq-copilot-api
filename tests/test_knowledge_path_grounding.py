"""The knowledge path must not judge a reading it was handed.

Reported from a live session — one conversation, two consecutive turns, the same reading:

    Turn 12 (DB path)        "VOC at 0.08 ppm exceeds the WHO Indoor Air Quality
                              Guideline (0.061 ppm) [14]"
    Turn 13 (knowledge path) "TVOC level is 0.08 ppm, which is within typical indoor
                              ranges and considered acceptable [3]"

The DB path was right. The knowledge path grounds definition answers in a live reading but
built no ``## Threshold Assessment`` section and, because semantic guideline search only
fires when the question is *about* standards, usually had no citation sources either. So it
received a number, no limit, and no computed verdict — and did the comparison itself. That is
precisely what ``threshold_assessment`` exists to take away from the model: measured on real
answers it got direction roughly right and attribution wrong, and under a stronger prompt it
invented numbers.

A second defect surfaced while fixing it, on the DB side. Reading rows come in two shapes: a
metric pack produces one column per metric, a single named metric produces a generic
``value`` column. Only the first was handled, so every POINT LOOKUP — "what is the CO2?" —
normalised to ``{"value": 453}``, matched no metric, and produced zero verdict lines. The
whole computed-verdict machinery was bypassed for the narrowest and most common question.
"""

import os
import sys
import unittest
from unittest.mock import patch

TEST_DIR = os.path.dirname(__file__)
SERVER_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

from executors.db_support import threshold_assessment  # noqa: E402
from executors import knowledge_executor  # noqa: E402
from executors.knowledge_executor import (  # noqa: E402
    CARD_TOOL_RESPONSE_DIRECTIVE,
    _readings_from_live_data,
    build_knowledge_grounding,
)

_PACK_PAYLOAD = {
    "metric": "co2",
    "rows": [
        {"lab_space": "smart_lab", "bucket": "2026-08-04T13:00:00+04:00",
         "co2": 452.0, "pm25": 8.2, "voc": 0.06, "humidity": 52.0},
    ],
}

_SINGLE_METRIC_PAYLOAD = {
    "metric": "voc",
    "rows": [{"lab_space": "smart_lab", "bucket": "2026-08-04T13:00:00+04:00", "value": 0.08}],
}

_VOC_SOURCE = [
    {
        "index": 1,
        "metric": "voc",
        "source_key": "WHO_IAQ_VOC_PPM",
        "source_label": "WHO Indoor Air Quality Guideline",
        "citation_tier": "regulatory",
        "threshold_value": 0.061,
        "threshold_unit": "ppm",
        "threshold_type": "max",
    }
]


class RowShapeNormalizationTests(unittest.TestCase):
    """Both reading-row shapes must resolve to metric -> value."""

    def test_metric_pack_row_is_used_directly(self):
        readings = threshold_assessment.readings_from_rows(_PACK_PAYLOAD["rows"], "co2")
        self.assertEqual(readings["voc"], 0.06)
        self.assertEqual(readings["co2"], 452.0)
        # Row bookkeeping columns are not metrics.
        self.assertNotIn("lab_space", readings)
        self.assertNotIn("bucket", readings)

    def test_value_shaped_row_takes_the_metric_name_from_the_payload(self):
        # The point-lookup bug: without the fallback this is {"value": 0.08}, which matches
        # no metric and yields no verdict at all.
        readings = threshold_assessment.readings_from_rows(
            _SINGLE_METRIC_PAYLOAD["rows"], _SINGLE_METRIC_PAYLOAD["metric"]
        )
        self.assertEqual(readings, {"voc": 0.08})

    def test_value_shaped_row_without_a_metric_name_yields_nothing(self):
        # Better an empty assessment than a reading filed under the literal key "value".
        self.assertEqual(threshold_assessment.readings_from_rows(_SINGLE_METRIC_PAYLOAD["rows"]), {})

    def test_empty_and_malformed_rows_are_safe(self):
        self.assertEqual(threshold_assessment.readings_from_rows(None), {})
        self.assertEqual(threshold_assessment.readings_from_rows([]), {})
        self.assertEqual(threshold_assessment.readings_from_rows(["not a dict"]), {})

    def test_a_point_lookup_produces_a_verdict(self):
        readings = threshold_assessment.readings_from_rows(_SINGLE_METRIC_PAYLOAD["rows"], "voc")
        section = threshold_assessment.build_assessment_section(readings, _VOC_SOURCE)
        self.assertIn("VOC", section.upper())
        # 0.08 ppm against a 0.061 ppm max is over the limit; the section must say so.
        self.assertIn("EXCEEDS", section.upper())


class KnowledgeGroundingTests(unittest.TestCase):
    def _grounding(self, payload, records=None):
        with patch.object(
            knowledge_executor, "get_thresholds_for_metrics", return_value=list(_VOC_SOURCE)
        ):
            return build_knowledge_grounding(
                user_question="what is VOC?",
                knowledge_cards=[],
                live_sensor_data=payload,
                guideline_records=list(records or []),
            )

    def test_live_reading_gets_a_computed_assessment_section(self):
        grounded, _sources, _records = self._grounding(_SINGLE_METRIC_PAYLOAD)
        self.assertIn("Threshold Assessment", grounded)
        self.assertIn("EXCEEDS", grounded.upper())

    def test_thresholds_are_fetched_for_the_metrics_on_screen(self):
        # Semantic guideline search only fires for standards questions, so without this a
        # definition question had no citation sources and nothing to compare against.
        _grounded, sources, records = self._grounding(_SINGLE_METRIC_PAYLOAD)
        self.assertTrue(sources, "expected citation sources for a live reading")
        self.assertTrue(any(r.get("metric") == "voc" for r in records))

    def test_no_live_reading_means_no_assessment_and_no_threshold_fetch(self):
        # A pure definition question with no readings must not grow a phantom assessment.
        with patch.object(knowledge_executor, "get_thresholds_for_metrics") as fetch:
            grounded, sources, _records = build_knowledge_grounding(
                user_question="what is VOC?",
                knowledge_cards=[],
                live_sensor_data=None,
                guideline_records=[],
            )
        fetch.assert_not_called()
        self.assertNotIn("Threshold Assessment", grounded)
        self.assertEqual(sources, [])

    def test_supplied_records_are_kept_alongside_the_fetched_ones(self):
        extra = [{"index": 9, "metric": "co2", "source_key": "RESET_AIR_V2",
                  "source_label": "RESET Air", "citation_tier": "regulatory",
                  "threshold_value": 1000, "threshold_unit": "ppm", "threshold_type": "max"}]
        _grounded, _sources, records = self._grounding(_SINGLE_METRIC_PAYLOAD, records=extra)
        keys = {r.get("source_key") for r in records}
        self.assertIn("RESET_AIR_V2", keys)
        self.assertIn("WHO_IAQ_VOC_PPM", keys)

    def test_records_are_not_duplicated(self):
        _grounded, _sources, records = self._grounding(
            _SINGLE_METRIC_PAYLOAD, records=list(_VOC_SOURCE)
        )
        self.assertEqual(len([r for r in records if r.get("source_key") == "WHO_IAQ_VOC_PPM"]), 1)

    def test_readings_helper_handles_a_non_dict_payload(self):
        self.assertEqual(_readings_from_live_data(None), {})
        self.assertEqual(_readings_from_live_data([1, 2, 3]), {})


class CardDirectiveTests(unittest.TestCase):
    def test_directive_defers_the_verdict_to_the_computed_section(self):
        text = CARD_TOOL_RESPONSE_DIRECTIVE.lower()
        self.assertIn("threshold assessment", text)
        self.assertIn("never from your own judgement", text)

    def test_directive_carries_the_verdict_rules_but_not_the_completeness_rules(self):
        # A definition question must not be made to recite the whole pollutant pack; that
        # is what METRIC COMPLETENESS would demand, and it belongs to assessment answers.
        self.assertIn("THRESHOLD VERDICTS", CARD_TOOL_RESPONSE_DIRECTIVE)
        self.assertNotIn("METRIC COMPLETENESS", CARD_TOOL_RESPONSE_DIRECTIVE)

    def test_directive_no_longer_carries_its_own_recommendation_rule(self):
        # The sixth clause competing with the role blocks' action policy; it survived an
        # earlier sweep only because its wording differed. See prompting/role_prompts.py.
        self.assertNotIn(
            "do not provide recommendations unless", CARD_TOOL_RESPONSE_DIRECTIVE.lower()
        )


if __name__ == "__main__":
    unittest.main()
