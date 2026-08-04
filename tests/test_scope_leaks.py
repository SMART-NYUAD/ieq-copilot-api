"""A question's scope must survive to the answer.

Two reported symptoms, one shape: the scope decision was made correctly and then widened or
relabelled by something downstream.

**"what about the air?" answered with lighting.** The router picked `air_quality`, the pack
resolved to `[co2, pm25, voc, humidity, ieq]` — no light. Then `with_ieq_sub_indices` saw the
composite `ieq` in the pack and appended ALL FOUR sub-indices, including `iil` (illumination)
and `iac` (acoustic). The threshold assessment computed a verdict for IIL, and
`_METRIC_COMPLETENESS` requires a flagged metric to appear, so reporting lighting in an air
answer became mandatory. The model was obeying; the pack had been widened behind it.

**"how is the comfort today?" answered with "the air quality in smart_lab today is good".**
`db_response_directive` routed comfort questions to the air-quality directive, which opens
"you are answering a current air-quality point lookup" and says to "center on pollutants".
The comfort metric pack was fetched correctly and then described as an air assessment — right
data, wrong subject. This one dates to the first commit, not to any recent change.
"""

import os
import sys
import unittest

TEST_DIR = os.path.dirname(__file__)
SERVER_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

from executors.db_support.metric_planning import (  # noqa: E402
    SCOPE_AIR_QUALITY,
    SCOPE_COMFORT,
    SCOPE_DIAGNOSTIC,
    SCOPE_FULL,
    SCOPE_IEQ_INDEX,
    plan_metrics,
    with_ieq_sub_indices,
)
from executors.db_support.response_helpers import db_response_directive  # noqa: E402
from prompting.db_prompts import (  # noqa: E402
    DB_TOOL_RESPONSE_DIRECTIVE_AIR_QUALITY_POINT_LOOKUP,
    DB_TOOL_RESPONSE_DIRECTIVE_COMFORT,
    DB_TOOL_RESPONSE_DIRECTIVE_IEQ,
)
from query_routing.intent_classifier import IntentType  # noqa: E402

_NON_AIR_SUB_INDICES = ("itc", "iac", "iil")


class SubIndexExpansionTests(unittest.TestCase):
    def _air_pack(self):
        return plan_metrics(
            question="what about the air?", explicit_metrics=[], hinted_metrics=[],
            intent=IntentType.CURRENT_STATUS_DB, metric_scope=SCOPE_AIR_QUALITY,
        )

    def test_air_scope_keeps_ieq_and_its_air_sub_index(self):
        # The fix must not cost the composite score or the air-quality sub-index — IAQ *is*
        # the air dimension, and IEQ is the summary the answer opens on.
        expanded = with_ieq_sub_indices(self._air_pack().selected, SCOPE_AIR_QUALITY)
        self.assertIn("ieq", expanded)
        self.assertIn("iaq", expanded)

    def test_air_scope_does_not_pull_in_thermal_acoustic_or_visual(self):
        expanded = with_ieq_sub_indices(self._air_pack().selected, SCOPE_AIR_QUALITY)
        for sub in _NON_AIR_SUB_INDICES:
            self.assertNotIn(sub, expanded, f"{sub} has nothing to do with air")

    def test_an_ieq_question_still_gets_the_whole_breakdown(self):
        # The expansion exists because a bare IEQ number cannot explain itself. When IEQ is
        # the SUBJECT, all four dimensions belong.
        pack = plan_metrics(
            question="what is the IEQ score?", explicit_metrics=[], hinted_metrics=[],
            intent=IntentType.POINT_LOOKUP_DB, metric_scope=SCOPE_IEQ_INDEX,
        )
        expanded = with_ieq_sub_indices(pack.selected, SCOPE_IEQ_INDEX)
        for sub in ("iaq",) + _NON_AIR_SUB_INDICES:
            self.assertIn(sub, expanded)

    def test_comfort_and_wide_scopes_keep_every_dimension(self):
        # Comfort genuinely spans thermal, acoustic and visual; diagnostic needs everything.
        for scope in (SCOPE_COMFORT, SCOPE_DIAGNOSTIC, SCOPE_FULL):
            expanded = with_ieq_sub_indices(["ieq"], scope)
            for sub in ("iaq",) + _NON_AIR_SUB_INDICES:
                self.assertIn(sub, expanded, f"{scope}/{sub}")

    def test_unknown_scope_keeps_the_historical_behaviour(self):
        # Direct callers that pass no scope must not silently lose sub-indices.
        expanded = with_ieq_sub_indices(["ieq"], None)
        for sub in ("iaq",) + _NON_AIR_SUB_INDICES:
            self.assertIn(sub, expanded)

    def test_expansion_is_a_no_op_without_the_composite(self):
        self.assertEqual(with_ieq_sub_indices(["co2", "pm25"], SCOPE_AIR_QUALITY), ["co2", "pm25"])

    def test_expansion_never_drops_a_requested_metric(self):
        pack = self._air_pack().selected
        expanded = with_ieq_sub_indices(pack, SCOPE_AIR_QUALITY)
        self.assertTrue(set(pack).issubset(set(expanded)))


class RowFilterTests(unittest.TestCase):
    """Scoping the request is not enough — the upstream API ignores it.

    The sensor API returns every metric column whatever was asked for, so a correctly
    scoped air-quality query still came back carrying itc/iac/iil. They reached the
    Threshold Assessment, got verdicts, and _METRIC_COMPLETENESS then made reporting them
    mandatory. Every layer behaved correctly on data that should not have been there.
    """

    _ROW = {
        "lab_space": "smart_lab", "bucket": "2026-08-04T13:00:00+04:00",
        "co2": 454.0, "pm25": 7.2, "voc": 0.06, "humidity": 51.0,
        "ieq": 79.0, "iaq": 94.0, "itc": 94.1, "iac": 64.1, "iil": 39.8,
    }
    _AIR = ["co2", "pm25", "voc", "humidity", "ieq", "iaq"]

    def test_unrequested_metrics_are_dropped(self):
        from executors.db_query_executor import _restrict_rows_to_metrics

        row = _restrict_rows_to_metrics([dict(self._ROW)], self._AIR)[0]
        for leaked in ("itc", "iac", "iil"):
            self.assertNotIn(leaked, row)

    def test_requested_metrics_and_bookkeeping_survive(self):
        from executors.db_query_executor import _restrict_rows_to_metrics

        row = _restrict_rows_to_metrics([dict(self._ROW)], self._AIR)[0]
        for kept in self._AIR + ["lab_space", "bucket"]:
            self.assertIn(kept, row)

    def test_a_comfort_scope_keeps_every_comfort_dimension(self):
        from executors.db_query_executor import _restrict_rows_to_metrics

        comfort = ["ieq", "itc", "iaq", "iac", "iil", "temperature", "humidity",
                   "co2", "pm25", "voc", "sound", "light"]
        row = _restrict_rows_to_metrics([dict(self._ROW)], comfort)[0]
        for kept in ("itc", "iac", "iil"):
            self.assertIn(kept, row, "lighting and acoustics ARE comfort dimensions")

    def test_unrecognised_row_shapes_are_left_alone(self):
        # Point-lookup and aggregate rows use generic columns. Filtering must never empty a
        # shape it does not recognise.
        from executors.db_query_executor import _restrict_rows_to_metrics

        odd = {"lab_space": "smart_lab", "bucket": "t", "value": 0.06, "avg_value": 0.05}
        self.assertEqual(_restrict_rows_to_metrics([dict(odd)], ["voc"])[0], odd)

    def test_empty_metric_list_is_a_no_op(self):
        from executors.db_query_executor import _restrict_rows_to_metrics

        self.assertEqual(_restrict_rows_to_metrics([dict(self._ROW)], [])[0], self._ROW)


class ComfortDirectiveTests(unittest.TestCase):
    def _directive(self, question, intent=IntentType.CURRENT_STATUS_DB):
        return db_response_directive(intent, question=question)

    def test_comfort_questions_no_longer_get_the_air_quality_directive(self):
        for question in (
            "how is the comfort today?",
            "is it comfortable in here?",
            "how comfortable is the space?",
        ):
            directive = self._directive(question)
            self.assertIs(directive, DB_TOOL_RESPONSE_DIRECTIVE_COMFORT, question)
            self.assertIsNot(
                directive, DB_TOOL_RESPONSE_DIRECTIVE_AIR_QUALITY_POINT_LOOKUP, question
            )

    def test_air_quality_questions_keep_theirs(self):
        for question in ("how is the air quality today?", "how is the air quality?"):
            self.assertIs(
                self._directive(question),
                DB_TOOL_RESPONSE_DIRECTIVE_AIR_QUALITY_POINT_LOOKUP,
                question,
            )

    def test_an_ieq_question_still_wins_over_comfort(self):
        # The IEQ branch is checked first on purpose: an explicit index ask wants the score
        # family, not a prose comfort assessment.
        self.assertIs(self._directive("what is the IEQ score?"), DB_TOOL_RESPONSE_DIRECTIVE_IEQ)

    def test_comfort_directive_does_not_frame_itself_as_air_quality(self):
        text = DB_TOOL_RESPONSE_DIRECTIVE_COMFORT.lower()
        self.assertIn("comfort assessment", text)
        self.assertIn("never the headline by default", text)
        self.assertNotIn("you are answering a current air-quality point lookup", text)

    def test_comfort_directive_covers_every_dimension(self):
        text = DB_TOOL_RESPONSE_DIRECTIVE_COMFORT.lower()
        for dimension in ("thermal", "acoustic", "visual", "noisy", "lit"):
            self.assertIn(dimension, text)

    def test_comfort_directive_carries_the_completeness_rules(self):
        # It renders an overall verdict, so the worst-dimension rule and the
        # may-not-drop-a-flagged-metric rule both apply, exactly as for air quality.
        self.assertIn("METRIC COMPLETENESS", DB_TOOL_RESPONSE_DIRECTIVE_COMFORT)
        self.assertIn("THRESHOLD VERDICTS", DB_TOOL_RESPONSE_DIRECTIVE_COMFORT)

    def test_comfort_directive_keeps_the_sub_index_polarity_rule(self):
        # A high ITC means comfortable, not hot. This inversion caused a real wrong answer
        # and must survive into any directive that mentions the sub-indices.
        text = DB_TOOL_RESPONSE_DIRECTIVE_COMFORT
        self.assertIn("HIGHER = BETTER", text)
        self.assertIn("IAC is acoustic comfort, not air quality", text)


if __name__ == "__main__":
    unittest.main()
