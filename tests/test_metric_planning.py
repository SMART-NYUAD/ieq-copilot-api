"""One metric decision per question, and a pack that cannot be truncated below itself.

The pack used to be chosen once and then re-decided by each handler with its own
hardcoded slice. Several of those slices cut a pack short: a comfort comparison was capped
at 8 against a 10-metric pack and silently lost ``sound`` and ``light`` — the two metrics
that make it a comfort assessment rather than an air-quality one.

The scope is now the router's decision (``RoutePlan.metric_scope``), with the keyword
classifier kept for the path where the router LLM was unreachable.
"""

import os
import sys
import unittest

TEST_DIR = os.path.dirname(__file__)
SERVER_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

from executors.db_support.metric_planning import (
    _PACKS,
    SCOPE_AIR_QUALITY,
    SCOPE_COMFORT,
    SCOPE_DIAGNOSTIC,
    SCOPE_FULL,
    SCOPE_IEQ_INDEX,
    SCOPE_NAMED,
    VALID_METRIC_SCOPES,
    classify_metric_scope,
    plan_metrics,
    with_ieq_sub_indices,
)
from query_routing.intent_classifier import IntentType


def _plan(question, explicit=(), hinted=(), intent=IntentType.CURRENT_STATUS_DB, **kwargs):
    return plan_metrics(
        question=question,
        explicit_metrics=list(explicit),
        hinted_metrics=list(hinted),
        intent=intent,
        **kwargs,
    )


class PackIntegrityTests(unittest.TestCase):
    """The invariant the old code stated in a comment and enforced nowhere."""

    def test_no_pack_is_truncated_below_its_own_length(self):
        for scope, pack in _PACKS.items():
            with self.subTest(scope=scope):
                plan = _plan("anything", metric_scope=scope)
                self.assertEqual(plan.selected, pack)

    def test_comfort_keeps_sound_and_light(self):
        # The regression: these two are what distinguish comfort from air quality.
        plan = _plan(
            "compare how comfortable smart_lab was this week versus last week",
            intent=IntentType.COMPARISON_DB,
        )
        self.assertEqual(plan.scope, SCOPE_COMFORT)
        self.assertIn("sound", plan.selected)
        self.assertIn("light", plan.selected)

    def test_named_metrics_never_displace_a_pack(self):
        # Extras trail the pack and fall outside the limit — a pack is a complete answer,
        # not a starting point.
        plan = _plan(
            "how is the air quality, and how loud is it?",
            explicit=["sound"],
            metric_scope=SCOPE_AIR_QUALITY,
        )
        self.assertEqual(plan.selected, _PACKS[SCOPE_AIR_QUALITY])
        self.assertIn("sound", plan.metrics)

    def test_every_pack_metric_is_resolvable(self):
        from executors import metric_registry

        for scope, pack in _PACKS.items():
            for metric in pack:
                with self.subTest(scope=scope, metric=metric):
                    self.assertIsNotNone(metric_registry.metric_column(metric))


class RouterSuppliedScopeTests(unittest.TestCase):
    def test_router_scope_overrides_the_keyword_guess(self):
        # Phrasing the keyword classifier reads as a bare CO2 question...
        inferred = _plan("what is the co2?", explicit=["co2"], hinted=["co2"])
        self.assertEqual(inferred.scope, SCOPE_NAMED)
        # ...but the router understood it as a comfort question.
        supplied = _plan(
            "what is the co2?", explicit=["co2"], hinted=["co2"], metric_scope=SCOPE_COMFORT
        )
        self.assertEqual(supplied.scope, SCOPE_COMFORT)
        self.assertFalse(supplied.scope_inferred)

    def test_missing_scope_falls_back_to_inference(self):
        plan = _plan("how is the air quality?", metric_scope=None)
        self.assertTrue(plan.scope_inferred)
        self.assertEqual(plan.scope, SCOPE_AIR_QUALITY)

    def test_unrecognised_scope_falls_back_rather_than_failing(self):
        plan = _plan("how is the air quality?", metric_scope="everything_please")
        self.assertTrue(plan.scope_inferred)
        self.assertEqual(plan.scope, SCOPE_AIR_QUALITY)

    def test_diagnostic_outranks_a_narrower_supplied_scope(self):
        # The router returns both signals together for "why is the IEQ low?" — answering
        # from the index family alone would report the score without its drivers.
        plan = _plan(
            "why is the IEQ low?",
            hinted=["ieq"],
            is_diagnostic=True,
            metric_scope=SCOPE_IEQ_INDEX,
        )
        self.assertEqual(plan.scope, SCOPE_DIAGNOSTIC)
        for metric in ("co2", "pm25", "voc", "humidity", "temperature"):
            self.assertIn(metric, plan.selected)

    def test_scope_vocabulary_matches_the_packs(self):
        # The router prompt offers exactly these; a pack must exist for each non-named one.
        self.assertEqual(VALID_METRIC_SCOPES - {SCOPE_NAMED}, set(_PACKS))


class KeywordFallbackTests(unittest.TestCase):
    """Behaviour of the inference path, unchanged from the pre-refactor classifier."""

    def test_named_metric_stays_named(self):
        self.assertEqual(_plan("what is the co2?", explicit=["co2"]).scope, SCOPE_NAMED)

    def test_air_quality_phrasing(self):
        self.assertEqual(_plan("how is the air quality?").scope, SCOPE_AIR_QUALITY)

    def test_ieq_ask_is_the_score_family_not_pollutants(self):
        plan = _plan(
            "Give me the IEQ data from May 1st to May 8th",
            explicit=["ieq"],
            hinted=["ieq", "co2", "pm25"],
            intent=IntentType.AGGREGATION_DB,
        )
        self.assertEqual(plan.scope, SCOPE_IEQ_INDEX)
        self.assertNotIn("co2", plan.selected)
        for sub in ("iaq", "itc", "iac", "iil"):
            self.assertIn(sub, plan.selected)

    def test_diagnostic_hint_wins_over_phrasing(self):
        # Phrasing the keyword heuristic does not read as diagnostic: without the hint it
        # collapses to the IEQ family and never sees the pollutants driving the score.
        question = "What is the main driver making the IEQ bad?"
        without_hint = _plan(question, hinted=["ieq"])
        self.assertEqual(without_hint.scope, SCOPE_IEQ_INDEX)
        self.assertNotIn("co2", without_hint.selected)

        with_hint = _plan(question, hinted=["ieq"], is_diagnostic=True)
        self.assertEqual(with_hint.scope, SCOPE_DIAGNOSTIC)
        self.assertIn("co2", with_hint.selected)

    def test_full_assessment_phrasing(self):
        self.assertEqual(
            _plan("give me a complete assessment of smart_lab", intent=IntentType.AGGREGATION_DB).scope,
            SCOPE_FULL,
        )

    def test_pollutant_trend_widens_to_air_quality(self):
        plan = _plan(
            "how has co2 trended last week?",
            explicit=["co2"],
            hinted=["co2"],
            intent=IntentType.AGGREGATION_DB,
        )
        self.assertEqual(plan.scope, SCOPE_AIR_QUALITY)

    def test_classify_is_callable_on_its_own(self):
        self.assertEqual(
            classify_metric_scope(
                question="is it comfortable in smart_lab?",
                explicit_metrics=[],
                hinted_metrics=[],
                intent=IntentType.CURRENT_STATUS_DB,
            ),
            SCOPE_COMFORT,
        )


class NamedScopeTests(unittest.TestCase):
    def test_named_metrics_are_kept_in_order(self):
        plan = _plan(
            "temperature and humidity please", explicit=["temperature", "humidity"]
        )
        self.assertEqual(plan.selected, ["temperature", "humidity"])

    def test_unknown_metric_is_dropped(self):
        plan = _plan("what is the unobtainium?", explicit=["unobtainium", "co2"])
        self.assertEqual(plan.selected, ["co2"])

    def test_named_list_is_bounded(self):
        plan = _plan(
            "everything named",
            explicit=["co2", "pm25", "voc", "humidity", "temperature", "sound", "light", "ieq"],
        )
        self.assertLessEqual(len(plan.selected), 6)


class IeqSubIndexExpansionTests(unittest.TestCase):
    def test_composite_gains_its_breakdown(self):
        self.assertEqual(
            with_ieq_sub_indices(["ieq", "co2"]),
            ["ieq", "co2", "iaq", "itc", "iac", "iil"],
        )

    def test_no_composite_is_left_alone(self):
        self.assertEqual(with_ieq_sub_indices(["co2", "pm25"]), ["co2", "pm25"])

    def test_existing_sub_indices_are_not_duplicated(self):
        self.assertEqual(with_ieq_sub_indices(["ieq", "iaq"]).count("iaq"), 1)


if __name__ == "__main__":
    unittest.main()
