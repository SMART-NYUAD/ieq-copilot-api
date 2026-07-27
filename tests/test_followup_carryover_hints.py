import os
import sys
import unittest


TEST_DIR = os.path.dirname(__file__)
SERVER_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

from query_routing.query_orchestrator import _build_planner_hints
from query_routing.llm_router_planner import RoutePlan
from query_routing.intent_classifier import IntentType
from executors.db_support.query_parsing import planner_metrics


def _route(metrics, fallback_used=False):
    return RoutePlan(
        intent=IntentType.AGGREGATION_DB,
        confidence=0.9,
        lab_name="smart_lab",
        time_phrase=None,
        model="test",
        fallback_used=fallback_used,
        second_lab_name=None,
        metrics=metrics,
        viewer_type=None,
        heatmap_action=None,
        heatmap_metric=None,
        download_format=None,
        download_metric=None,
        download_interval=None,
    )


class FollowupCarryoverHintsTests(unittest.TestCase):
    """Regex carry-over is emergency-only: it applies only when the LLM router was
    unreachable (``fallback_used``). When the router ran, ``resolved_question`` is the
    canonical carry-over, so the regex hints must NOT be injected on top of it."""

    def test_carried_metric_populates_metrics_priority_on_fallback(self):
        # LLM router unreachable: the prior turn's metric must reach the DB executor
        # via planner hints, since there is no resolved_question to carry it.
        hints = _build_planner_hints(_route([], fallback_used=True), carried_metric="temperature")
        self.assertEqual(hints["metrics_priority"], ["temperature"])
        # And it survives the DB executor's hint normalization.
        self.assertEqual(planner_metrics(hints), ["temperature"])

    def test_explicit_current_metric_wins_over_carried(self):
        hints = _build_planner_hints(
            _route(["humidity"], fallback_used=True), carried_metric="temperature"
        )
        self.assertEqual(hints["metrics_priority"], ["humidity"])

    def test_carried_pm25_normalizes_on_fallback(self):
        hints = _build_planner_hints(_route([], fallback_used=True), carried_metric="pm2.5")
        self.assertEqual(planner_metrics(hints), ["pm25"])

    def test_carried_time_phrase_forwarded_on_fallback(self):
        hints = _build_planner_hints(_route([], fallback_used=True), carried_time_phrase="june 2")
        self.assertEqual(hints.get("carried_time_phrase"), "june 2")

    def test_carryover_suppressed_when_router_ran(self):
        # Regression guard for the "what about on the 11" wrong-date bug: when the LLM
        # router ran (fallback_used=False), resolved_question is authoritative and the
        # regex carry-over must be suppressed so a stale prior date/metric can't override
        # the day the user actually asked for.
        hints = _build_planner_hints(
            _route([], fallback_used=False),
            carried_time_phrase="june 9",
            carried_metric="temperature",
        )
        self.assertNotIn("carried_time_phrase", hints)
        self.assertEqual(hints["metrics_priority"], [])

    def test_no_carry_leaves_hints_empty(self):
        hints = _build_planner_hints(_route([]))
        self.assertEqual(hints["metrics_priority"], [])
        self.assertNotIn("carried_time_phrase", hints)


if __name__ == "__main__":
    unittest.main()
