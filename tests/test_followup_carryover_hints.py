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


def _route(metrics):
    return RoutePlan(
        intent=IntentType.AGGREGATION_DB,
        confidence=0.9,
        lab_name="smart_lab",
        time_phrase=None,
        model="test",
        fallback_used=False,
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
    def test_carried_metric_populates_metrics_priority(self):
        # Follow-up like "what about last week?" — the router found no metric for
        # this turn, so the prior turn's metric must reach the DB executor via
        # planner hints (previously dropped entirely).
        hints = _build_planner_hints(_route([]), carried_metric="temperature")
        self.assertEqual(hints["metrics_priority"], ["temperature"])
        # And it survives the DB executor's hint normalization.
        self.assertEqual(planner_metrics(hints), ["temperature"])

    def test_explicit_current_metric_wins_over_carried(self):
        hints = _build_planner_hints(_route(["humidity"]), carried_metric="temperature")
        self.assertEqual(hints["metrics_priority"], ["humidity"])

    def test_carried_pm25_normalizes(self):
        hints = _build_planner_hints(_route([]), carried_metric="pm2.5")
        self.assertEqual(planner_metrics(hints), ["pm25"])

    def test_carried_time_phrase_still_forwarded(self):
        hints = _build_planner_hints(_route([]), carried_time_phrase="june 2")
        self.assertEqual(hints.get("carried_time_phrase"), "june 2")

    def test_no_carry_leaves_hints_empty(self):
        hints = _build_planner_hints(_route([]))
        self.assertEqual(hints["metrics_priority"], [])
        self.assertNotIn("carried_time_phrase", hints)


if __name__ == "__main__":
    unittest.main()
