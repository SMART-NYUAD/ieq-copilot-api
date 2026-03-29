import os
import sys
import unittest


TEST_DIR = os.path.dirname(__file__)
SERVER_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

from query_routing.observability import (
    evaluate_rollout_slo,
    get_observability_snapshot,
    record_endpoint_executor,
    record_critic_outcome,
    record_rollout_selection,
    record_route_plan,
    record_shadow_comparison,
    reset_observability_metrics,
)
from query_routing.intent_classifier import IntentType, RouteDecision
from query_routing.llm_router_planner import (
    AnswerStrategy,
    DecompositionTemplate,
    IntentCategory,
    RoutePlan,
)


class ObservabilityMetricsTests(unittest.TestCase):
    def setUp(self):
        reset_observability_metrics()

    def test_snapshot_reports_fallback_and_template_distribution(self):
        plan_fallback = RoutePlan(
            decision=RouteDecision(intent=IntentType.AGGREGATION_DB, confidence=0.4, reason="fallback"),
            intent_category=IntentCategory.STRUCTURED_FACTUAL_DB,
            route_source="planner_rule_fallback",
            planner_model="qwen3:30b",
            planner_fallback_used=True,
            planner_fallback_reason="planner_error:timeout:TimeoutError",
            planner_raw=None,
            planner_parameters={},
        )
        plan_success = RoutePlan(
            decision=RouteDecision(intent=IntentType.COMPARISON_DB, confidence=0.91, reason="ok"),
            intent_category=IntentCategory.ANALYTICAL_VISUALIZATION,
            route_source="llm_planner",
            planner_model="qwen3:30b",
            planner_fallback_used=False,
            planner_fallback_reason=None,
            planner_raw=None,
            planner_parameters={},
            answer_strategy=AnswerStrategy.DECOMPOSE,
            secondary_intents=(IntentType.DEFINITION_EXPLANATION,),
            decomposition_template=DecompositionTemplate.TREND_INTERPRETATION,
        )

        record_route_plan(plan_fallback)
        record_route_plan(plan_success)
        record_critic_outcome("pass")
        record_critic_outcome("warn")
        record_rollout_selection("policy")
        record_rollout_selection("legacy")
        record_shadow_comparison(sampled=True, active_executor="db_query", shadow_executor="knowledge_qa")
        record_shadow_comparison(sampled=True, active_executor="db_query", shadow_executor="db_query")
        record_endpoint_executor(
            latest_question_hash="abc123",
            endpoint_key="query_sync",
            executor="db_query",
        )
        record_endpoint_executor(
            latest_question_hash="abc123",
            endpoint_key="query_sync",
            executor="knowledge_qa",
        )

        snapshot = get_observability_snapshot()
        self.assertEqual(snapshot["planner_total"], 2)
        self.assertEqual(snapshot["planner_fallback_total"], 1)
        self.assertAlmostEqual(snapshot["planner_fallback_rate"], 0.5)
        self.assertIn("planner_error:timeout:TimeoutError", snapshot["planner_fallback_reason_distribution"])
        self.assertEqual(snapshot["critic_total"], 2)
        self.assertEqual(snapshot["critic_failure_total"], 1)
        self.assertAlmostEqual(snapshot["critic_failure_rate"], 0.5)
        self.assertEqual(snapshot["decomposition_template_usage"].get("trend_interpretation"), 1)
        self.assertEqual(snapshot["rollout_policy_total"], 1)
        self.assertEqual(snapshot["rollout_legacy_total"], 1)
        self.assertEqual(snapshot["shadow_total"], 2)
        self.assertEqual(snapshot["shadow_diff_total"], 1)
        self.assertAlmostEqual(snapshot["shadow_diff_rate"], 0.5)
        self.assertEqual(snapshot["sync_stream_total"], 2)
        self.assertEqual(snapshot["sync_stream_flip_total"], 1)

    def test_rollout_slo_gate_blocks_on_high_rates(self):
        snapshot = {
            "planner_fallback_rate": 0.12,
            "critic_failure_rate": 0.06,
            "shadow_diff_rate": 0.25,
            "sync_stream_flip_rate": 0.02,
        }
        gate = evaluate_rollout_slo(snapshot)
        self.assertFalse(gate["fallback_max_ok"])
        self.assertFalse(gate["critic_max_ok"])
        self.assertFalse(gate["shadow_max_ok"])
        self.assertFalse(gate["parity_max_ok"])
        self.assertTrue(gate["rollout_blocked"])


if __name__ == "__main__":
    unittest.main()
