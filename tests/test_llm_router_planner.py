import os
import sys
import unittest
from unittest.mock import patch


TEST_DIR = os.path.dirname(__file__)
SERVER_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

from query_routing.intent_classifier import IntentType
from query_routing.llm_router_planner import (
    AnswerStrategy,
    DecompositionTemplate,
    IntentCategory,
    extract_query_signals,
    plan_route,
)


class LlmRouterPlannerTests(unittest.TestCase):
    def test_query_scope_class_non_domain_for_general_time_question(self):
        signals = extract_query_signals("What day is today?", lab_name=None)
        self.assertEqual(signals.get("query_scope_class"), "non_domain")
        self.assertFalse(bool(signals.get("asks_for_db_facts")))

    def test_query_scope_class_ambiguous_for_under_scoped_question(self):
        signals = extract_query_signals("How is it?", lab_name=None)
        self.assertEqual(signals.get("query_scope_class"), "ambiguous")

    @patch("query_routing.llm_router_planner._call_router_planner")
    def test_plan_route_accepts_minimal_planner_contract(self, mock_call):
        mock_call.return_value = {
            "intent_category": "structured_factual_db",
            "intent": "aggregation_db",
            "confidence": 0.81,
        }

        plan = plan_route("Average CO2 this week in smart_lab")

        self.assertEqual(plan.route_source, "llm_planner")
        self.assertEqual(plan.decision.intent, IntentType.AGGREGATION_DB)
        self.assertEqual(plan.answer_strategy, AnswerStrategy.DIRECT)
        self.assertEqual(plan.secondary_intents, tuple())
        self.assertIn("metrics_priority", plan.planner_parameters)
        self.assertIn("response_mode", plan.planner_parameters)
        self.assertNotIn("needs_clarification", plan.planner_parameters)

    @patch("query_routing.llm_router_planner._call_router_planner")
    def test_plan_route_success(self, mock_call):
        mock_call.return_value = {
            "intent_category": "analytical_visualization",
            "intent": "comparison_db",
            "confidence": 0.92,
            "reason": "explicit_compare_request",
            "metrics_priority": ["ieq", "co2", "pm25"],
            "response_mode": "db",
            "needs_cards": False,
            "card_topics": [],
            "max_cards": 1,
        }

        plan = plan_route("Compare smart_lab vs concrete_lab this month")

        self.assertEqual(plan.route_source, "llm_planner")
        self.assertFalse(plan.planner_fallback_used)
        self.assertEqual(plan.intent_category, IntentCategory.ANALYTICAL_VISUALIZATION)
        self.assertEqual(plan.decision.intent, IntentType.COMPARISON_DB)

    @patch("query_routing.llm_router_planner._call_router_planner")
    def test_non_domain_scope_blocks_db_intent(self, mock_call):
        mock_call.return_value = {
            "intent_category": "structured_factual_db",
            "intent": "current_status_db",
            "confidence": 0.95,
            "reason": "time_hint",
            "response_mode": "db",
            "metrics_priority": ["ieq"],
            "needs_cards": False,
            "card_topics": [],
            "max_cards": 1,
        }

        plan = plan_route("What day is today?")
        self.assertEqual(plan.decision.intent, IntentType.DEFINITION_EXPLANATION)
        self.assertEqual(plan.intent_category, IntentCategory.SEMANTIC_EXPLANATORY)
        self.assertEqual(plan.planner_parameters.get("response_mode"), "knowledge_only")
        self.assertEqual(
            (plan.planner_parameters.get("query_signals") or {}).get("query_scope_class"),
            "non_domain",
        )

    @patch("query_routing.llm_router_planner._call_router_planner")
    def test_plan_route_invalid_output_falls_back(self, mock_call):
        mock_call.return_value = {
            "intent_category": "prediction",
            "intent": "definition_explanation",
            "confidence": 0.5,
            "reason": "bad_alignment",
            "metrics_priority": ["ieq"],
            "response_mode": "db",
            "needs_cards": False,
            "card_topics": [],
            "max_cards": 1,
        }

        plan = plan_route("Predict PM2.5 for next week")

        self.assertTrue(plan.planner_fallback_used)
        self.assertEqual(plan.route_source, "planner_rule_fallback")
        self.assertIn("planner_error", str(plan.planner_fallback_reason or ""))

    @patch("query_routing.llm_router_planner._call_router_planner")
    def test_plan_route_timeout_falls_back(self, mock_call):
        mock_call.side_effect = TimeoutError("timed out")

        plan = plan_route("Average CO2 this week")

        self.assertTrue(plan.planner_fallback_used)
        self.assertEqual(plan.route_source, "planner_rule_fallback")
        self.assertIn("planner_error", str(plan.planner_fallback_reason or ""))

    @patch("query_routing.llm_router_planner._call_router_planner")
    def test_comfort_query_uses_full_comfort_metric_pack(self, mock_call):
        mock_call.return_value = {
            "intent_category": "structured_factual_db",
            "intent": "current_status_db",
            "confidence": 0.89,
            "reason": "comfort_assessment",
            "response_mode": "db",
            "metrics_priority": ["ieq", "temperature", "humidity"],
            "needs_cards": False,
            "card_topics": [],
            "max_cards": 1,
        }
        plan = plan_route("Is it comfortable in the smart_lab?")
        metrics = plan.planner_parameters.get("metrics_priority") or []
        self.assertEqual(
            metrics[:7],
            ["ieq", "temperature", "humidity", "co2", "pm25", "tvoc", "sound"],
        )

    def test_plan_route_legacy_mode(self):
        old_mode = os.getenv("ROUTER_MODE")
        os.environ["ROUTER_MODE"] = "legacy"
        try:
            plan = plan_route("Why is PM2.5 high today?")
            self.assertTrue(plan.planner_fallback_used)
            self.assertEqual(plan.route_source, "legacy_classifier")
            self.assertEqual(plan.planner_fallback_reason, "router_mode_legacy")
        finally:
            if old_mode is None:
                os.environ.pop("ROUTER_MODE", None)
            else:
                os.environ["ROUTER_MODE"] = old_mode

    @patch("query_routing.llm_router_planner._call_router_planner")
    def test_forces_db_mode_when_deterministic_signals_require_facts(self, mock_call):
        mock_call.return_value = {
            "intent_category": "structured_factual_db",
            "intent": "aggregation_db",
            "confidence": 0.9,
            "reason": "db_query",
            "response_mode": "knowledge_only",
            "metrics_priority": ["ieq"],
            "needs_cards": True,
            "card_topics": ["definitions"],
            "max_cards": 2,
        }
        plan = plan_route("Average CO2 this week in smart_lab")
        self.assertEqual(plan.planner_parameters.get("response_mode"), "db")

    @patch("query_routing.llm_router_planner._call_router_planner")
    def test_planner_parameter_normalization(self, mock_call):
        mock_call.return_value = {
            "intent_category": "semantic_explanatory",
            "intent": "definition_explanation",
            "confidence": 0.92,
            "reason": "normalize_fields",
            "response_mode": "knowledge_only",
            "metrics_priority": ["bad_metric", "ieq", "ieq"],
            "needs_cards": "yes",
            "card_topics": ["Definitions", "metric explanations", "bad_topic"],
            "max_cards": 99,
        }
        plan = plan_route("What does IEQ mean?")
        params = plan.planner_parameters
        self.assertEqual(
            params.get("metrics_priority"),
            ["ieq", "co2", "pm25", "humidity", "tvoc"],
        )
        self.assertTrue(params.get("needs_cards"))
        self.assertEqual(params.get("card_topics"), ["definitions", "metric_explanations"])
        self.assertEqual(params.get("max_cards"), 4)

    @patch("query_routing.llm_router_planner._call_router_planner")
    def test_fallback_routes_match_prompt_examples(self, mock_call):
        mock_call.side_effect = TimeoutError("force deterministic fallback")
        self.assertIn(
            plan_route("Why is PM2.5 high today in smart_lab?").decision.intent,
            {IntentType.AGGREGATION_DB, IntentType.ANOMALY_ANALYSIS_DB},
        )
        self.assertEqual(
            plan_route("What does IEQ mean?").decision.intent,
            IntentType.DEFINITION_EXPLANATION,
        )
        self.assertEqual(
            plan_route("Is smart_lab comfortable right now?").decision.intent,
            IntentType.CURRENT_STATUS_DB,
        )
        self.assertEqual(
            plan_route("Compare smart_lab vs concrete_lab this month").decision.intent,
            IntentType.COMPARISON_DB,
        )
        self.assertEqual(
            plan_route("Predict PM2.5 for next week").decision.intent,
            IntentType.FORECAST_DB,
        )

    @patch("query_routing.llm_router_planner._call_router_planner")
    def test_strategy_and_secondary_intents_are_parsed(self, mock_call):
        mock_call.return_value = {
            "intent_category": "analytical_visualization",
            "intent": "anomaly_analysis_db",
            "confidence": 0.86,
            "strategy": "decompose",
            "secondary_intents": ["aggregation_db", "comparison_db", "forecast_db"],
        }

        plan = plan_route("Why was PM2.5 unstable in smart_lab this week?")
        self.assertEqual(plan.answer_strategy, AnswerStrategy.DECOMPOSE)
        self.assertEqual(plan.secondary_intents, (IntentType.AGGREGATION_DB,))
        self.assertEqual(plan.decomposition_template, DecompositionTemplate.ANOMALY_EXPLANATION)

    @patch("query_routing.llm_router_planner._call_router_planner")
    def test_decompose_not_allowed_for_forecast_falls_back_to_direct(self, mock_call):
        mock_call.return_value = {
            "intent_category": "prediction",
            "intent": "forecast_db",
            "confidence": 0.9,
            "strategy": "decompose",
            "secondary_intents": ["definition_explanation"],
        }

        plan = plan_route("Predict PM2.5 for next week")
        self.assertEqual(plan.answer_strategy, AnswerStrategy.DIRECT)
        self.assertEqual(plan.secondary_intents, tuple())
        self.assertIsNone(plan.decomposition_template)

    @patch("query_routing.llm_router_planner._call_router_planner")
    def test_invalid_planner_json_attempts_repair_before_fallback(self, mock_call):
        mock_call.side_effect = [
            ValueError("missing_json_object"),
            {
                "intent_category": "structured_factual_db",
                "intent": "current_status_db",
                "confidence": 0.77,
            },
        ]

        plan = plan_route("Current CO2 in smart_lab")
        self.assertFalse(plan.planner_fallback_used)
        self.assertEqual(plan.decision.intent, IntentType.CURRENT_STATUS_DB)
        self.assertEqual(mock_call.call_count, 2)

    @patch("query_routing.llm_router_planner._call_router_planner")
    def test_timeout_fallback_reason_uses_taxonomy(self, mock_call):
        mock_call.side_effect = TimeoutError("planner timed out")

        plan = plan_route("Average CO2 this week")
        reason = str(plan.planner_fallback_reason or "")
        self.assertIn("planner_error:timeout", reason)

    @patch("query_routing.llm_router_planner._call_router_planner")
    def test_parse_error_fallback_reason_uses_taxonomy(self, mock_call):
        mock_call.side_effect = ValueError("missing_json_object")

        plan = plan_route("Summarize comfort trends")
        reason = str(plan.planner_fallback_reason or "")
        self.assertIn("planner_error:parse_error", reason)


if __name__ == "__main__":
    unittest.main()
