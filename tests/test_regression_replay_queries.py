import os
import sys
import unittest
from unittest.mock import patch


TEST_DIR = os.path.dirname(__file__)
SERVER_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)
REPO_DIR = os.path.abspath(os.path.join(SERVER_DIR, ".."))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from query_routing.intent_classifier import IntentType, RouteDecision
from query_routing.llm_router_planner import (
    AnswerStrategy,
    DecompositionTemplate,
    IntentCategory,
    RoutePlan,
)
from query_routing.query_orchestrator import execute_query


class RegressionReplayQueriesTests(unittest.TestCase):
    @patch("query_routing.query_orchestrator.get_route_plan")
    def test_replay_ambiguous_follow_up_returns_clarify(self, mock_get_route_plan):
        mock_get_route_plan.return_value = RoutePlan(
            decision=RouteDecision(
                intent=IntentType.AGGREGATION_DB,
                confidence=0.29,
                reason="ambiguous_reference",
            ),
            intent_category=IntentCategory.STRUCTURED_FACTUAL_DB,
            route_source="llm_planner",
            planner_model="qwen3:30b",
            planner_fallback_used=False,
            planner_fallback_reason=None,
            planner_raw={},
            planner_parameters={"response_mode": "db"},
            answer_strategy=AnswerStrategy.DIRECT,
        )

        result = execute_query("How is it over there?", 4, "smart_lab")
        self.assertEqual(result["timescale"], "clarify")
        self.assertEqual(result["metadata"]["executor"], "clarify_gate")
        self.assertTrue(result["metadata"]["clarification_required"])

    @patch("query_routing.query_orchestrator.answer_env_question_with_metadata")
    @patch("query_routing.query_orchestrator.get_route_plan")
    def test_replay_definition_query_routes_to_knowledge(self, mock_get_route_plan, mock_answer):
        mock_get_route_plan.return_value = RoutePlan(
            decision=RouteDecision(
                intent=IntentType.DEFINITION_EXPLANATION,
                confidence=0.95,
                reason="definition",
            ),
            intent_category=IntentCategory.SEMANTIC_EXPLANATORY,
            route_source="llm_planner",
            planner_model="qwen3:30b",
            planner_fallback_used=False,
            planner_fallback_reason=None,
            planner_raw={},
            planner_parameters={"response_mode": "knowledge_only"},
            answer_strategy=AnswerStrategy.DIRECT,
        )
        mock_answer.return_value = {
            "answer": "IEQ summarizes indoor environmental quality dimensions.",
            "cards_retrieved": 2,
            "knowledge_cards_retrieved": 2,
            "evidence": {
                "evidence_kind": "knowledge_qa",
                "metric_aliases": [],
                "provenance_sources": [],
                "confidence_notes": [],
                "recommendation_allowed": True,
            },
        }

        result = execute_query("What does IEQ mean?", 4, None)
        self.assertEqual(result["metadata"]["executor"], "knowledge_qa")
        self.assertEqual((result["metadata"].get("evidence") or {}).get("evidence_kind"), "knowledge_qa")

    @patch("query_routing.query_orchestrator.answer_env_question_with_metadata")
    @patch("query_routing.query_orchestrator.run_db_query")
    @patch("query_routing.query_orchestrator.get_route_plan")
    def test_replay_two_task_decomposition_is_stable(self, mock_get_route_plan, mock_run_db, mock_answer):
        mock_get_route_plan.return_value = RoutePlan(
            decision=RouteDecision(
                intent=IntentType.COMPARISON_DB,
                confidence=0.89,
                reason="comparison_plus_interpretation",
            ),
            intent_category=IntentCategory.ANALYTICAL_VISUALIZATION,
            route_source="llm_planner",
            planner_model="qwen3:30b",
            planner_fallback_used=False,
            planner_fallback_reason=None,
            planner_raw={},
            planner_parameters={"response_mode": "db", "decomposition_template": "trend_interpretation"},
            answer_strategy=AnswerStrategy.DECOMPOSE,
            secondary_intents=(IntentType.DEFINITION_EXPLANATION,),
            decomposition_template=DecompositionTemplate.TREND_INTERPRETATION,
        )
        mock_run_db.return_value = {
            "answer": "smart_lab had lower PM2.5 than concrete_lab.",
            "timescale": "24hour",
            "cards_retrieved": 1,
            "llm_used": True,
            "time_window": {"label": "last 24 hours", "start": "x", "end": "y"},
            "resolved_lab_name": "smart_lab",
            "sources": [],
            "visualization_type": "bar",
            "chart": {"type": "bar"},
            "forecast": None,
            "correlation": None,
            "data": [],
            "evidence": {
                "evidence_kind": "db_query",
                "metric_aliases": ["pm25"],
                "provenance_sources": [],
                "confidence_notes": [],
                "recommendation_allowed": True,
            },
        }
        mock_answer.return_value = {
            "answer": "This suggests stronger ventilation performance in smart_lab.",
            "cards_retrieved": 1,
            "knowledge_cards_retrieved": 1,
            "evidence": {
                "evidence_kind": "knowledge_qa",
                "metric_aliases": [],
                "provenance_sources": [],
                "confidence_notes": [],
                "recommendation_allowed": True,
            },
        }

        result = execute_query(
            "Compare PM2.5 for smart_lab and concrete_lab, then explain what it implies.",
            4,
            "smart_lab",
        )
        self.assertEqual(result["metadata"]["executor"], "decompose")
        self.assertEqual(result["metadata"]["decomposition_template"], "trend_interpretation")
        self.assertIn("Interpretation:", result["answer"])


if __name__ == "__main__":
    unittest.main()
