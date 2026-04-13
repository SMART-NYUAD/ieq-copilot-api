import os
import sys
import unittest
import asyncio
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
    extract_query_signals,
    plan_route,
)
from query_routing.query_orchestrator import execute_query
from query_routing.observability import get_observability_snapshot, reset_observability_metrics
from query_routing.query_orchestrator import stream_query
from query_routing.router_types import RouteDecisionContract, RouteExecutor


class GeneralQaRoutingTests(unittest.TestCase):
    def setUp(self):
        reset_observability_metrics()

    @patch("query_routing.llm_router_planner._call_router_planner")
    def test_planner_sets_knowledge_only_mode_for_general_qa(self, mock_call):
        mock_call.return_value = {
            "intent_category": "semantic_explanatory",
            "intent": "definition_explanation",
            "confidence": 0.91,
            "reason": "definition_guidance_question",
            "metrics_priority": ["ieq"],
            "response_mode": "knowledge_only",
            "needs_cards": True,
            "card_topics": ["definitions"],
            "max_cards": 2,
        }
        plan = plan_route("What does IEQ mean and what guidance should I follow?")
        self.assertEqual(plan.decision.intent, IntentType.DEFINITION_EXPLANATION)
        self.assertEqual(plan.planner_parameters.get("response_mode"), "knowledge_only")
        self.assertTrue(plan.planner_parameters.get("query_signals", {}).get("is_general_knowledge_question"))
        self.assertFalse(plan.planner_parameters.get("query_signals", {}).get("asks_for_db_facts"))
        self.assertGreaterEqual(mock_call.call_count, 1)

    @patch("query_routing.llm_router_planner._call_router_planner")
    def test_planner_accepts_explicit_response_mode(self, mock_call):
        mock_call.return_value = {
            "intent_category": "semantic_explanatory",
            "intent": "definition_explanation",
            "confidence": 0.95,
            "reason": "knowledge_question",
            "response_mode": "knowledge_only",
            "metrics_priority": ["ieq"],
            "needs_cards": True,
            "card_topics": ["definitions"],
            "max_cards": 2,
        }
        plan = plan_route("Explain IEQ comfort bands.")
        self.assertEqual(plan.planner_parameters.get("response_mode"), "knowledge_only")
        self.assertGreaterEqual(mock_call.call_count, 1)

    @patch("query_routing.llm_router_planner._call_router_planner")
    def test_definition_question_with_default_lab_hint_stays_knowledge_mode(self, mock_call):
        mock_call.return_value = {
            "intent_category": "semantic_explanatory",
            "intent": "definition_explanation",
            "confidence": 0.9,
            "reason": "definition",
            "response_mode": "knowledge_only",
            "metrics_priority": ["ieq"],
            "needs_cards": True,
            "card_topics": ["definitions"],
            "max_cards": 2,
        }
        plan = plan_route("What is air quality and what does it mean?", lab_name="smart_lab")
        signals = plan.planner_parameters.get("query_signals", {})
        self.assertTrue(signals.get("is_general_knowledge_question"))
        self.assertFalse(signals.get("asks_for_db_facts"))
        self.assertGreaterEqual(mock_call.call_count, 1)

    @patch("query_routing.llm_router_planner._call_router_planner")
    def test_comfort_question_in_specific_lab_forces_db_facts_signal(self, mock_call):
        mock_call.return_value = {
            "intent_category": "structured_factual_db",
            "intent": "current_status_db",
            "confidence": 0.9,
            "reason": "comfort_assessment",
            "response_mode": "knowledge_only",
            "metrics_priority": ["ieq"],
            "needs_cards": False,
            "card_topics": [],
            "max_cards": 1,
        }
        plan = plan_route("Is it comfortable in the smart_lab?", lab_name=None)
        signals = plan.planner_parameters.get("query_signals", {})
        self.assertTrue(signals.get("is_comfort_assessment_phrase"))
        self.assertTrue(signals.get("asks_for_db_facts"))
        self.assertEqual(plan.planner_parameters.get("response_mode"), "db")

    @patch("query_routing.query_orchestrator.run_db_query")
    @patch("query_routing.query_orchestrator.answer_env_question_with_metadata")
    @patch("query_routing.query_orchestrator.get_route_plan")
    def test_orchestrator_uses_knowledge_executor(
        self, mock_route_plan, mock_answer_with_metadata, mock_run_db
    ):
        mock_route_plan.return_value = RoutePlan(
            decision=RouteDecision(
                intent=IntentType.DEFINITION_EXPLANATION,
                confidence=0.9,
                reason="general_knowledge",
            ),
            intent_category=IntentCategory.SEMANTIC_EXPLANATORY,
            route_source="llm_planner",
            planner_model="qwen3:30b",
            planner_fallback_used=False,
            planner_fallback_reason=None,
            planner_raw={},
            planner_parameters={"response_mode": "knowledge_only"},
        )
        mock_answer_with_metadata.return_value = {
            "answer": "IEQ stands for indoor environmental quality.",
            "cards_retrieved": 3,
            "knowledge_cards_retrieved": 3,
        }

        result = execute_query(
            question="What does IEQ mean?",
            k=4,
            lab_name=None,
        )

        self.assertEqual(result["metadata"]["executor"], "knowledge_qa")
        self.assertEqual((result["metadata"].get("evidence") or {}).get("evidence_kind"), "knowledge_qa")
        self.assertEqual(result["timescale"], "knowledge")
        self.assertEqual(result["cards_retrieved"], 3)
        self.assertEqual(result["metadata"]["knowledge_cards_retrieved"], 3)
        self.assertFalse(result["metadata"]["intent_rerouted_to_db"])
        mock_run_db.assert_not_called()

    @patch("query_routing.query_orchestrator.run_db_query")
    @patch("query_routing.query_orchestrator.answer_env_question_with_metadata")
    @patch("query_routing.query_orchestrator.get_route_plan")
    def test_orchestrator_prefers_db_when_qa_needs_scoped_facts(
        self, mock_route_plan, mock_answer_with_metadata, mock_run_db
    ):
        mock_route_plan.return_value = RoutePlan(
            decision=RouteDecision(
                intent=IntentType.DEFINITION_EXPLANATION,
                confidence=0.9,
                reason="qa_with_scope",
            ),
            intent_category=IntentCategory.SEMANTIC_EXPLANATORY,
            route_source="llm_planner",
            planner_model="qwen3:30b",
            planner_fallback_used=False,
            planner_fallback_reason=None,
            planner_raw={},
            planner_parameters={
                "response_mode": "knowledge_only",
                "needs_measured_data": True,
                "has_explicit_scope": True,
                "resolved_lab_name": "smart_lab",
                "query_signals": {"asks_for_db_facts": True},
            },
            has_explicit_scope=True,
            resolved_lab_name="smart_lab",
        )
        mock_run_db.return_value = {
            "answer": "Scoped grounded answer",
            "timescale": "1hour",
            "llm_used": True,
            "time_window": {"label": "this week", "start": "x", "end": "y"},
            "resolved_lab_name": "smart_lab",
            "sources": [],
            "visualization_type": "none",
            "chart": None,
            "forecast": None,
            "correlation": None,
            "data": [],
        }

        result = execute_query(
            question="What is IEQ in smart lab this week?",
            k=4,
            lab_name="smart lab",
        )

        self.assertEqual(result["metadata"]["executor"], "db_query")
        self.assertEqual((result["metadata"].get("evidence") or {}).get("evidence_kind"), "db_query")
        mock_answer_with_metadata.assert_not_called()
        mock_run_db.assert_called_once()

    @patch("query_routing.query_orchestrator.run_db_query")
    @patch("query_routing.query_orchestrator.answer_env_question_with_metadata")
    @patch("query_routing.query_orchestrator.get_route_plan")
    def test_orchestrator_blocks_knowledge_mode_for_comfort_scoped_query(
        self, mock_route_plan, mock_answer_with_metadata, mock_run_db
    ):
        mock_route_plan.return_value = RoutePlan(
            decision=RouteDecision(
                intent=IntentType.DEFINITION_EXPLANATION,
                confidence=0.9,
                reason="comfort_assessment",
            ),
            intent_category=IntentCategory.SEMANTIC_EXPLANATORY,
            route_source="llm_planner",
            planner_model="qwen3:30b",
            planner_fallback_used=False,
            planner_fallback_reason=None,
            planner_raw={},
            planner_parameters={
                "response_mode": "knowledge_only",
                "needs_measured_data": True,
                "has_explicit_scope": True,
                "resolved_lab_name": "smart_lab",
                "query_signals": {"asks_for_db_facts": True, "is_comfort_assessment_phrase": True},
            },
            has_explicit_scope=True,
            resolved_lab_name="smart_lab",
        )
        mock_run_db.return_value = {
            "answer": "Comfort assessment from measured facts",
            "timescale": "1hour",
            "llm_used": True,
            "time_window": {"label": "last 24 hours", "start": "x", "end": "y"},
            "resolved_lab_name": "smart_lab",
            "sources": [],
            "visualization_type": "none",
            "chart": None,
            "forecast": None,
            "correlation": None,
            "data": [],
        }
        result = execute_query(
            question="Is it comfortable in the smart_lab?",
            k=4,
            lab_name=None,
        )
        self.assertEqual(result["metadata"]["executor"], "db_query")
        self.assertEqual((result["metadata"].get("evidence") or {}).get("evidence_kind"), "db_query")
        mock_answer_with_metadata.assert_not_called()
        mock_run_db.assert_called_once()

    @patch("query_routing.query_orchestrator.run_db_query")
    @patch("query_routing.query_orchestrator.answer_env_question_with_metadata")
    @patch("query_routing.query_orchestrator.get_route_plan")
    def test_orchestrator_returns_clarify_response_when_confidence_low(
        self, mock_route_plan, mock_answer_with_metadata, mock_run_db
    ):
        mock_route_plan.return_value = RoutePlan(
            decision=RouteDecision(
                intent=IntentType.AGGREGATION_DB,
                confidence=0.31,
                reason="ambiguous_request",
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

        result = execute_query(
            question="How is it over there?",
            k=4,
            lab_name="smart_lab",
        )

        self.assertEqual(result["timescale"], "clarify")
        self.assertEqual(result["metadata"]["executor"], "clarify_gate")
        self.assertEqual((result["metadata"].get("evidence") or {}).get("evidence_kind"), "clarify_gate")
        self.assertTrue(result["metadata"]["clarification_required"])
        mock_answer_with_metadata.assert_not_called()
        mock_run_db.assert_not_called()

    @patch("query_routing.query_orchestrator.run_db_query")
    @patch("query_routing.query_orchestrator.answer_env_question_with_metadata")
    @patch("query_routing.query_orchestrator.get_route_plan")
    def test_orchestrator_clarify_opt_out_executes_db(
        self, mock_route_plan, mock_answer_with_metadata, mock_run_db
    ):
        mock_route_plan.return_value = RoutePlan(
            decision=RouteDecision(
                intent=IntentType.AGGREGATION_DB,
                confidence=0.31,
                reason="ambiguous_request",
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
        mock_run_db.return_value = {
            "answer": "Scoped grounded answer",
            "timescale": "1hour",
            "llm_used": True,
            "time_window": {"label": "last 24 hours", "start": "x", "end": "y"},
            "resolved_lab_name": "smart_lab",
            "sources": [],
            "visualization_type": "none",
            "chart": None,
            "forecast": None,
            "correlation": None,
            "data": [],
        }

        result = execute_query(
            question="How is it over there?",
            k=4,
            lab_name="smart_lab",
            allow_clarify=False,
        )

        self.assertEqual(result["metadata"]["executor"], "db_query")
        self.assertEqual((result["metadata"].get("evidence") or {}).get("evidence_kind"), "db_query")
        self.assertFalse(result["metadata"]["clarification_required"])
        mock_answer_with_metadata.assert_not_called()
        mock_run_db.assert_called_once()

    @patch("query_routing.query_orchestrator.run_db_query")
    @patch("query_routing.query_orchestrator.get_route_plan")
    def test_orchestrator_routes_to_clarify_on_db_invariant_violation(self, mock_route_plan, mock_run_db):
        mock_route_plan.return_value = RoutePlan(
            decision=RouteDecision(
                intent=IntentType.AGGREGATION_DB,
                confidence=0.9,
                reason="aggregation",
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
        mock_run_db.return_value = {
            "answer": "Please clarify metric/time/lab scope.",
            "timescale": "clarify",
            "llm_used": False,
            "time_window": {"label": "last 24 hours", "start": "x", "end": "y"},
            "resolved_lab_name": None,
            "sources": [],
            "visualization_type": "none",
            "chart": None,
            "forecast": None,
            "correlation": None,
            "data": [],
            "invariant_violation": {"allowed": False, "violations": ["metric_not_justified"]},
            "evidence": {
                "evidence_kind": "clarify_gate",
                "metric_aliases": [],
                "provenance_sources": [],
                "confidence_notes": ["db_invariant_violation"],
                "recommendation_allowed": False,
            },
        }
        result = execute_query(
            question="How is it there?",
            k=4,
            lab_name=None,
            allow_clarify=True,
        )
        self.assertEqual(result["timescale"], "clarify")
        self.assertEqual(result["metadata"]["executor"], "clarify_gate")
        self.assertTrue(result["metadata"]["clarification_required"])
        self.assertIn("db_invariant_violation", result["metadata"])

    @patch("query_routing.query_orchestrator.run_db_query")
    @patch("query_routing.query_orchestrator.answer_env_question_with_metadata")
    @patch("query_routing.query_orchestrator.get_route_plan")
    def test_orchestrator_executes_fixed_template_decomposition(
        self, mock_route_plan, mock_answer_with_metadata, mock_run_db
    ):
        mock_route_plan.return_value = RoutePlan(
            decision=RouteDecision(
                intent=IntentType.AGGREGATION_DB,
                confidence=0.9,
                reason="trend_with_interpretation",
            ),
            intent_category=IntentCategory.STRUCTURED_FACTUAL_DB,
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
            "answer": "CO2 trended upward this week.",
            "timescale": "1hour",
            "cards_retrieved": 1,
            "llm_used": True,
            "time_window": {"label": "this week", "start": "x", "end": "y"},
            "resolved_lab_name": "smart_lab",
            "sources": [],
            "visualization_type": "line",
            "chart": {"type": "line"},
            "forecast": None,
            "correlation": None,
            "data": [],
            "evidence": {
                "evidence_kind": "db_query",
                "metric_aliases": ["co2"],
                "provenance_sources": [],
                "confidence_notes": [],
                "recommendation_allowed": True,
            },
        }
        mock_answer_with_metadata.return_value = {
            "answer": "This likely reflects lower ventilation periods.",
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

        result = execute_query(
            question="How did CO2 trend this week and what does it imply?",
            k=4,
            lab_name="smart_lab",
        )

        self.assertEqual(result["metadata"]["executor"], "decompose")
        self.assertEqual(result["metadata"]["decomposition_template"], "trend_interpretation")
        self.assertEqual(result["metadata"]["decomposition_task_count"], 2)
        self.assertIn("Interpretation:", result["answer"])
        self.assertEqual((result["metadata"].get("evidence") or {}).get("evidence_kind"), "decomposed_composition")
        mock_run_db.assert_called_once()
        mock_answer_with_metadata.assert_called_once()

    @patch("query_routing.query_orchestrator.run_db_query")
    @patch("query_routing.query_orchestrator.get_route_plan")
    def test_response_keeps_answer_without_critic_rewrite_on_recommendation_prompt(self, mock_route_plan, mock_run_db):
        mock_route_plan.return_value = RoutePlan(
            decision=RouteDecision(
                intent=IntentType.AGGREGATION_DB,
                confidence=0.92,
                reason="aggregation",
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
        mock_run_db.return_value = {
            "answer": "CO2 average was 650 ppm.",
            "timescale": "1hour",
            "cards_retrieved": 0,
            "llm_used": True,
            "time_window": {"label": "this week", "start": "x", "end": "y"},
            "resolved_lab_name": "smart_lab",
            "sources": [],
            "visualization_type": "none",
            "chart": None,
            "forecast": None,
            "correlation": None,
            "data": [],
            "evidence": {
                "evidence_kind": "db_query",
                "metric_aliases": ["co2"],
                "provenance_sources": [],
                "confidence_notes": [],
                "recommendation_allowed": True,
            },
        }

        result = execute_query(
            question="What should we do to improve CO2 this week?",
            k=4,
            lab_name="smart_lab",
        )

        self.assertNotIn("critic_status", result["metadata"])
        self.assertNotIn("critic_issues", result["metadata"])
        self.assertEqual(result["answer"], "CO2 average was 650 ppm.")

    @patch("query_routing.query_orchestrator.run_db_query")
    @patch("query_routing.query_orchestrator.get_route_plan")
    def test_response_does_not_block_on_evidence_kind_mismatch(self, mock_route_plan, mock_run_db):
        mock_route_plan.return_value = RoutePlan(
            decision=RouteDecision(
                intent=IntentType.AGGREGATION_DB,
                confidence=0.95,
                reason="aggregation",
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
        mock_run_db.return_value = {
            "answer": "CO2 average was 650 ppm.",
            "timescale": "1hour",
            "cards_retrieved": 0,
            "llm_used": True,
            "time_window": {"label": "this week", "start": "x", "end": "y"},
            "resolved_lab_name": "smart_lab",
            "sources": [],
            "visualization_type": "none",
            "chart": None,
            "forecast": None,
            "correlation": None,
            "data": [],
            "evidence": {
                "evidence_kind": "knowledge_qa",
                "metric_aliases": [],
                "provenance_sources": [],
                "confidence_notes": [],
                "recommendation_allowed": True,
            },
        }

        result = execute_query(
            question="Average CO2 this week",
            k=4,
            lab_name="smart_lab",
        )

        self.assertNotIn("critic_status", result["metadata"])
        self.assertNotIn("critic_blocked", result["metadata"])
        self.assertNotIn("critic_issues", result["metadata"])
        self.assertEqual(result["answer"], "CO2 average was 650 ppm.")

    @patch("query_routing.query_orchestrator.run_db_query")
    @patch("query_routing.query_orchestrator.get_route_plan")
    def test_response_keeps_answer_without_date_consistency_critic(self, mock_route_plan, mock_run_db):
        mock_route_plan.return_value = RoutePlan(
            decision=RouteDecision(
                intent=IntentType.AGGREGATION_DB,
                confidence=0.95,
                reason="aggregation",
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
        mock_run_db.return_value = {
            "answer": "From Feb 27, 2026, 10:15 AM to Feb 28, 2026, 10:15 AM, IEQ stayed stable.",
            "timescale": "1hour",
            "cards_retrieved": 0,
            "llm_used": True,
            "time_window": {
                "label": "last 24 hours",
                "start": "2026-03-27T14:15:28+04:00",
                "end": "2026-03-28T14:15:28+04:00",
                "display_start": "Mar 27, 2026, 2:15 PM GMT+4",
                "display_end": "Mar 28, 2026, 2:15 PM GMT+4",
            },
            "resolved_lab_name": "smart_lab",
            "sources": [],
            "visualization_type": "none",
            "chart": None,
            "forecast": None,
            "correlation": None,
            "data": [],
            "evidence": {
                "evidence_kind": "db_query",
                "metric_aliases": ["ieq"],
                "provenance_sources": [],
                "confidence_notes": [],
                "recommendation_allowed": True,
            },
        }

        result = execute_query(
            question="How was IEQ in the last 24 hours?",
            k=4,
            lab_name="smart_lab",
        )

        self.assertNotIn("critic_status", result["metadata"])
        self.assertNotIn("critic_blocked", result["metadata"])
        self.assertNotIn("critic_issues", result["metadata"])
        self.assertEqual(
            result["answer"],
            "From Feb 27, 2026, 10:15 AM to Feb 28, 2026, 10:15 AM, IEQ stayed stable.",
        )

    @patch("query_routing.query_orchestrator.run_db_query")
    @patch("query_routing.query_orchestrator.get_route_plan")
    def test_db_default_window_metadata_for_point_and_aggregation(self, mock_route_plan, mock_run_db):
        mock_route_plan.return_value = RoutePlan(
            decision=RouteDecision(
                intent=IntentType.POINT_LOOKUP_DB,
                confidence=0.91,
                reason="point_lookup",
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
        mock_run_db.return_value = {
            "answer": "Current CO2 is 710 ppm.",
            "timescale": "1hour",
            "cards_retrieved": 0,
            "llm_used": True,
            "time_window": {"label": "last 1 hour", "start": "x", "end": "y"},
            "resolved_lab_name": "smart_lab",
            "sources": [],
            "visualization_type": "none",
            "chart": None,
            "forecast": None,
            "correlation": None,
            "data": [],
            "evidence": {
                "evidence_kind": "db_query",
                "metric_aliases": ["co2"],
                "provenance_sources": [],
                "confidence_notes": [],
                "recommendation_allowed": True,
            },
        }

        point_result = execute_query("Current CO2 in smart_lab", 4, "smart_lab")
        self.assertEqual(point_result["metadata"]["time_window"]["label"], "last 1 hour")

        mock_route_plan.return_value = RoutePlan(
            decision=RouteDecision(
                intent=IntentType.AGGREGATION_DB,
                confidence=0.91,
                reason="aggregation",
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
        mock_run_db.return_value = {
            "answer": "Average CO2 was 680 ppm.",
            "timescale": "24hour",
            "cards_retrieved": 0,
            "llm_used": True,
            "time_window": {"label": "last 24 hours", "start": "x", "end": "y"},
            "resolved_lab_name": "smart_lab",
            "sources": [],
            "visualization_type": "none",
            "chart": None,
            "forecast": None,
            "correlation": None,
            "data": [],
            "evidence": {
                "evidence_kind": "db_query",
                "metric_aliases": ["co2"],
                "provenance_sources": [],
                "confidence_notes": [],
                "recommendation_allowed": True,
            },
        }

        agg_result = execute_query("Average CO2 in smart_lab", 4, "smart_lab")
        self.assertEqual(agg_result["metadata"]["time_window"]["label"], "last 24 hours")

    @patch("query_routing.query_orchestrator.answer_env_question_with_metadata")
    @patch("query_routing.query_orchestrator.get_route_plan")
    def test_follow_up_context_inheritance_stays_knowledge_mode(self, mock_route_plan, mock_answer_with_metadata):
        mock_route_plan.return_value = RoutePlan(
            decision=RouteDecision(
                intent=IntentType.DEFINITION_EXPLANATION,
                confidence=0.9,
                reason="follow_up",
            ),
            intent_category=IntentCategory.SEMANTIC_EXPLANATORY,
            route_source="llm_planner",
            planner_model="qwen3:30b",
            planner_fallback_used=False,
            planner_fallback_reason=None,
            planner_raw={},
            planner_parameters={"response_mode": "knowledge_only"},
        )
        mock_answer_with_metadata.return_value = {
            "answer": "Yes, IEQ combines air, thermal, and comfort indicators.",
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

        follow_up_question = (
            "Can you summarize that in one sentence?\n\n"
            "Previous conversation context (most recent last):\n"
            "User: What does IEQ mean?\n"
            "Assistant: IEQ stands for indoor environmental quality."
        )
        result = execute_query(follow_up_question, 4, None)

        self.assertEqual(result["metadata"]["executor"], "knowledge_qa")
        self.assertEqual((result["metadata"].get("evidence") or {}).get("evidence_kind"), "knowledge_qa")

    @patch("query_routing.query_orchestrator.run_db_query")
    @patch("query_routing.query_orchestrator.get_route_plan")
    def test_db_follow_up_inherits_lab_and_time_from_context(self, mock_route_plan, mock_run_db):
        mock_route_plan.return_value = RoutePlan(
            decision=RouteDecision(
                intent=IntentType.CURRENT_STATUS_DB,
                confidence=0.9,
                reason="follow_up_metric_only",
            ),
            intent_category=IntentCategory.STRUCTURED_FACTUAL_DB,
            route_source="llm_planner",
            planner_model="qwen3:30b",
            planner_fallback_used=False,
            planner_fallback_reason=None,
            planner_raw={},
            planner_parameters={
                "response_mode": "db",
                "query_signals": {
                    "has_lab_reference": False,
                    "has_time_window_hint": False,
                    "has_metric_reference": True,
                    "asks_for_db_facts": True,
                },
            },
            answer_strategy=AnswerStrategy.DIRECT,
        )
        mock_run_db.return_value = {
            "answer": "PM2.5 stayed low this week in shores_office.",
            "timescale": "1hour",
            "llm_used": True,
            "time_window": {"label": "this week", "start": "x", "end": "y"},
            "resolved_lab_name": "shores_office",
            "sources": [],
            "visualization_type": "none",
            "chart": None,
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

        result = execute_query(
            question="And what about the PM2.5?",
            k=4,
            lab_name=None,
            conversation_context=(
                "Previous conversation context (most recent last):\n"
                "User: Which metric is driving poor IEQ in shores_office this week?\n"
                "Assistant: CO2 is the main driver this week."
            ),
        )

        kwargs = mock_run_db.call_args.kwargs
        self.assertEqual(kwargs.get("lab_name"), "shores_office")
        self.assertIn("this week", str(kwargs.get("question") or "").lower())
        self.assertTrue(bool(result["metadata"].get("memory_carryover_applied")))
        self.assertEqual(result["metadata"].get("memory_carried_lab_name"), "shores_office")
        self.assertEqual(result["metadata"].get("memory_carried_time_phrase"), "this week")

    @patch("query_routing.query_orchestrator.run_db_query")
    @patch("query_routing.query_orchestrator.get_route_plan")
    def test_db_follow_up_inherits_scope_for_data_backed_clarification_selection(
        self, mock_route_plan, mock_run_db
    ):
        mock_route_plan.return_value = RoutePlan(
            decision=RouteDecision(
                intent=IntentType.CURRENT_STATUS_DB,
                confidence=0.9,
                reason="clarification_selection",
            ),
            intent_category=IntentCategory.STRUCTURED_FACTUAL_DB,
            route_source="llm_planner",
            planner_model="qwen3:30b",
            planner_fallback_used=False,
            planner_fallback_reason=None,
            planner_raw={},
            planner_parameters={
                "response_mode": "db",
                "query_signals": {
                    "has_lab_reference": False,
                    "has_time_window_hint": False,
                    "has_metric_reference": False,
                    "asks_for_db_facts": True,
                },
            },
            answer_strategy=AnswerStrategy.DIRECT,
        )
        mock_run_db.return_value = {
            "answer": "Using prior scope in smart_lab for last week.",
            "timescale": "1hour",
            "llm_used": True,
            "time_window": {"label": "last week", "start": "x", "end": "y"},
            "resolved_lab_name": "smart_lab",
            "sources": [],
            "visualization_type": "none",
            "chart": None,
            "forecast": None,
            "correlation": None,
            "data": [],
            "evidence": {
                "evidence_kind": "db_query",
                "metric_aliases": ["ieq"],
                "provenance_sources": [],
                "confidence_notes": [],
                "recommendation_allowed": True,
            },
        }

        result = execute_query(
            question="a data-backed answer with exact values from the database",
            k=4,
            lab_name=None,
            conversation_context=(
                "Previous conversation context (most recent last):\n"
                "User: What is the air quality in the smart lab for the last week?\n"
                "Assistant: It was good.\n"
                "User: Would you say it's good the air quality there?\n"
                "Assistant: Quick clarification: do you want (1) a data-backed answer with exact values from the database, "
                "or (2) a high-level conceptual explanation?"
            ),
        )

        kwargs = mock_run_db.call_args.kwargs
        self.assertEqual(kwargs.get("lab_name"), "smart_lab")
        self.assertIn("last week", str(kwargs.get("question") or "").lower())
        self.assertTrue(bool(result["metadata"].get("memory_carryover_applied")))
        self.assertEqual(result["metadata"].get("memory_carried_lab_name"), "smart_lab")
        self.assertEqual(result["metadata"].get("memory_carried_time_phrase"), "last week")

    @patch("query_routing.query_orchestrator.run_db_query")
    @patch("query_routing.query_orchestrator.get_route_plan")
    def test_db_metric_only_followup_inherits_lab_and_time(self, mock_route_plan, mock_run_db):
        mock_route_plan.return_value = RoutePlan(
            decision=RouteDecision(
                intent=IntentType.CURRENT_STATUS_DB,
                confidence=0.9,
                reason="metric_only_followup",
            ),
            intent_category=IntentCategory.STRUCTURED_FACTUAL_DB,
            route_source="llm_planner",
            planner_model="qwen3:30b",
            planner_fallback_used=False,
            planner_fallback_reason=None,
            planner_raw={},
            planner_parameters={
                "response_mode": "db",
                "query_signals": {
                    "has_lab_reference": False,
                    "has_time_window_hint": False,
                    "has_metric_reference": True,
                    "asks_for_db_facts": True,
                },
            },
            answer_strategy=AnswerStrategy.DIRECT,
        )
        mock_run_db.return_value = {
            "answer": "PM2.5 stayed low this week in shores_office.",
            "timescale": "1hour",
            "llm_used": True,
            "time_window": {"label": "this week", "start": "x", "end": "y"},
            "resolved_lab_name": "shores_office",
            "sources": [],
            "visualization_type": "none",
            "chart": None,
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

        result = execute_query(
            question="How is PM2.5?",
            k=4,
            lab_name=None,
            conversation_context=(
                "Previous conversation context (most recent last):\n"
                "User: What is the air quality in shores_office this week?\n"
                "Assistant: Air quality has been stable."
            ),
        )

        kwargs = mock_run_db.call_args.kwargs
        self.assertEqual(kwargs.get("lab_name"), "shores_office")
        self.assertIn("this week", str(kwargs.get("question") or "").lower())
        self.assertTrue(bool(result["metadata"].get("memory_carryover_applied")))
        self.assertEqual(result["metadata"].get("memory_carried_lab_name"), "shores_office")
        self.assertEqual(result["metadata"].get("memory_carried_time_phrase"), "this week")

    @patch("query_routing.query_orchestrator.run_db_query")
    @patch("query_routing.query_orchestrator.get_route_plan")
    def test_db_lab_only_clarification_reply_inherits_previous_scope(self, mock_route_plan, mock_run_db):
        mock_route_plan.return_value = RoutePlan(
            decision=RouteDecision(
                intent=IntentType.CURRENT_STATUS_DB,
                confidence=0.9,
                reason="lab_only_clarification_reply",
            ),
            intent_category=IntentCategory.STRUCTURED_FACTUAL_DB,
            route_source="llm_planner",
            planner_model="qwen3:30b",
            planner_fallback_used=False,
            planner_fallback_reason=None,
            planner_raw={},
            planner_parameters={
                "response_mode": "db",
                "query_signals": {
                    "has_lab_reference": True,
                    "has_time_window_hint": False,
                    "has_metric_reference": False,
                    "asks_for_db_facts": True,
                },
            },
            answer_strategy=AnswerStrategy.DIRECT,
        )
        mock_run_db.return_value = {
            "answer": "Air quality was good in smart_lab last week.",
            "timescale": "1hour",
            "llm_used": True,
            "time_window": {"label": "last week", "start": "x", "end": "y"},
            "resolved_lab_name": "smart_lab",
            "sources": [],
            "visualization_type": "none",
            "chart": None,
            "forecast": None,
            "correlation": None,
            "data": [],
            "evidence": {
                "evidence_kind": "db_query",
                "metric_aliases": ["ieq"],
                "provenance_sources": [],
                "confidence_notes": [],
                "recommendation_allowed": True,
            },
        }

        result = execute_query(
            question="smart_lab",
            k=4,
            lab_name=None,
            conversation_context=(
                "Previous conversation context (most recent last):\n"
                "User: What is the air quality there for the last week?\n"
                "Assistant: I can answer this with measured data, but I need the lab first. "
                "Which lab should I use (for example: smart_lab, concrete_lab, or shores_office)?"
            ),
        )

        kwargs = mock_run_db.call_args.kwargs
        self.assertEqual(kwargs.get("lab_name"), "smart_lab")
        self.assertIn("last week", str(kwargs.get("question") or "").lower())
        self.assertTrue(bool(result["metadata"].get("memory_carryover_applied")))
        self.assertEqual(result["metadata"].get("memory_carried_time_phrase"), "last week")

    @patch("query_routing.query_orchestrator.run_db_query")
    @patch("query_routing.query_orchestrator.answer_env_question_with_metadata")
    @patch("query_routing.query_orchestrator.get_route_plan")
    def test_non_domain_scope_forces_knowledge_executor(
        self, mock_route_plan, mock_answer_with_metadata, mock_run_db
    ):
        mock_route_plan.return_value = RoutePlan(
            decision=RouteDecision(
                intent=IntentType.CURRENT_STATUS_DB,
                confidence=0.92,
                reason="planner_time_hint",
            ),
            intent_category=IntentCategory.STRUCTURED_FACTUAL_DB,
            route_source="llm_planner",
            planner_model="qwen3:30b",
            planner_fallback_used=False,
            planner_fallback_reason=None,
            planner_raw={},
            planner_parameters={
                "response_mode": "db",
                "query_signals": {"query_scope_class": "non_domain", "asks_for_db_facts": False},
            },
        )
        mock_answer_with_metadata.return_value = {
            "answer": "Today is Saturday.",
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
            question="What day is today?",
            k=4,
            lab_name=None,
        )

        self.assertEqual(result["metadata"]["executor"], "knowledge_qa")
        self.assertEqual(result["metadata"]["query_scope_class"], "non_domain")
        self.assertEqual((result["metadata"].get("evidence") or {}).get("evidence_kind"), "knowledge_qa")
        mock_run_db.assert_not_called()
        mock_answer_with_metadata.assert_called_once()

    @patch("query_routing.query_orchestrator.run_db_query")
    @patch("query_routing.query_orchestrator.answer_env_question_with_metadata")
    @patch("query_routing.query_orchestrator.get_route_plan")
    def test_non_domain_scope_keeps_llm_response_without_hardcoded_rewrite(
        self, mock_route_plan, mock_answer_with_metadata, mock_run_db
    ):
        mock_route_plan.return_value = RoutePlan(
            decision=RouteDecision(
                intent=IntentType.DEFINITION_EXPLANATION,
                confidence=0.88,
                reason="non_domain_scope",
            ),
            intent_category=IntentCategory.SEMANTIC_EXPLANATORY,
            route_source="llm_planner",
            planner_model="qwen3:30b",
            planner_fallback_used=False,
            planner_fallback_reason=None,
            planner_raw={},
            planner_parameters={
                "response_mode": "knowledge_only",
                "query_signals": {"query_scope_class": "non_domain", "asks_for_db_facts": False},
            },
        )
        mock_answer_with_metadata.return_value = {
            "answer": "I don't know from the available data.",
            "cards_retrieved": 0,
            "knowledge_cards_retrieved": 0,
            "evidence": {
                "evidence_kind": "knowledge_qa",
                "metric_aliases": [],
                "provenance_sources": [],
                "confidence_notes": [],
                "recommendation_allowed": True,
            },
        }

        result = execute_query(
            question="What day is today?",
            k=4,
            lab_name=None,
        )

        self.assertIn("I don't know from the available data.", result["answer"])
        self.assertFalse(bool(result["metadata"].get("scope_guardrail_applied")))
        self.assertEqual(result["metadata"]["executor"], "knowledge_qa")
        mock_run_db.assert_not_called()
        mock_answer_with_metadata.assert_called_once()

    @patch("query_routing.query_orchestrator.run_db_query")
    @patch("query_routing.query_orchestrator.answer_env_question_with_metadata")
    @patch("query_routing.query_orchestrator.get_route_plan")
    def test_non_domain_identity_query_returns_conversational_intro(
        self, mock_route_plan, mock_answer_with_metadata, mock_run_db
    ):
        mock_route_plan.return_value = RoutePlan(
            decision=RouteDecision(
                intent=IntentType.DEFINITION_EXPLANATION,
                confidence=0.9,
                reason="social_identity",
            ),
            intent_category=IntentCategory.SEMANTIC_EXPLANATORY,
            route_source="llm_planner",
            planner_model="qwen3:30b",
            planner_fallback_used=False,
            planner_fallback_reason=None,
            planner_raw={},
            planner_parameters={
                "response_mode": "knowledge_only",
                "query_signals": {
                    "query_scope_class": "non_domain",
                    "asks_for_db_facts": False,
                    "is_social_identity_query": True,
                },
            },
        )

        mock_answer_with_metadata.return_value = {
            "answer": "I am your campus IEQ assistant and I can help with both general questions and sensor-grounded environmental insights.",
            "cards_retrieved": 0,
            "knowledge_cards_retrieved": 0,
            "evidence": {
                "evidence_kind": "knowledge_qa",
                "metric_aliases": [],
                "provenance_sources": [],
                "confidence_notes": [],
                "recommendation_allowed": True,
            },
        }

        result = execute_query(
            question="Who are you?",
            k=4,
            lab_name=None,
        )

        self.assertIn("campus IEQ assistant", result["answer"])
        self.assertFalse(bool(result["metadata"].get("scope_guardrail_applied")))
        self.assertEqual(result["metadata"]["executor"], "knowledge_qa")
        mock_run_db.assert_not_called()
        mock_answer_with_metadata.assert_called_once()

    def test_query_scope_class_domain_for_scoped_ieq_question(self):
        signals = extract_query_signals("What is today's IEQ in smart_lab?", lab_name=None)
        self.assertEqual(signals.get("query_scope_class"), "domain")
        self.assertTrue(bool(signals.get("asks_for_db_facts")))

    def test_query_signals_ignore_appended_conversation_context(self):
        question = (
            "What does CO2 mean and how does it affect occupants?\n\n"
            "Previous conversation context (most recent last):\n"
            "User: show PM2.5 in shores_office for the last 24 hours\n"
            "Assistant: PM2.5 was low."
        )
        signals = extract_query_signals(question, lab_name=None)
        self.assertTrue(bool(signals.get("is_general_knowledge_question")))
        self.assertFalse(bool(signals.get("asks_for_db_facts")))

    def test_query_signals_treat_warning_trend_anomaly_as_domain_knowledge(self):
        question = "What is the difference between a warning trend and an anomaly?"
        signals = extract_query_signals(question, lab_name=None)
        self.assertTrue(bool(signals.get("is_general_knowledge_question")))
        self.assertEqual(signals.get("query_scope_class"), "ambiguous")
        self.assertFalse(bool(signals.get("asks_for_db_facts")))

    def test_query_signals_interpret_multi_metric_without_scope_is_general_knowledge(self):
        question = "How should I interpret PM2.5, TVOC, and humidity together?"
        signals = extract_query_signals(question, lab_name=None)
        self.assertTrue(bool(signals.get("is_general_knowledge_question")))
        self.assertEqual(signals.get("query_scope_class"), "ambiguous")
        self.assertFalse(bool(signals.get("asks_for_db_facts")))

    def test_query_signals_detect_shores_office_as_explicit_scope(self):
        question = "Give me a full picture of air quality in shores_office right now"
        signals = extract_query_signals(question, lab_name=None)
        self.assertTrue(bool(signals.get("has_lab_reference")))
        self.assertTrue(bool(signals.get("requests_current_measured_data")))
        self.assertIn("shores_office", signals.get("lab_candidates") or [])

    @patch("query_routing.query_orchestrator.run_db_query")
    @patch("query_routing.query_orchestrator.get_route_plan")
    def test_execute_query_publishes_observability_snapshot(self, mock_route_plan, mock_run_db):
        mock_route_plan.return_value = RoutePlan(
            decision=RouteDecision(
                intent=IntentType.AGGREGATION_DB,
                confidence=0.88,
                reason="aggregation",
            ),
            intent_category=IntentCategory.STRUCTURED_FACTUAL_DB,
            route_source="planner_rule_fallback",
            planner_model="qwen3:30b",
            planner_fallback_used=True,
            planner_fallback_reason="planner_error:timeout:TimeoutError",
            planner_raw={},
            planner_parameters={"response_mode": "db"},
            answer_strategy=AnswerStrategy.DIRECT,
        )
        mock_run_db.return_value = {
            "answer": "CO2 average was 640 ppm.",
            "timescale": "24hour",
            "cards_retrieved": 0,
            "llm_used": True,
            "time_window": {"label": "last 24 hours", "start": "x", "end": "y"},
            "resolved_lab_name": "smart_lab",
            "sources": [],
            "visualization_type": "none",
            "chart": None,
            "forecast": None,
            "correlation": None,
            "data": [],
            "evidence": {
                "evidence_kind": "db_query",
                "metric_aliases": ["co2"],
                "provenance_sources": [],
                "confidence_notes": [],
                "recommendation_allowed": True,
            },
        }

        result = execute_query("Average CO2 this week", 4, "smart_lab")
        obs = result["metadata"].get("observability") or {}
        self.assertIn("planner_fallback_rate", obs)
        self.assertIn("decomposition_template_usage", obs)
        self.assertIn("rollout_slo", result["metadata"])

        snapshot = get_observability_snapshot()
        self.assertIn("planner_total", snapshot)

    @patch("query_routing.query_orchestrator.stream_db_query")
    @patch("query_routing.query_orchestrator.get_route_plan")
    def test_stream_query_applies_conversation_memory_for_db_path(self, mock_get_route_plan, mock_stream_db_query):
        async def _fake_stream(*args, **kwargs):
            yield "ok"

        mock_get_route_plan.return_value = RoutePlan(
            decision=RouteDecision(
                intent=IntentType.CURRENT_STATUS_DB,
                confidence=0.9,
                reason="follow_up_metric_only",
            ),
            intent_category=IntentCategory.STRUCTURED_FACTUAL_DB,
            route_source="llm_planner",
            planner_model="qwen3:30b",
            planner_fallback_used=False,
            planner_fallback_reason=None,
            planner_raw={},
            planner_parameters={
                "response_mode": "db",
                "query_signals": {
                    "has_lab_reference": False,
                    "has_time_window_hint": False,
                    "has_metric_reference": True,
                    "asks_for_db_facts": True,
                },
            },
            answer_strategy=AnswerStrategy.DIRECT,
        )
        mock_stream_db_query.side_effect = _fake_stream

        async def _consume():
            chunks = []
            async for chunk in stream_query(
                question="And what about the PM2.5?",
                k=4,
                lab_name=None,
                conversation_context=(
                    "Previous conversation context (most recent last):\n"
                    "User: Which metric is driving poor IEQ in shores_office this week?\n"
                    "Assistant: CO2 is the main driver this week."
                ),
            ):
                chunks.append(chunk)
            return chunks

        _ = asyncio.run(_consume())
        stream_kwargs = mock_stream_db_query.call_args.kwargs
        self.assertEqual(stream_kwargs.get("lab_name"), "shores_office")
        self.assertIn("this week", str(stream_kwargs.get("question") or "").lower())

    @patch("query_routing.query_orchestrator.stream_db_query")
    @patch("query_routing.query_orchestrator.resolve_query_context")
    def test_stream_query_clarify_gate_yields_prompt_and_stops(self, mock_resolve_query_context, mock_stream_db_query):
        async def _fake_stream(*args, **kwargs):
            yield "should-not-be-called"

        clarify_plan = RoutePlan(
            decision=RouteDecision(
                intent=IntentType.AGGREGATION_DB,
                confidence=0.2,
                reason="ambiguous",
            ),
            intent_category=IntentCategory.STRUCTURED_FACTUAL_DB,
            route_source="llm_planner",
            planner_model="qwen3:30b",
            planner_fallback_used=False,
            planner_fallback_reason=None,
            planner_raw={},
            planner_parameters={
                "response_mode": "db",
                "query_signals": {
                    "query_scope_class": "ambiguous",
                    "asks_for_db_facts": False,
                },
            },
            answer_strategy=AnswerStrategy.CLARIFY,
        )
        mock_resolve_query_context.return_value = {
            "original_question": "How is it over there?",
            "effective_question": "How is it over there?",
            "effective_lab_name": None,
            "memory_carryover": {},
            "route_contract": RouteDecisionContract(
                latest_user_question="How is it over there?",
                latest_question_hash="abcd1234",
                policy_version="route-policy-v1",
                route_plan=clarify_plan,
                needs_measured_data=False,
                executor=RouteExecutor.CLARIFY_GATE,
                execution_intent=IntentType.AGGREGATION_DB,
                execution_intent_value=IntentType.AGGREGATION_DB.value,
                query_scope_class="ambiguous",
                rule_trace=("test_forced_clarify",),
            ),
        }
        mock_stream_db_query.side_effect = _fake_stream

        async def _consume():
            chunks = []
            async for chunk in stream_query(
                question="How is it over there?",
                k=4,
                lab_name=None,
                conversation_context="",
            ):
                chunks.append(chunk)
            return chunks

        chunks = asyncio.run(_consume())
        self.assertEqual(len(chunks), 1)
        self.assertIn("Just checking", chunks[0])
        mock_stream_db_query.assert_not_called()

    @patch("query_routing.query_orchestrator.stream_db_query")
    @patch("query_routing.query_orchestrator.answer_env_question_with_metadata")
    @patch("query_routing.query_orchestrator.run_db_query")
    @patch("query_routing.query_orchestrator.get_route_plan")
    def test_stream_query_decompose_streams_chunked_merged_answer(
        self,
        mock_get_route_plan,
        mock_run_db_query,
        mock_answer_with_metadata,
        mock_stream_db_query,
    ):
        async def _fake_stream(*args, **kwargs):
            yield "should-not-be-called"

        mock_get_route_plan.return_value = RoutePlan(
            decision=RouteDecision(
                intent=IntentType.AGGREGATION_DB,
                confidence=0.93,
                reason="needs_interpretation",
            ),
            intent_category=IntentCategory.STRUCTURED_FACTUAL_DB,
            route_source="llm_planner",
            planner_model="qwen3:30b",
            planner_fallback_used=False,
            planner_fallback_reason=None,
            planner_raw={},
            planner_parameters={"response_mode": "db", "query_signals": {"asks_for_db_facts": True}},
            answer_strategy=AnswerStrategy.DECOMPOSE,
            secondary_intents=(IntentType.DEFINITION_EXPLANATION,),
            decomposition_template=DecompositionTemplate.TREND_INTERPRETATION,
        )
        mock_run_db_query.return_value = {
            "answer": "Primary DB answer.",
            "timescale": "1hour",
            "cards_retrieved": 0,
            "llm_used": True,
            "time_window": {"label": "this week", "start": "x", "end": "y"},
            "resolved_lab_name": "smart_lab",
            "sources": [],
            "visualization_type": "none",
            "chart": None,
            "forecast": None,
            "correlation": None,
            "data": [],
            "evidence": {
                "evidence_kind": "db_query",
                "metric_aliases": ["co2"],
                "provenance_sources": [],
                "confidence_notes": [],
                "recommendation_allowed": True,
            },
        }
        mock_answer_with_metadata.return_value = {
            "answer": "Secondary interpretation.",
            "cards_retrieved": 0,
            "knowledge_cards_retrieved": 0,
            "evidence": {
                "evidence_kind": "knowledge_qa",
                "metric_aliases": [],
                "provenance_sources": [],
                "confidence_notes": [],
                "recommendation_allowed": True,
            },
        }
        mock_stream_db_query.side_effect = _fake_stream

        async def _consume():
            chunks = []
            async for chunk in stream_query(
                question="How did CO2 trend this week and what does it imply?",
                k=4,
                lab_name="smart_lab",
                conversation_context="",
            ):
                chunks.append(chunk)
            return chunks

        chunks = asyncio.run(_consume())
        streamed_answer = "".join(chunks)
        self.assertIn("Primary DB answer.", streamed_answer)
        self.assertIn("Interpretation:", streamed_answer)
        self.assertIn("Secondary interpretation.", streamed_answer)
        self.assertTrue(all(len(chunk) <= 8 for chunk in chunks))
        mock_stream_db_query.assert_not_called()


if __name__ == "__main__":
    unittest.main()
