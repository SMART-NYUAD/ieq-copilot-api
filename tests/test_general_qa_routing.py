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
    extract_query_signals,
    plan_route,
)
from query_routing.query_orchestrator import execute_query
from query_routing.observability import get_observability_snapshot, reset_observability_metrics


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
                "query_signals": {"asks_for_db_facts": True},
            },
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
                "query_signals": {"asks_for_db_facts": True, "is_comfort_assessment_phrase": True},
            },
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
    def test_critic_warns_on_missing_recommendation(self, mock_route_plan, mock_run_db):
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

        self.assertEqual(result["metadata"]["critic_status"], "warn")
        self.assertIn("missing_recommendation", result["metadata"]["critic_issues"])
        self.assertTrue(result["answer"].startswith("Note:"))

    @patch("query_routing.query_orchestrator.run_db_query")
    @patch("query_routing.query_orchestrator.get_route_plan")
    def test_critic_blocks_on_evidence_answer_mismatch(self, mock_route_plan, mock_run_db):
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

        self.assertEqual(result["metadata"]["critic_status"], "block")
        self.assertTrue(result["metadata"]["critic_blocked"])
        self.assertIn("evidence_answer_mismatch", result["metadata"]["critic_issues"])
        self.assertIn("re-check evidence alignment", result["answer"])

    @patch("query_routing.query_orchestrator.run_db_query")
    @patch("query_routing.query_orchestrator.get_route_plan")
    def test_critic_blocks_on_date_consistency_mismatch(self, mock_route_plan, mock_run_db):
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

        self.assertEqual(result["metadata"]["critic_status"], "block")
        self.assertTrue(result["metadata"]["critic_blocked"])
        self.assertIn("date_consistency_mismatch", result["metadata"]["critic_issues"])
        self.assertIn("Verified window: Mar 27, 2026, 2:15 PM GMT+4 to Mar 28, 2026, 2:15 PM GMT+4.", result["answer"])

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
    def test_non_domain_scope_rewrites_vague_insufficient_message(
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

        self.assertIn("I focus on Indoor Environmental Quality (IEQ) questions.", result["answer"])
        self.assertNotIn("I don't know from the available data.", result["answer"])
        self.assertTrue(bool(result["metadata"].get("scope_guardrail_applied")))
        self.assertEqual(result["metadata"]["executor"], "knowledge_qa")
        mock_run_db.assert_not_called()

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
        self.assertIn("critic_failure_rate", obs)
        self.assertIn("decomposition_template_usage", obs)
        self.assertIn("rollout_slo", result["metadata"])

        snapshot = get_observability_snapshot()
        self.assertGreaterEqual(snapshot.get("critic_total", 0), 1)


if __name__ == "__main__":
    unittest.main()
