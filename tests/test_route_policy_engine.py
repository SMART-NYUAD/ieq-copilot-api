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

from http_routes.query_runtime import resolve_stream_runtime
from query_routing.intent_classifier import IntentType, RouteDecision
from query_routing.observability import reset_observability_metrics
from query_routing.query_orchestrator import get_route_decision_contract
from query_routing.route_policy_engine import build_route_decision_contract
from query_routing.router_types import AnswerStrategy, IntentCategory, RouteExecutor, RoutePlan


def _make_route_plan(
    *,
    intent: IntentType,
    confidence: float,
    response_mode: str,
    query_signals: dict,
    answer_strategy: AnswerStrategy = AnswerStrategy.DIRECT,
    needs_measured_data: bool | None = None,
    has_explicit_scope: bool | None = None,
    resolved_lab_name: str | None = None,
    clarify_reason: str | None = None,
) -> RoutePlan:
    declared_needs = needs_measured_data
    if declared_needs is None:
        declared_needs = bool(query_signals.get("asks_for_db_facts")) or bool(
            query_signals.get("requests_current_measured_data")
        )
    return RoutePlan(
        decision=RouteDecision(intent=intent, confidence=confidence, reason="test"),
        intent_category=IntentCategory.SEMANTIC_EXPLANATORY
        if intent in {IntentType.DEFINITION_EXPLANATION, IntentType.UNKNOWN_FALLBACK}
        else IntentCategory.STRUCTURED_FACTUAL_DB,
        route_source="llm_planner",
        planner_model="test-model",
        planner_fallback_used=False,
        planner_fallback_reason=None,
        planner_raw={},
        planner_parameters={
            "response_mode": response_mode,
            "needs_measured_data": declared_needs,
            "has_explicit_scope": has_explicit_scope,
            "resolved_lab_name": resolved_lab_name,
            "clarify_reason": clarify_reason,
            "query_signals": query_signals,
        },
        answer_strategy=answer_strategy,
        has_explicit_scope=bool(has_explicit_scope),
        resolved_lab_name=resolved_lab_name,
        clarify_reason=clarify_reason,
    )


class RoutePolicyEngineTests(unittest.TestCase):
    def setUp(self):
        reset_observability_metrics()

    def test_conceptual_prompt_forces_knowledge_executor(self):
        plan = _make_route_plan(
            intent=IntentType.DEFINITION_EXPLANATION,
            confidence=0.92,
            response_mode="db",
            query_signals={
                "query_scope_class": "ambiguous",
                "is_general_knowledge_question": True,
                "asks_for_db_facts": False,
                "has_metric_reference": True,
                "has_time_window_hint": False,
                "has_lab_reference": False,
            },
        )
        contract = build_route_decision_contract(
            latest_user_question="How should I interpret PM2.5, TVOC, and humidity together?",
            route_plan=plan,
            allow_clarify=True,
        )
        self.assertEqual(contract.executor, RouteExecutor.KNOWLEDGE_QA)
        self.assertFalse(contract.needs_measured_data)

    def test_scoped_prompt_forces_db_executor(self):
        plan = _make_route_plan(
            intent=IntentType.DEFINITION_EXPLANATION,
            confidence=0.9,
            response_mode="knowledge_only",
            query_signals={
                "query_scope_class": "domain",
                "is_general_knowledge_question": False,
                "asks_for_db_facts": True,
                "has_metric_reference": True,
                "has_time_window_hint": True,
                "has_lab_reference": True,
            },
            has_explicit_scope=True,
            resolved_lab_name="smart_lab",
        )
        contract = build_route_decision_contract(
            latest_user_question="What was average CO2 in smart_lab last week?",
            route_plan=plan,
            allow_clarify=True,
        )
        self.assertEqual(contract.executor, RouteExecutor.DB_QUERY)
        self.assertTrue(contract.needs_measured_data)

    def test_hypothetical_without_live_scope_forces_knowledge(self):
        plan = _make_route_plan(
            intent=IntentType.AGGREGATION_DB,
            confidence=0.91,
            response_mode="knowledge_only",
            query_signals={
                "query_scope_class": "ambiguous",
                "is_hypothetical_conditional": True,
                "requests_current_measured_data": False,
                "asks_for_db_facts": True,
                "has_metric_reference": True,
            },
            needs_measured_data=False,
        )
        contract = build_route_decision_contract(
            latest_user_question="If humidity is persistently above 70%, what risk should be flagged?",
            route_plan=plan,
            allow_clarify=True,
        )
        self.assertEqual(contract.executor, RouteExecutor.KNOWLEDGE_QA)
        self.assertFalse(contract.needs_measured_data)
        self.assertIn("planner_knowledge_mode", contract.rule_trace)

    def test_single_lab_baseline_forces_db_without_second_space(self):
        plan = _make_route_plan(
            intent=IntentType.COMPARISON_DB,
            confidence=0.9,
            response_mode="db",
            query_signals={
                "query_scope_class": "domain",
                "is_baseline_reference_query": True,
                "single_explicit_lab_with_baseline_reference": True,
                "has_explicit_second_space": False,
                "asks_for_db_facts": True,
            },
            has_explicit_scope=True,
            resolved_lab_name="concrete_lab",
        )
        contract = build_route_decision_contract(
            latest_user_question="Compare humidity in concrete_lab against its baseline for this morning",
            route_plan=plan,
            allow_clarify=True,
        )
        self.assertEqual(contract.executor, RouteExecutor.DB_QUERY)
        self.assertTrue(contract.needs_measured_data)
        self.assertIn("measured_scope_forces_db", contract.rule_trace)

    @patch("query_routing.route_policy_engine.load_settings")
    def test_ambiguous_low_confidence_routes_to_clarify(self, mock_load_settings):
        mock_load_settings.return_value.router_clarify_threshold = 0.7
        plan = _make_route_plan(
            intent=IntentType.AGGREGATION_DB,
            confidence=0.45,
            response_mode="db",
            query_signals={
                "query_scope_class": "ambiguous",
                "asks_for_db_facts": False,
            },
            answer_strategy=AnswerStrategy.DIRECT,
        )
        contract = build_route_decision_contract(
            latest_user_question="How is it over there?",
            route_plan=plan,
            allow_clarify=True,
        )
        self.assertEqual(contract.executor, RouteExecutor.CLARIFY_GATE)

    def test_explicit_measured_scope_ignores_clarify_strategy(self):
        plan = _make_route_plan(
            intent=IntentType.DEFINITION_EXPLANATION,
            confidence=0.95,
            response_mode="db",
            query_signals={
                "query_scope_class": "domain",
                "asks_for_db_facts": True,
                "has_metric_reference": True,
                "has_lab_reference": True,
                "has_time_window_hint": False,
                "has_db_scope_phrase": False,
            },
            answer_strategy=AnswerStrategy.CLARIFY,
            has_explicit_scope=True,
            resolved_lab_name="smart_lab",
        )
        contract = build_route_decision_contract(
            latest_user_question="How is the light in the smart lab?",
            route_plan=plan,
            allow_clarify=True,
        )
        self.assertEqual(contract.executor, RouteExecutor.DB_QUERY)
        self.assertTrue(contract.needs_measured_data)
        self.assertIn("measured_scope_forces_db", contract.rule_trace)

    def test_non_domain_scope_routes_to_knowledge(self):
        plan = _make_route_plan(
            intent=IntentType.CURRENT_STATUS_DB,
            confidence=0.95,
            response_mode="db",
            query_signals={
                "query_scope_class": "non_domain",
                "asks_for_db_facts": False,
            },
        )
        contract = build_route_decision_contract(
            latest_user_question="What day is today?",
            route_plan=plan,
            allow_clarify=True,
        )
        self.assertEqual(contract.executor, RouteExecutor.KNOWLEDGE_QA)
        self.assertFalse(contract.needs_measured_data)

    def test_stream_runtime_uses_same_policy_contract(self):
        plan = _make_route_plan(
            intent=IntentType.DEFINITION_EXPLANATION,
            confidence=0.92,
            response_mode="knowledge_only",
            query_signals={
                "query_scope_class": "ambiguous",
                "is_general_knowledge_question": True,
                "asks_for_db_facts": False,
            },
        )
        expected = build_route_decision_contract(
            latest_user_question="What is IEQ?",
            route_plan=plan,
            allow_clarify=True,
        )

        async def _resolve():
            return await resolve_stream_runtime(
                latest_user_question="What is IEQ?",
                lab_name=None,
                allow_clarify=True,
                get_route_plan_fn=lambda _q, _l: plan,
            )

        import asyncio

        runtime = asyncio.run(_resolve())
        self.assertEqual(runtime.route_contract.executor, expected.executor)
        self.assertEqual(runtime.route_contract.query_scope_class, expected.query_scope_class)

    @patch("query_routing.query_orchestrator.get_route_plan")
    def test_orchestrator_always_returns_policy_contract(self, mock_get_route_plan):
        plan = _make_route_plan(
            intent=IntentType.DEFINITION_EXPLANATION,
            confidence=0.9,
            response_mode="knowledge_only",
            query_signals={"query_scope_class": "ambiguous", "asks_for_db_facts": False},
        )
        mock_get_route_plan.return_value = plan
        contract = get_route_decision_contract("What is IEQ?", None, True)
        self.assertEqual(contract.policy_version, "route-policy-v1")

    def test_semantic_intent_never_routes_to_clarify(self):
        plan = _make_route_plan(
            intent=IntentType.DEFINITION_EXPLANATION,
            confidence=0.2,
            response_mode="db",
            query_signals={
                "query_scope_class": "ambiguous",
                "is_general_knowledge_question": True,
                "asks_for_db_facts": False,
            },
            answer_strategy=AnswerStrategy.CLARIFY,
        )
        contract = build_route_decision_contract(
            latest_user_question="What does IEQ mean?",
            route_plan=plan,
            allow_clarify=True,
        )
        self.assertEqual(contract.executor, RouteExecutor.KNOWLEDGE_QA)
        self.assertNotEqual(contract.executor, RouteExecutor.CLARIFY_GATE)

    def test_general_knowledge_metric_comparison_without_scope_stays_knowledge(self):
        plan = _make_route_plan(
            intent=IntentType.DEFINITION_EXPLANATION,
            confidence=0.85,
            response_mode="knowledge_only",
            query_signals={
                "query_scope_class": "ambiguous",
                "is_general_knowledge_question": True,
                "asks_for_db_facts": False,
                "has_metric_reference": True,
                "has_db_scope_phrase": True,
                "has_lab_reference": False,
                "has_time_window_hint": False,
            },
        )
        contract = build_route_decision_contract(
            latest_user_question="how does PM2.5 compare with CO2 or humidity",
            route_plan=plan,
            allow_clarify=True,
        )
        self.assertEqual(contract.executor, RouteExecutor.KNOWLEDGE_QA)
        self.assertFalse(contract.needs_measured_data)
        self.assertIn("conceptual_semantic_forces_knowledge", contract.rule_trace)

    def test_high_confidence_definition_without_scope_signals_stays_knowledge(self):
        plan = _make_route_plan(
            intent=IntentType.DEFINITION_EXPLANATION,
            confidence=0.9,
            response_mode="db",
            query_signals={
                "query_scope_class": "domain",
                "is_general_knowledge_question": False,
                "asks_for_db_facts": False,
                "has_time_window_hint": False,
                "has_db_scope_phrase": False,
                "is_air_assessment_phrase": False,
            },
        )
        contract = build_route_decision_contract(
            latest_user_question="What does CO2 mean?",
            route_plan=plan,
            allow_clarify=True,
        )
        self.assertEqual(contract.executor, RouteExecutor.KNOWLEDGE_QA)
        self.assertFalse(contract.needs_measured_data)

    def test_planner_clarify_reason_forces_clarify_gate(self):
        plan = _make_route_plan(
            intent=IntentType.AGGREGATION_DB,
            confidence=0.94,
            response_mode="db",
            query_signals={"query_scope_class": "domain", "asks_for_db_facts": True},
            needs_measured_data=True,
            has_explicit_scope=True,
            resolved_lab_name="smart_lab",
            clarify_reason="ambiguous_intent",
        )
        contract = build_route_decision_contract(
            latest_user_question="Compare trends and explain comfort",
            route_plan=plan,
            allow_clarify=True,
        )
        self.assertEqual(contract.executor, RouteExecutor.CLARIFY_GATE)
        self.assertIn("planner_clarify_reason:ambiguous_intent", contract.rule_trace)

    def test_measured_data_without_scope_no_longer_forces_clarify(self):
        plan = _make_route_plan(
            intent=IntentType.CURRENT_STATUS_DB,
            confidence=0.91,
            response_mode="db",
            query_signals={"query_scope_class": "ambiguous", "asks_for_db_facts": True},
            needs_measured_data=True,
            has_explicit_scope=False,
            resolved_lab_name=None,
        )
        contract = build_route_decision_contract(
            latest_user_question="Is the air good right now?",
            route_plan=plan,
            allow_clarify=True,
        )
        self.assertEqual(contract.executor, RouteExecutor.DB_QUERY)
        self.assertIn("measured_scope_forces_db", contract.rule_trace)

    def test_no_lab_clarify_reason_ignored_for_conceptual_query(self):
        plan = _make_route_plan(
            intent=IntentType.DEFINITION_EXPLANATION,
            confidence=0.9,
            response_mode="knowledge_only",
            query_signals={
                "query_scope_class": "ambiguous",
                "is_general_knowledge_question": True,
                "asks_for_db_facts": False,
            },
            needs_measured_data=False,
            has_explicit_scope=False,
            resolved_lab_name=None,
            clarify_reason="no_lab",
        )
        contract = build_route_decision_contract(
            latest_user_question="How should I interpret PM2.5, TVOC, and humidity together?",
            route_plan=plan,
            allow_clarify=True,
        )
        self.assertEqual(contract.executor, RouteExecutor.KNOWLEDGE_QA)
        self.assertFalse(contract.needs_measured_data)
        self.assertIn("conceptual_semantic_forces_knowledge", contract.rule_trace)

    def test_semantic_no_lab_clarify_ignored_even_if_declared_measured_data_true(self):
        plan = _make_route_plan(
            intent=IntentType.DEFINITION_EXPLANATION,
            confidence=0.9,
            response_mode="db",
            query_signals={
                "query_scope_class": "ambiguous",
                "is_general_knowledge_question": False,
                "asks_for_db_facts": False,
                "requests_current_measured_data": False,
                "has_metric_reference": True,
                "has_lab_reference": False,
                "has_time_window_hint": False,
            },
            needs_measured_data=True,
            has_explicit_scope=False,
            resolved_lab_name=None,
            clarify_reason="no_lab",
        )
        contract = build_route_decision_contract(
            latest_user_question="Which is a more reliable indicator of poor ventilation, CO2 or TVOC?",
            route_plan=plan,
            allow_clarify=True,
        )
        self.assertEqual(contract.executor, RouteExecutor.KNOWLEDGE_QA)
        self.assertFalse(contract.needs_measured_data)
        self.assertIn("conceptual_semantic_forces_knowledge", contract.rule_trace)

    def test_no_lab_clarify_reason_ignored_when_scope_is_explicit(self):
        plan = _make_route_plan(
            intent=IntentType.CURRENT_STATUS_DB,
            confidence=1.0,
            response_mode="db",
            query_signals={
                "query_scope_class": "domain",
                "asks_for_db_facts": True,
                "has_lab_reference": True,
            },
            needs_measured_data=True,
            has_explicit_scope=True,
            resolved_lab_name="shores_office",
            clarify_reason="no_lab",
        )
        contract = build_route_decision_contract(
            latest_user_question="Give me a full picture of air quality in shores_office right now",
            route_plan=plan,
            allow_clarify=True,
        )
        self.assertEqual(contract.executor, RouteExecutor.DB_QUERY)
        self.assertTrue(contract.needs_measured_data)
        self.assertIn("measured_scope_forces_db", contract.rule_trace)


if __name__ == "__main__":
    unittest.main()
