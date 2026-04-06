import os
import sys
import unittest
from unittest.mock import patch


TEST_DIR = os.path.dirname(__file__)
SERVER_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

from http_routes.query_runtime import resolve_agent_stream_runtime
from query_routing.intent_classifier import IntentType, RouteDecision
from query_routing.router_types import (
    AgentAction,
    IntentCategory,
    RouteDecisionContract,
    RouteExecutor,
    RoutePlan,
)


def _build_contract(*, enough_evidence=None, goal_coverage=()):
    route_plan = RoutePlan(
        decision=RouteDecision(intent=IntentType.COMPARISON_DB, confidence=0.95, reason="compare"),
        intent_category=IntentCategory.ANALYTICAL_VISUALIZATION,
        route_source="llm_planner",
        planner_model="model",
        planner_fallback_used=False,
        planner_fallback_reason=None,
        planner_raw={},
        planner_parameters={"response_mode": "db"},
        agent_action=AgentAction.TOOL_CALL,
        tool_name="compare_spaces",
        enough_evidence=enough_evidence,
        goal_coverage=tuple(goal_coverage),
    )
    return RouteDecisionContract(
        latest_user_question="q",
        latest_question_hash="h",
        policy_version="v1",
        route_plan=route_plan,
        needs_measured_data=True,
        executor=RouteExecutor.DB_QUERY,
        execution_intent=IntentType.COMPARISON_DB,
        execution_intent_value=IntentType.COMPARISON_DB.value,
        query_scope_class="domain",
        rule_trace=tuple(),
    )


class QueryRuntimeAgentLoopTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._prior_env = {
            "AGENTIC_MODE": os.environ.get("AGENTIC_MODE"),
            "AGENT_MAX_STEPS": os.environ.get("AGENT_MAX_STEPS"),
            "AGENT_MAX_CONSECUTIVE_FAILURES": os.environ.get("AGENT_MAX_CONSECUTIVE_FAILURES"),
            "AGENT_STALL_THRESHOLD": os.environ.get("AGENT_STALL_THRESHOLD"),
        }
        os.environ["AGENTIC_MODE"] = "true"
        os.environ["AGENT_MAX_STEPS"] = "4"
        os.environ["AGENT_MAX_CONSECUTIVE_FAILURES"] = "2"
        os.environ["AGENT_STALL_THRESHOLD"] = "2"

    def tearDown(self):
        for key, value in self._prior_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    async def test_tool_budget_exhausted_for_repeated_comparison_calls(self):
        contract = _build_contract()

        with (
            patch(
                "http_routes.query_runtime.execute_agent_tool_call",
                return_value={
                    "ok": True,
                    "tool_name": "compare_spaces",
                    "intent": IntentType.COMPARISON_DB.value,
                    "result": {"answer": "Comparison summary only."},
                },
            ),
            patch(
                "http_routes.query_runtime.summarize_tool_observation",
                return_value="comparison summary only",
            ),
        ):
            result = await resolve_agent_stream_runtime(
                latest_user_question=(
                    "Compare smart_lab and concrete_lab for CO2 for last 7 days, "
                    "then explain why and what actions to take."
                ),
                k=5,
                lab_name=None,
                allow_clarify=True,
                endpoint_key="query_stream",
                get_route_decision_contract_fn=lambda *args, **kwargs: contract,
            )

        self.assertEqual(result.finish_reason, "tool_budget_exhausted")
        self.assertEqual(result.tools_called, ["compare_spaces"])
        self.assertEqual(len(result.agent_step_trace), 2)
        self.assertEqual(result.agent_step_trace[1].get("blocked_reason"), "tool_budget_exhausted")

    async def test_goal_coverage_complete_stops_loop_without_evidence_flag(self):
        contract = _build_contract(enough_evidence=None, goal_coverage=())
        with (
            patch(
                "http_routes.query_runtime.execute_agent_tool_call",
                return_value={
                    "ok": True,
                    "tool_name": "compare_spaces",
                    "intent": IntentType.COMPARISON_DB.value,
                    "result": {"answer": "Concrete is better because ventilation is stronger. I recommend increasing airflow in smart_lab."},
                },
            ),
            patch(
                "http_routes.query_runtime.summarize_tool_observation",
                return_value="Concrete is better because ventilation is stronger. I recommend increasing airflow.",
            ),
        ):
            result = await resolve_agent_stream_runtime(
                latest_user_question=(
                    "Compare smart_lab and concrete_lab for CO2 for last 7 days, "
                    "then explain why and what actions to take."
                ),
                k=5,
                lab_name=None,
                allow_clarify=True,
                endpoint_key="query_stream",
                get_route_decision_contract_fn=lambda *args, **kwargs: contract,
            )
        self.assertEqual(result.finish_reason, "goal_coverage_complete")
        self.assertEqual(result.tools_called, ["compare_spaces"])
        self.assertEqual(len(result.agent_step_trace), 1)

    async def test_enough_evidence_signal_stops_loop_when_goals_met(self):
        contract = _build_contract(enough_evidence=True, goal_coverage=("compare", "explain", "recommend"))
        with (
            patch(
                "http_routes.query_runtime.execute_agent_tool_call",
                return_value={
                    "ok": True,
                    "tool_name": "compare_spaces",
                    "intent": IntentType.COMPARISON_DB.value,
                    "result": {"answer": "Concrete is better because ventilation is stronger. You should increase fresh-air exchange in smart_lab."},
                },
            ),
            patch(
                "http_routes.query_runtime.summarize_tool_observation",
                return_value="Concrete is better because ventilation is stronger. You should increase fresh-air exchange.",
            ),
        ):
            result = await resolve_agent_stream_runtime(
                latest_user_question=(
                    "Compare smart_lab and concrete_lab for CO2 for last 7 days, "
                    "then explain why and what actions to take."
                ),
                k=5,
                lab_name=None,
                allow_clarify=True,
                endpoint_key="query_stream",
                get_route_decision_contract_fn=lambda *args, **kwargs: contract,
            )
        self.assertEqual(result.finish_reason, "evidence_sufficient")
        self.assertEqual(result.tools_called, ["compare_spaces"])
        self.assertEqual(len(result.agent_step_trace), 1)


if __name__ == "__main__":
    unittest.main()
