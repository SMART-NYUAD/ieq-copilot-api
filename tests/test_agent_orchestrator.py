import os
import sys
import unittest
from dataclasses import dataclass


TEST_DIR = os.path.dirname(__file__)
SERVER_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

from query_routing.agent_orchestrator import run_agentic_query_loop
from query_routing.intent_classifier import IntentType, RouteDecision
from query_routing.router_types import AgentAction, IntentCategory, RouteExecutor, RoutePlan


@dataclass(frozen=True)
class _Contract:
    route_plan: RoutePlan
    executor: RouteExecutor
    execution_intent: IntentType
    latest_question_hash: str = "abc"
    policy_version: str = "v1"
    needs_measured_data: bool = True
    rule_trace: tuple[str, ...] = tuple()


class AgentOrchestratorTests(unittest.TestCase):
    def test_agent_loop_finalizes_and_attaches_metadata(self):
        plans = [
            _Contract(
                route_plan=RoutePlan(
                    decision=RouteDecision(IntentType.AGGREGATION_DB, 0.82, "step1"),
                    intent_category=IntentCategory.STRUCTURED_FACTUAL_DB,
                    route_source="llm_planner",
                    planner_model="model",
                    planner_fallback_used=False,
                    agent_action=AgentAction.TOOL_CALL,
                    tool_name="query_db",
                ),
                executor=RouteExecutor.DB_QUERY,
                execution_intent=IntentType.AGGREGATION_DB,
            ),
            _Contract(
                route_plan=RoutePlan(
                    decision=RouteDecision(IntentType.AGGREGATION_DB, 0.86, "step2"),
                    intent_category=IntentCategory.STRUCTURED_FACTUAL_DB,
                    route_source="llm_planner",
                    planner_model="model",
                    planner_fallback_used=False,
                    agent_action=AgentAction.FINALIZE,
                ),
                executor=RouteExecutor.DB_QUERY,
                execution_intent=IntentType.AGGREGATION_DB,
            ),
        ]
        state = {"idx": 0}

        def _next_contract(*args, **kwargs):
            idx = min(state["idx"], len(plans) - 1)
            state["idx"] += 1
            return plans[idx]

        def _execute_with_contract(**kwargs):
            _ = kwargs
            return {
                "answer": "db answer",
                "timescale": "1hour",
                "cards_retrieved": 0,
                "recent_card": False,
                "metadata": {"executor": "db_query"},
            }

        def _clarify_result(**kwargs):
            _ = kwargs
            return {"answer": "clarify", "timescale": "clarify", "cards_retrieved": 0, "recent_card": False, "metadata": {}}

        result = run_agentic_query_loop(
            question="Average CO2 this week?",
            generation_question="Average CO2 this week?",
            k=5,
            lab_name="smart_lab",
            allow_clarify=True,
            max_steps=3,
            max_consecutive_failures=2,
            stall_threshold=2,
            get_route_decision_contract_fn=_next_contract,
            execute_with_contract_fn=_execute_with_contract,
            build_clarify_result_fn=_clarify_result,
            build_clarify_prompt_fn=lambda *_: "clarify",
        )
        meta = result["metadata"]
        self.assertEqual(meta["agent_mode"], "enabled")
        self.assertEqual(meta["agent_finish_reason"], "finalize")
        self.assertEqual(meta["agent_steps"], 2)
        self.assertEqual(meta["tools_called"], ["query_db"])

    def test_agent_loop_enforces_tool_budget_for_comparison(self):
        contract = _Contract(
            route_plan=RoutePlan(
                decision=RouteDecision(IntentType.COMPARISON_DB, 0.95, "compare"),
                intent_category=IntentCategory.ANALYTICAL_VISUALIZATION,
                route_source="llm_planner",
                planner_model="model",
                planner_fallback_used=False,
                agent_action=AgentAction.TOOL_CALL,
                tool_name="compare_spaces",
            ),
            executor=RouteExecutor.DB_QUERY,
            execution_intent=IntentType.COMPARISON_DB,
        )

        def _contract_fn(*args, **kwargs):
            _ = (args, kwargs)
            return contract

        with (
            unittest.mock.patch(
                "query_routing.agent_orchestrator.execute_agent_tool_call",
                return_value={
                    "ok": True,
                    "tool_name": "compare_spaces",
                    "intent": IntentType.COMPARISON_DB.value,
                    "result": {"answer": "comparison only"},
                },
            ),
            unittest.mock.patch(
                "query_routing.agent_orchestrator.summarize_tool_observation",
                return_value="comparison only",
            ),
        ):
            result = run_agentic_query_loop(
                question="Compare smart_lab and concrete_lab, then explain why and what actions to take.",
                generation_question="Compare smart_lab and concrete_lab, then explain why and what actions to take.",
                k=5,
                lab_name=None,
                allow_clarify=True,
                max_steps=4,
                max_consecutive_failures=2,
                stall_threshold=2,
                get_route_decision_contract_fn=_contract_fn,
                execute_with_contract_fn=lambda **kwargs: {
                    "answer": "finalized",
                    "timescale": "1hour",
                    "cards_retrieved": 0,
                    "recent_card": False,
                    "metadata": {"executor": "db_query"},
                },
                build_clarify_result_fn=lambda **kwargs: {
                    "answer": "clarify",
                    "timescale": "clarify",
                    "cards_retrieved": 0,
                    "recent_card": False,
                    "metadata": {},
                },
                build_clarify_prompt_fn=lambda *_: "clarify",
            )
        self.assertEqual(result["metadata"]["agent_finish_reason"], "tool_budget_exhausted")
        self.assertEqual(result["metadata"]["tools_called"], ["compare_spaces"])


if __name__ == "__main__":
    unittest.main()
