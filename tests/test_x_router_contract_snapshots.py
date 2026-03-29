import json
import os
import sys
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient


TEST_DIR = os.path.dirname(__file__)
SERVER_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)
REPO_DIR = os.path.abspath(os.path.join(SERVER_DIR, ".."))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from http_routes.openai_compat_routes import router as openai_router
from query_routing.intent_classifier import IntentType, RouteDecision
from query_routing.llm_router_planner import AnswerStrategy, IntentCategory, RoutePlan


def _extract_first_stream_chunk(response_text: str) -> dict:
    for block in response_text.split("\n\n"):
        line = block.strip()
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            continue
        parsed = json.loads(payload)
        if isinstance(parsed, dict) and parsed.get("object") == "chat.completion.chunk":
            return parsed
    raise AssertionError("No chat.completion.chunk found")


class XRouterContractSnapshotTests(unittest.TestCase):
    def setUp(self):
        app = FastAPI()
        app.include_router(openai_router)
        self.client = TestClient(app)

    @patch("http_routes.openai_compat_routes.execute_query")
    def test_openai_non_stream_x_router_snapshot(self, mock_execute_query):
        mock_execute_query.return_value = {
            "answer": "CO2 average was 640 ppm.",
            "timescale": "24hour",
            "cards_retrieved": 2,
            "recent_card": False,
            "metadata": {
                "route_source": "llm_planner",
                "route_type": "aggregation_db",
                "intent_category": "structured_factual_db",
                "route_confidence": 0.9,
                "route_reason": "aggregation",
                "planner_model": "qwen3:30b",
                "planner_fallback_used": False,
                "planner_fallback_reason": None,
                "executor": "db_query",
                "execution_intent": "aggregation_db",
                "intent_rerouted_to_db": False,
                "query_scope_class": "domain",
                "latest_question_hash": "abc123hash000001",
                "policy_version": "route-policy-v1",
                "rule_trace": ["measured_scope_forces_db"],
                "needs_measured_data": True,
                "evidence": {"evidence_kind": "db_query"},
            },
            "visualization_type": "bar",
            "chart": {"type": "bar"},
        }
        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "rag-router",
                "stream": False,
                "conversation_id": "conv12345",
                "messages": [{"role": "user", "content": "Average CO2 in smart_lab this week"}],
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        x_router = payload["x_router"]
        required_keys = {
            "route_source",
            "route_type",
            "intent_category",
            "route_confidence",
            "route_reason",
            "planner_model",
            "planner_fallback_used",
            "planner_fallback_reason",
            "executor",
            "execution_intent",
            "intent_rerouted_to_db",
            "query_scope_class",
            "latest_question_hash",
            "policy_version",
            "rule_trace",
            "needs_measured_data",
            "evidence",
            "conversation_id",
            "conversation_context_applied",
            "turn_index",
        }
        self.assertTrue(required_keys.issubset(set(x_router.keys())))
        self.assertEqual(x_router["executor"], "db_query")
        self.assertEqual(x_router["policy_version"], "route-policy-v1")
        self.assertEqual(x_router["conversation_id"], "conv12345")

    @patch("http_routes.openai_compat_routes.stream_answer_env_question")
    @patch("http_routes.openai_compat_routes.fetch_knowledge_stats")
    @patch("http_routes.openai_compat_routes.resolve_execution_intent")
    @patch("http_routes.openai_compat_routes.get_route_plan")
    def test_openai_stream_knowledge_x_router_snapshot(
        self,
        mock_get_route_plan,
        mock_resolve_execution_intent,
        mock_fetch_knowledge_stats,
        mock_stream_answer_env_question,
    ):
        async def _fake_stream(*args, **kwargs):
            yield "hello"

        mock_get_route_plan.return_value = RoutePlan(
            decision=RouteDecision(intent=IntentType.DEFINITION_EXPLANATION, confidence=0.92, reason="definition"),
            intent_category=IntentCategory.SEMANTIC_EXPLANATORY,
            route_source="llm_planner",
            planner_model="qwen3:30b",
            planner_fallback_used=False,
            planner_fallback_reason=None,
            planner_raw={},
            planner_parameters={"response_mode": "knowledge_only"},
            answer_strategy=AnswerStrategy.DIRECT,
        )
        mock_resolve_execution_intent.return_value = IntentType.DEFINITION_EXPLANATION
        mock_fetch_knowledge_stats.return_value = {"cards_retrieved": 2, "knowledge_cards_retrieved": 2}
        mock_stream_answer_env_question.side_effect = _fake_stream
        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "rag-router",
                "stream": True,
                "messages": [{"role": "user", "content": "What does IEQ mean?"}],
            },
        )
        self.assertEqual(response.status_code, 200)
        first_chunk = _extract_first_stream_chunk(response.text)
        x_router = first_chunk["x_router"]
        self.assertEqual(x_router["executor"], "knowledge_qa")
        self.assertEqual(x_router["route_type"], "definition_explanation")
        self.assertIn("latest_question_hash", x_router)
        self.assertEqual(x_router["policy_version"], "route-policy-v1")
        self.assertIn("rule_trace", x_router)

    @patch("http_routes.openai_compat_routes.fetch_db_context")
    @patch("http_routes.openai_compat_routes.resolve_execution_intent")
    @patch("http_routes.openai_compat_routes.get_route_plan")
    def test_openai_stream_db_invariant_violation_snapshot(
        self,
        mock_get_route_plan,
        mock_resolve_execution_intent,
        mock_fetch_db_context,
    ):
        mock_get_route_plan.return_value = RoutePlan(
            decision=RouteDecision(intent=IntentType.AGGREGATION_DB, confidence=0.92, reason="aggregation"),
            intent_category=IntentCategory.STRUCTURED_FACTUAL_DB,
            route_source="llm_planner",
            planner_model="qwen3:30b",
            planner_fallback_used=False,
            planner_fallback_reason=None,
            planner_raw={},
            planner_parameters={"response_mode": "db"},
            answer_strategy=AnswerStrategy.DIRECT,
        )
        mock_resolve_execution_intent.return_value = IntentType.AGGREGATION_DB
        mock_fetch_db_context.return_value = {
            "sources": [],
            "visualization_type": "none",
            "chart": None,
            "invariant_violation": {"allowed": False, "violations": ["metric_not_justified"]},
            "fallback_answer": "Please include metric/time/lab.",
        }
        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "rag-router",
                "stream": True,
                "messages": [{"role": "user", "content": "How is it there?"}],
            },
        )
        self.assertEqual(response.status_code, 200)
        first_chunk = _extract_first_stream_chunk(response.text)
        x_router = first_chunk["x_router"]
        self.assertEqual(x_router["executor"], "clarify_gate")
        self.assertTrue(x_router["clarification_required"])
        self.assertIn("db_invariant_violation", x_router)
        self.assertEqual(x_router.get("db_clarify_text"), "Please include metric/time/lab.")


if __name__ == "__main__":
    unittest.main()
