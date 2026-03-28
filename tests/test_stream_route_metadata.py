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
from http_routes.query_routes import router as query_router
from query_routing.intent_classifier import IntentType, RouteDecision
from query_routing.llm_router_planner import (
    AnswerStrategy,
    DecompositionTemplate,
    IntentCategory,
    RoutePlan,
)


class StreamRouteMetadataTests(unittest.TestCase):
    def setUp(self):
        app = FastAPI()
        app.include_router(query_router)
        app.include_router(openai_router)
        self.client = TestClient(app)

    @patch("http_routes.query_routes.stream_db_query")
    @patch("http_routes.query_routes.prepare_db_query")
    @patch("http_routes.query_routes.resolve_execution_intent")
    @patch("http_routes.query_routes.get_route_plan")
    def test_query_stream_meta_includes_cards_count(
        self,
        mock_get_route_plan,
        mock_resolve_execution_intent,
        mock_prepare_db_query,
        mock_stream_db_query,
    ):
        async def _fake_stream(*args, **kwargs):
            yield "ok"

        mock_get_route_plan.return_value = RoutePlan(
            decision=RouteDecision(intent=IntentType.AGGREGATION_DB, confidence=0.9, reason="agg"),
            intent_category=IntentCategory.STRUCTURED_FACTUAL_DB,
            route_source="llm_planner",
            planner_model="qwen3:30b",
            planner_fallback_used=False,
            planner_fallback_reason=None,
            planner_raw={},
            planner_parameters={"response_mode": "db"},
            answer_strategy=AnswerStrategy.DECOMPOSE,
            secondary_intents=(IntentType.DEFINITION_EXPLANATION,),
            decomposition_template=DecompositionTemplate.TREND_INTERPRETATION,
        )
        mock_resolve_execution_intent.return_value = IntentType.AGGREGATION_DB
        mock_prepare_db_query.return_value = {
            "timescale": "1hour",
            "resolved_lab_name": "smart_lab",
            "time_window": {"label": "this week", "start": "x", "end": "y"},
            "sources": [],
            "forecast": None,
            "visualization_type": "bar",
            "chart": {"type": "bar"},
            "cards_retrieved": 3,
        }
        mock_stream_db_query.side_effect = _fake_stream

        response = self.client.post(
            "/query/stream",
            json={"question": "Average CO2 this week", "lab_name": "smart_lab"},
        )
        self.assertEqual(response.status_code, 200)
        body = response.text
        self.assertIn('"cards_retrieved": 3', body)
        self.assertIn('"route_type": "aggregation_db"', body)
        self.assertIn('"execution_intent": "aggregation_db"', body)
        self.assertIn('"decomposition_template": "trend_interpretation"', body)

    @patch("http_routes.query_routes.get_route_plan")
    def test_query_stream_non_domain_uses_scope_guardrail_message(self, mock_get_route_plan):
        mock_get_route_plan.return_value = RoutePlan(
            decision=RouteDecision(intent=IntentType.DEFINITION_EXPLANATION, confidence=0.9, reason="non_domain"),
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
            answer_strategy=AnswerStrategy.DIRECT,
            secondary_intents=tuple(),
            decomposition_template=None,
        )

        response = self.client.post(
            "/query/stream",
            json={"question": "What day is today?", "lab_name": None},
        )
        self.assertEqual(response.status_code, 200)
        body = response.text
        self.assertIn('"executor": "scope_guardrail"', body)
        self.assertIn('"query_scope_class": "non_domain"', body)
        self.assertIn("I focus on Indoor Environmental Quality (IEQ) questions.", body)

    @patch("http_routes.openai_compat_routes.stream_db_query")
    @patch("http_routes.openai_compat_routes.prepare_db_query")
    @patch("http_routes.openai_compat_routes.resolve_execution_intent")
    @patch("http_routes.openai_compat_routes.get_route_plan")
    def test_openai_stream_includes_router_metadata(
        self,
        mock_get_route_plan,
        mock_resolve_execution_intent,
        mock_prepare_db_query,
        mock_stream_db_query,
    ):
        async def _fake_stream(*args, **kwargs):
            yield "hello"

        mock_get_route_plan.return_value = RoutePlan(
            decision=RouteDecision(intent=IntentType.COMPARISON_DB, confidence=0.93, reason="compare"),
            intent_category=IntentCategory.ANALYTICAL_VISUALIZATION,
            route_source="llm_planner",
            planner_model="qwen3:30b",
            planner_fallback_used=False,
            planner_fallback_reason=None,
            planner_raw={},
            planner_parameters={"response_mode": "db"},
            answer_strategy=AnswerStrategy.DECOMPOSE,
            secondary_intents=(IntentType.DEFINITION_EXPLANATION,),
            decomposition_template=DecompositionTemplate.TREND_INTERPRETATION,
        )
        mock_resolve_execution_intent.return_value = IntentType.COMPARISON_DB
        mock_prepare_db_query.return_value = {
            "sources": [{"source_kind": "db_query"}],
            "visualization_type": "bar",
            "chart": {"type": "bar"},
        }
        mock_stream_db_query.side_effect = _fake_stream

        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "rag-router",
                "stream": True,
                "messages": [{"role": "user", "content": "Compare smart_lab vs concrete_lab"}],
                "lab_name": "smart_lab",
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.text
        self.assertIn('"x_router"', body)
        self.assertIn('"route_type": "comparison_db"', body)
        self.assertIn('"execution_intent": "comparison_db"', body)
        self.assertIn('"x_visualization_type": "bar"', body)
        self.assertIn('"decomposition_template": "trend_interpretation"', body)

    @patch("http_routes.query_routes.execute_query")
    def test_query_non_stream_preserves_metadata_shape(self, mock_execute_query):
        mock_execute_query.return_value = {
            "answer": "ok",
            "timescale": "1hour",
            "cards_retrieved": 2,
            "recent_card": False,
            "metadata": {
                "executor": "db_query",
                "route_type": "aggregation_db",
                "critic_status": "warn",
                "critic_issues": ["missing_recommendation"],
                "evidence": {"evidence_kind": "db_query"},
            },
            "visualization_type": "bar",
            "chart": {"type": "bar"},
        }
        response = self.client.post("/query", json={"question": "Average CO2 this week"})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["metadata"]["executor"], "db_query")
        self.assertEqual(payload["metadata"]["critic_status"], "warn")
        self.assertEqual(payload["metadata"]["evidence"]["evidence_kind"], "db_query")
        self.assertEqual(payload["cards_retrieved"], 2)
        self.assertEqual(payload["visualization_type"], "bar")

    @patch("http_routes.openai_compat_routes.execute_query")
    def test_openai_non_stream_includes_router_metadata(self, mock_execute_query):
        mock_execute_query.return_value = {
            "answer": "ok",
            "timescale": "1hour",
            "metadata": {
                "route_type": "aggregation_db",
                "executor": "db_query",
                "critic_status": "pass",
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
                "messages": [{"role": "user", "content": "Average CO2 this week"}],
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["x_router"]["route_type"], "aggregation_db")
        self.assertEqual(payload["x_router"]["critic_status"], "pass")
        self.assertEqual(payload["x_router"]["evidence"]["evidence_kind"], "db_query")
        self.assertEqual(payload["x_visualization_type"], "bar")


if __name__ == "__main__":
    unittest.main()
