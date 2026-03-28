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

from http_routes.query_routes import router as query_router
from query_routing.intent_classifier import IntentType, RouteDecision
from query_routing.llm_router_planner import (
    AnswerStrategy,
    DecompositionTemplate,
    IntentCategory,
    RoutePlan,
)


class QueryRoutePreviewTests(unittest.TestCase):
    def setUp(self):
        app = FastAPI()
        app.include_router(query_router)
        self.client = TestClient(app)

    @patch("http_routes.query_routes.get_route_plan")
    def test_preview_route_includes_planner_metadata(self, mock_get_route_plan):
        mock_get_route_plan.return_value = RoutePlan(
            decision=RouteDecision(
                intent=IntentType.FORECAST_DB,
                confidence=0.88,
                reason="forecast_phrase",
            ),
            intent_category=IntentCategory.PREDICTION,
            route_source="llm_planner",
            planner_model="qwen3:30b",
            planner_fallback_used=False,
            planner_fallback_reason=None,
            planner_raw={"intent": "forecast_db"},
            answer_strategy=AnswerStrategy.DIRECT,
            secondary_intents=(IntentType.AGGREGATION_DB,),
        )

        response = self.client.post(
            "/query/route",
            json={"question": "Predict PM2.5 for next day", "lab_name": "smart_lab"},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["route_source"], "llm_planner")
        self.assertEqual(payload["route_type"], "forecast_db")
        self.assertEqual(payload["intent_category"], "prediction")
        self.assertEqual(payload["planner_model"], "qwen3:30b")
        self.assertFalse(payload["planner_fallback_used"])
        self.assertEqual(payload["answer_strategy"], "direct")
        self.assertEqual(payload["secondary_intents"], ["aggregation_db"])
        self.assertIsNone(payload["decomposition_template"])

    @patch("http_routes.query_routes.get_route_plan")
    def test_preview_route_includes_template_id(self, mock_get_route_plan):
        mock_get_route_plan.return_value = RoutePlan(
            decision=RouteDecision(
                intent=IntentType.ANOMALY_ANALYSIS_DB,
                confidence=0.86,
                reason="anomaly_with_explanation",
            ),
            intent_category=IntentCategory.ANALYTICAL_VISUALIZATION,
            route_source="llm_planner",
            planner_model="qwen3:30b",
            planner_fallback_used=False,
            planner_fallback_reason=None,
            planner_raw={"intent": "anomaly_analysis_db"},
            answer_strategy=AnswerStrategy.DECOMPOSE,
            secondary_intents=(IntentType.DEFINITION_EXPLANATION,),
            decomposition_template=DecompositionTemplate.ANOMALY_EXPLANATION,
        )

        response = self.client.post(
            "/query/route",
            json={"question": "Why did PM2.5 spike this morning?", "lab_name": "smart_lab"},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["answer_strategy"], "decompose")
        self.assertEqual(payload["decomposition_template"], "anomaly_explanation")


if __name__ == "__main__":
    unittest.main()
