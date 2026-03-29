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

ROOT_DIR = os.path.abspath(os.path.join(REPO_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from prompting.shared_prompts import SHARED_SYSTEM_PROMPT, build_grounded_context_sections
from executors.env_query_langchain import build_card_grounded_context, search_knowledge_cards
from executors.db_query_executor import _build_db_payload
from executors.db_support.response_helpers import db_response_directive, is_air_quality_query_text
from knowledge_cards.loader import KnowledgeCardValidationError, normalize_card


class _FakeCursor:
    def __init__(self, rows):
        self.rows = rows
        self.last_sql = None
        self.last_params = None

    def execute(self, sql, params=None):
        self.last_sql = sql
        self.last_params = params

    def fetchall(self):
        return self.rows

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class KnowledgeCardTests(unittest.TestCase):
    def test_normalize_card_maps_tvoc_ugm3_to_tvoc(self):
        card = normalize_card(
            {
                "card_type": "interpretation",
                "topic": "tvoc",
                "title": "TVOC acceptable",
                "summary": "A summary",
                "content": "Detailed content",
                "audience": "general",
                "severity_level": "fair",
                "metric_name": "tvoc_ugm3",
                "condition_json": {},
                "recommendation_json": {},
                "tags": ["tvoc", "voc"],
                "source_label": "RESET",
                "source_url_key": "RESET_AIR",
            }
        )
        self.assertEqual(card["metric_name"], "tvoc")
        self.assertEqual(card["source_metadata"]["original_metric_name"], "tvoc_ugm3")

    def test_normalize_card_rejects_missing_fields(self):
        with self.assertRaises(KnowledgeCardValidationError):
            normalize_card({"card_type": "interpretation"})

    def test_build_db_payload_includes_knowledge_cards_block(self):
        payload = _build_db_payload(
            intent=type("Intent", (), {"value": "aggregation_db"})(),
            metric_alias="pm25",
            window_label="last hour",
            rows=[{"value": 12.0}],
            knowledge_cards=[
                {
                    "card_type": "interpretation",
                    "topic": "pm25",
                    "title": "PM2.5 acceptable",
                    "summary": "summary",
                    "content": "content",
                    "severity_level": "fair",
                    "source_label": "RESET",
                }
            ],
        )
        self.assertIn("knowledge_cards", payload)
        self.assertEqual(payload["knowledge_cards"][0]["topic"], "pm25")

    def test_prompt_context_has_labeled_sections(self):
        rendered = build_grounded_context_sections(
            measured_room_facts={"rows": [{"metric": "pm25", "value": 28.0}]},
            backend_semantic_state=None,
            knowledge_cards=[{"topic": "pm25", "title": "PM2.5 acceptable"}],
            communication_guardrails=[{"topic": "overall_air_quality", "title": "Avoid overclaiming health risk"}],
        )
        self.assertIn("## Measured Room Facts", rendered)
        self.assertIn("## Knowledge Interpretation Cards", rendered)
        self.assertIn("## Communication Guardrails", rendered)

    def test_shared_system_prompt_prioritizes_user_question(self):
        self.assertIn("user's exact question is the primary task", SHARED_SYSTEM_PROMPT)
        self.assertIn("If the user asks for \"risk(s)\", focus on concrete risks", SHARED_SYSTEM_PROMPT)

    def test_card_grounded_context_converts_windows_to_gmt4(self):
        rendered = build_card_grounded_context(
            cards=[
                {
                    "space": "smart_lab",
                    "window_start": "2026-03-28T10:15:28+00:00",
                    "window_end": "2026-03-28T11:15:28+00:00",
                    "overall_air_label": "moderate",
                    "summary_text": "Sample",
                    "distance": 0.1,
                }
            ],
            knowledge_cards=[],
            allow_general_knowledge=True,
        )
        self.assertIn("2026-03-28T14:15:28+04:00", rendered)
        self.assertIn("2026-03-28T15:15:28+04:00", rendered)

    @patch("executors.env_query_langchain.embed_texts")
    @patch("executors.env_query_langchain.get_cursor")
    def test_search_knowledge_cards_prefers_explanations_for_definition_query(
        self,
        mock_get_cursor,
        mock_embed_texts,
    ):
        mock_embed_texts.return_value = [[0.1, 0.2]]
        rows = [
            {
                "knowledge_card_id": "1",
                "card_type": "interpretation",
                "topic": "pm25",
                "title": "PM2.5 acceptable",
                "summary": "acceptable",
                "content": "interpretation content",
                "severity_level": "fair",
                "source_label": "RESET",
                "source_url_key": "RESET_AIR",
                "source_metadata": {},
                "distance": 0.91,
            },
            {
                "knowledge_card_id": "2",
                "card_type": "explanation",
                "topic": "pm25",
                "title": "What PM2.5 means",
                "summary": "definition",
                "content": "explanation content",
                "severity_level": None,
                "source_label": "EPA",
                "source_url_key": "EPA_PM25_AQI",
                "source_metadata": {},
                "distance": 0.72,
            },
        ]
        mock_get_cursor.return_value = _FakeCursor(rows)

        result = search_knowledge_cards("what does pm2.5 mean", k=2)
        self.assertEqual(result[0]["card_type"], "explanation")

    @patch("executors.env_query_langchain.embed_texts")
    @patch("executors.env_query_langchain.get_cursor")
    def test_search_knowledge_cards_can_return_interpretation_for_assessment_query(
        self,
        mock_get_cursor,
        mock_embed_texts,
    ):
        mock_embed_texts.return_value = [[0.1, 0.2]]
        rows = [
            {
                "knowledge_card_id": "1",
                "card_type": "caveat",
                "topic": "overall_air_quality",
                "title": "Avoid overclaiming health risk",
                "summary": "guardrail",
                "content": "guardrail content",
                "severity_level": None,
                "source_label": "Internal",
                "source_url_key": "INTERNAL_GUARDRAIL",
                "source_metadata": {},
                "distance": 0.95,
            },
            {
                "knowledge_card_id": "2",
                "card_type": "interpretation",
                "topic": "pm25",
                "title": "PM2.5 acceptable but not ideal",
                "summary": "assessment",
                "content": "interpretation content",
                "severity_level": "fair",
                "source_label": "RESET",
                "source_url_key": "RESET_AIR",
                "source_metadata": {},
                "distance": 0.80,
            },
        ]
        mock_get_cursor.return_value = _FakeCursor(rows)

        result = search_knowledge_cards("is 28 ug/m3 pm2.5 okay", k=2)
        self.assertEqual(result[0]["card_type"], "interpretation")

    @patch("executors.env_query_langchain.embed_texts")
    @patch("executors.env_query_langchain.get_cursor")
    def test_search_knowledge_cards_keeps_caveat_for_health_risk_queries(
        self,
        mock_get_cursor,
        mock_embed_texts,
    ):
        mock_embed_texts.return_value = [[0.1, 0.2]]
        rows = [
            {
                "knowledge_card_id": "1",
                "card_type": "interpretation",
                "topic": "pm25",
                "title": "PM2.5 poor indoor air",
                "summary": "poor",
                "content": "interpretation content",
                "severity_level": "poor",
                "source_label": "RESET",
                "source_url_key": "RESET_AIR",
                "source_metadata": {},
                "distance": 0.88,
            },
            {
                "knowledge_card_id": "2",
                "card_type": "caveat",
                "topic": "overall_air_quality",
                "title": "Avoid overclaiming health risk",
                "summary": "guardrail",
                "content": "guardrail content",
                "severity_level": None,
                "source_label": "Internal",
                "source_url_key": "INTERNAL_GUARDRAIL",
                "source_metadata": {},
                "distance": 0.70,
            },
        ]
        mock_get_cursor.return_value = _FakeCursor(rows)

        result = search_knowledge_cards("is this a health risk", k=2)
        self.assertEqual(result[0]["card_type"], "caveat")

    def test_db_point_lookup_directive_is_question_first_for_risk_queries(self):
        directive = db_response_directive("current_status_db", question="What are the main air quality risks today?")
        self.assertIn("First, directly answer the exact question asked.", directive)
        self.assertIn("start with the risk level", directive)

    def test_issue_triage_question_treated_as_air_quality_query(self):
        self.assertTrue(is_air_quality_query_text("Is there any issue right now in the smart lab?"))


if __name__ == "__main__":
    unittest.main()
