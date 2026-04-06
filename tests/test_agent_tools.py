import os
import sys
import unittest
from unittest.mock import patch


TEST_DIR = os.path.dirname(__file__)
SERVER_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

from query_routing.agent_tools import execute_agent_tool_call, summarize_tool_observation
from query_routing.intent_classifier import IntentType


class AgentToolsTests(unittest.TestCase):
    @patch("query_routing.agent_tools.run_db_query")
    def test_query_db_tool_uses_requested_intent(self, mock_run_db):
        mock_run_db.return_value = {"answer": "ok", "timescale": "1hour", "cards_retrieved": 0}
        payload = execute_agent_tool_call(
            tool_name="query_db",
            question="Average CO2 this week",
            k=5,
            lab_name="smart_lab",
            arguments={"intent": IntentType.AGGREGATION_DB.value},
        )
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["intent"], IntentType.AGGREGATION_DB.value)
        self.assertEqual(mock_run_db.call_args.kwargs.get("intent"), IntentType.AGGREGATION_DB)

    @patch("query_routing.agent_tools.answer_env_question_with_metadata")
    def test_knowledge_tool_returns_structured_payload(self, mock_answer):
        mock_answer.return_value = {
            "answer": "IEQ means indoor environmental quality.",
            "cards_retrieved": 2,
            "knowledge_cards_retrieved": 2,
        }
        payload = execute_agent_tool_call(
            tool_name="search_knowledge_cards",
            question="What is IEQ?",
            k=3,
            lab_name=None,
            arguments=None,
        )
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["intent"], IntentType.DEFINITION_EXPLANATION.value)
        self.assertIn("answer", payload["result"])

    def test_tool_summary_handles_failure(self):
        summary = summarize_tool_observation({"ok": False, "tool_name": "query_db", "error": "timeout"})
        self.assertIn("failed", summary)


if __name__ == "__main__":
    unittest.main()
