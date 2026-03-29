import asyncio
import os
import sys
import unittest


TEST_DIR = os.path.dirname(__file__)
SERVER_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)
REPO_DIR = os.path.abspath(os.path.join(SERVER_DIR, ".."))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from http_routes.query_runtime import execute_non_stream_query


class QueryRuntimeContextTests(unittest.TestCase):
    def test_execute_non_stream_query_passes_context_separately(self):
        seen = {}

        def _fake_execute_query(
            latest_user_question,
            k,
            lab_name,
            allow_clarify,
            endpoint_key,
            conversation_context,
        ):
            seen["latest_user_question"] = latest_user_question
            seen["conversation_context"] = conversation_context
            return {
                "answer": "ok",
                "timescale": "knowledge",
                "cards_retrieved": 0,
                "recent_card": False,
                "metadata": {"executor": "knowledge_qa"},
                "visualization_type": "none",
                "chart": None,
            }

        result = asyncio.run(
            execute_non_stream_query(
                question="What does IEQ mean?",
                latest_user_question="What does IEQ mean?",
                conversation_context=(
                    "Previous conversation context (most recent last):\n"
                    "User: Explain PM2.5\nAssistant: PM2.5 is particulate matter."
                ),
                k=4,
                lab_name=None,
                allow_clarify=True,
                conversation_id=None,
                context_applied=True,
                endpoint_key="query_sync",
                execute_query_fn=_fake_execute_query,
            )
        )
        self.assertEqual(seen["latest_user_question"], "What does IEQ mean?")
        self.assertIn("Previous conversation context", seen["conversation_context"])
        self.assertEqual(result["result"]["answer"], "ok")


if __name__ == "__main__":
    unittest.main()
