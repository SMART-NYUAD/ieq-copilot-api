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

from http_routes.route_helpers import execute_non_stream_query
from storage.conversation_context import ConversationContext


class QueryRuntimeContextTests(unittest.TestCase):
    def test_execute_non_stream_query_passes_context_to_executor(self):
        seen = {}

        def _fake_execute_query(ctx, k, allow_clarify, endpoint_key):
            seen["effective_question"] = ctx.effective_question
            seen["llm_history"] = ctx.llm_history
            seen["conversation_id"] = ctx.conversation_id
            return {
                "answer": "ok",
                "timescale": "knowledge",
                "cards_retrieved": 0,
                "recent_card": False,
                "metadata": {"executor": "knowledge_qa"},
            }

        prior_block = (
            "Previous conversation context (most recent last):\n"
            "User: Explain PM2.5\nAssistant: PM2.5 is particulate matter."
        )
        ctx = ConversationContext(
            conversation_id="test-conv-id",
            original_question="What does IEQ mean?",
            raw_block=prior_block,
            effective_question="What does IEQ mean?",
            effective_lab=None,
            routing_snippet="User: Explain PM2.5\nAssistant: PM2.5 is particulate matter.",
            llm_history="User: Explain PM2.5\nAssistant: PM2.5 is particulate matter.",
        )

        result = asyncio.run(
            execute_non_stream_query(
                ctx=ctx,
                k=4,
                allow_clarify=True,
                endpoint_key="query_sync",
                execute_query_fn=_fake_execute_query,
            )
        )
        self.assertEqual(seen["effective_question"], "What does IEQ mean?")
        self.assertIn("PM2.5", seen["llm_history"])
        self.assertEqual(result["result"]["answer"], "ok")


if __name__ == "__main__":
    unittest.main()
