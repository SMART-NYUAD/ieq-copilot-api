"""The knowledge path must degrade to grounded text when the answer LLM is down.

Previously the sync path let the Ollama exception escape (HTTP 500) and the stream
swallowed it and emitted only ``done`` — a silently blank answer that was also
skipped by turn persistence. Both now fall back to the retrieved card text, matching
the DB / IFC / sensor executors.
"""

import asyncio
import json
import os
import sys
import unittest
from unittest.mock import patch

TEST_DIR = os.path.dirname(__file__)
SERVER_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

from executors import knowledge_executor as kx


_CARDS = [
    {
        "card_type": "explanation",
        "title": "CO2 basics",
        "summary": "CO2 is a gas exhaled by occupants; indoor levels track ventilation.",
        "content": "Longer body text.",
        "source_label": "Internal",
    }
]


def _events(chunks):
    out = []
    for chunk in chunks:
        raw = str(chunk).removeprefix("data: ").strip()
        if raw:
            out.append(json.loads(raw))
    return out


class DeterministicKnowledgeAnswerTests(unittest.TestCase):
    def test_quotes_the_top_card(self):
        answer = kx._deterministic_knowledge_answer(_CARDS)
        self.assertIn("CO2 is a gas exhaled by occupants", answer)
        self.assertIn("CO2 basics", answer)

    def test_falls_back_to_content_when_summary_missing(self):
        answer = kx._deterministic_knowledge_answer(
            [{"title": "", "summary": "", "content": "Body only."}]
        )
        self.assertEqual(answer, "Body only.")

    def test_no_cards_returns_the_unavailable_notice_not_an_invention(self):
        answer = kx._deterministic_knowledge_answer([])
        self.assertEqual(answer, kx._NO_LLM_KNOWLEDGE_ANSWER)

    def test_long_card_is_truncated(self):
        answer = kx._deterministic_knowledge_answer([{"summary": "x" * 5000}])
        self.assertLessEqual(len(answer), kx._DETERMINISTIC_ANSWER_MAX_CHARS + 1)
        self.assertTrue(answer.endswith("…"))


class SyncKnowledgeFallbackTests(unittest.TestCase):
    def _answer(self, generate_side_effect):
        with patch.object(kx, "_build_knowledge_context", return_value={"knowledge_cards": _CARDS}), \
             patch.object(kx, "wants_guideline_detail", return_value=False), \
             patch.object(kx, "generate_ollama_text", side_effect=generate_side_effect):
            return kx.answer_env_question_with_metadata(user_question="what is co2?", k=3)

    def test_llm_exception_does_not_propagate(self):
        result = self._answer(RuntimeError("ollama unreachable"))
        self.assertFalse(result["llm_used"])
        self.assertIn("CO2 is a gas exhaled by occupants", result["answer"])

    def test_empty_llm_output_falls_back(self):
        result = self._answer(lambda *a, **kw: "   ")
        self.assertFalse(result["llm_used"])
        self.assertIn("CO2 basics", result["answer"])

    def test_successful_generation_marks_llm_used(self):
        result = self._answer(lambda *a, **kw: "CO2 is carbon dioxide.")
        self.assertTrue(result["llm_used"])
        self.assertEqual(result["answer"], "CO2 is carbon dioxide.")


class StreamKnowledgeFallbackTests(unittest.TestCase):
    def _collect(self):
        class _FailingClient:
            async def __aenter__(self):
                raise RuntimeError("ollama unreachable")

            async def __aexit__(self, *exc):
                return False

        async def _run():
            chunks = []
            async for chunk in kx.stream_knowledge_tokens(user_question="what is co2?", k=3):
                chunks.append(chunk)
            return chunks

        with patch.object(kx, "_build_knowledge_context", return_value={"knowledge_cards": _CARDS}), \
             patch.object(kx, "wants_guideline_detail", return_value=False), \
             patch.object(kx.httpx, "AsyncClient", return_value=_FailingClient()):
            return asyncio.new_event_loop().run_until_complete(_run())

    def test_stream_is_never_empty_when_the_llm_is_unreachable(self):
        events = _events(self._collect())
        tokens = [e for e in events if e["event"] == "token"]
        self.assertEqual(len(tokens), 1)
        self.assertIn("CO2 is a gas exhaled by occupants", tokens[0]["text"])
        self.assertEqual(events[-1]["event"], "done")

    def test_stream_still_emits_a_sources_frame(self):
        events = _events(self._collect())
        self.assertEqual([e["event"] for e in events][-2:], ["sources", "done"])


if __name__ == "__main__":
    unittest.main()
