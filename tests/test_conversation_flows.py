"""Opt-in, live multi-turn conversation eval.

This is NOT part of the deterministic suite: it drives the golden conversation in
``conversation_harness`` against the real Ollama router/answer models, so it is
skipped unless ``RUN_LIVE_CONVERSATION_EVAL`` is set (and the router is reachable).

Two modes:

  RUN_LIVE_CONVERSATION_EVAL=1  python -m unittest tests.test_conversation_flows
      Runs the conversation, prints the scorecard, and asserts only the *baseline*
      guarantees: every turn produced a non-empty answer and the LLM router was
      actually used (no regex fallback). Safe to run at any point in the plan.

  STRICT_CONVERSATION_EVAL=1    (implies live)
      Additionally asserts every P0–P4 check passes — i.e. use it as the completion
      gate once all phases have landed. Expected to FAIL on the pre-Phase baseline;
      the failures enumerate exactly what the plan still needs to deliver.
"""

import os
import sys
import unittest

TEST_DIR = os.path.dirname(__file__)
SERVER_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)


def _truthy(name: str) -> bool:
    return str(os.getenv(name, "")).strip().lower() in {"1", "true", "yes", "on"}


_LIVE = _truthy("RUN_LIVE_CONVERSATION_EVAL") or _truthy("STRICT_CONVERSATION_EVAL")
_STRICT = _truthy("STRICT_CONVERSATION_EVAL")


@unittest.skipUnless(
    _LIVE,
    "Set RUN_LIVE_CONVERSATION_EVAL=1 (or STRICT_CONVERSATION_EVAL=1) to run the live "
    "multi-turn conversation eval against the Ollama models.",
)
class GoldenConversationEval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests.conversation_harness import (
            GOLDEN_CONVERSATION,
            print_transcript,
            run_conversation,
            summarize,
        )

        cls.results = run_conversation(GOLDEN_CONVERSATION)
        print_transcript(cls.results)
        cls.passed, cls.total = summarize(cls.results)

    def test_every_turn_answered(self):
        for tr in self.results:
            with self.subTest(turn=tr.index, user=tr.user):
                self.assertTrue(tr.answer.strip(), f"turn {tr.index} produced an empty answer")

    def test_llm_router_was_used(self):
        # If any turn fell back to regex, the router LLM was unreachable and the eval
        # is not a real LLM conversation — fail loudly rather than report a hollow pass.
        fell_back = [tr.index for tr in self.results if tr.fallback_used]
        self.assertEqual(fell_back, [], f"turns fell back to regex (router unreachable): {fell_back}")

    @unittest.skipUnless(_STRICT, "Set STRICT_CONVERSATION_EVAL=1 to enforce all P0–P4 checks.")
    def test_all_expectations_pass(self):
        for tr in self.results:
            for chk in tr.checks:
                with self.subTest(turn=tr.index, phase=chk.phase, check=chk.name):
                    self.assertTrue(chk.passed, f"turn {tr.index} ({chk.phase}) {chk.name}: {chk.detail}")


if __name__ == "__main__":
    unittest.main()
