"""Router transport/parse failure handling and the JSON-constrained request.

The router degrades to a regex plan in two very different situations, and conflating them
cost real debugging time: an unreachable Ollama is an infrastructure problem worth
retrying, while a malformed plan is a model/prompt problem that a deterministic retry can
only reproduce.
"""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

TEST_DIR = os.path.dirname(__file__)
SERVER_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

from query_routing import llm_router_planner as planner
from query_routing.intent_classifier import IntentType


def _chat_response(content: str):
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"message": {"content": content}}
    return resp


_VALID_PLAN = json.dumps(
    {
        "intent": "aggregation_db",
        "lab": "smart_lab",
        "metrics": ["co2"],
        "time_phrase": "last week",
        "confidence": 0.9,
        "resolved_question": "what was the average co2 in smart_lab last week?",
    }
)


class RouterRequestShapeTests(unittest.TestCase):
    def test_request_constrains_output_to_json(self):
        _, body = planner._router_chat_request("what is the co2?", "smart_lab", "")
        self.assertEqual(body["format"], "json")

    def test_token_budget_fits_a_full_plan(self):
        _, body = planner._router_chat_request("what is the co2?", None, "")
        # The plan carries resolved_question and clarification_question; a tight budget
        # truncates the JSON and silently drops the route to the regex fallback.
        self.assertGreaterEqual(body["options"]["num_predict"], 512)

    def test_temperature_stays_deterministic(self):
        _, body = planner._router_chat_request("what is the co2?", None, "")
        self.assertEqual(body["options"]["temperature"], 0.0)


class RouterTransportFailureTests(unittest.TestCase):
    def test_transport_error_is_retried_then_falls_back(self):
        with patch.object(planner, "requests") as fake_requests, \
             patch.object(planner, "router_max_retries", return_value=3), \
             patch.object(planner, "router_retry_jitter_ms", return_value=0):
            fake_requests.post.side_effect = ConnectionError("connection refused")
            plan = planner.plan_route("average co2 last week", "smart_lab")

        self.assertTrue(plan.fallback_used)
        self.assertEqual(fake_requests.post.call_count, 3)

    def test_transport_failure_logs_as_unreachable(self):
        with patch.object(planner, "requests") as fake_requests, \
             patch.object(planner, "router_max_retries", return_value=1), \
             patch.object(planner._log, "warning") as warn:
            fake_requests.post.side_effect = ConnectionError("connection refused")
            planner.plan_route("average co2 last week", None)

        self.assertIn("unreachable", warn.call_args.args[0])

    def test_recovers_on_a_later_attempt(self):
        with patch.object(planner, "requests") as fake_requests, \
             patch.object(planner, "router_max_retries", return_value=3), \
             patch.object(planner, "router_retry_jitter_ms", return_value=0):
            fake_requests.post.side_effect = [
                ConnectionError("boom"),
                _chat_response(_VALID_PLAN),
            ]
            plan = planner.plan_route("average co2 last week", "smart_lab")

        self.assertFalse(plan.fallback_used)
        self.assertEqual(plan.intent, IntentType.AGGREGATION_DB)


class RouterParseFailureTests(unittest.TestCase):
    def _plan_with_content(self, content: str, retries: int = 3):
        with patch.object(planner, "requests") as fake_requests, \
             patch.object(planner, "router_max_retries", return_value=retries), \
             patch.object(planner, "router_retry_jitter_ms", return_value=0):
            fake_requests.post.return_value = _chat_response(content)
            plan = planner.plan_route("average co2 last week", "smart_lab")
            return plan, fake_requests.post.call_count

    def test_unparseable_content_is_not_retried(self):
        plan, calls = self._plan_with_content("I think you want the CO2 trend.")
        self.assertTrue(plan.fallback_used)
        # Same prompt at temperature 0 would produce the same junk — one attempt is enough.
        self.assertEqual(calls, 1)

    def test_empty_content_is_not_retried(self):
        plan, calls = self._plan_with_content("")
        self.assertTrue(plan.fallback_used)
        self.assertEqual(calls, 1)

    def test_unknown_intent_is_a_parse_failure(self):
        plan, calls = self._plan_with_content(json.dumps({"intent": "make_coffee"}))
        self.assertTrue(plan.fallback_used)
        self.assertEqual(calls, 1)

    def test_parse_failure_logs_the_offending_content(self):
        with patch.object(planner, "requests") as fake_requests, \
             patch.object(planner, "router_max_retries", return_value=2), \
             patch.object(planner._log, "warning") as warn:
            fake_requests.post.return_value = _chat_response("not json at all")
            planner.plan_route("average co2 last week", None)

        message = warn.call_args.args[0]
        self.assertIn("unusable plan", message)
        self.assertIn("not json at all", warn.call_args.args[-1])

    def test_truncated_json_falls_back_rather_than_half_routing(self):
        plan, _ = self._plan_with_content('{"intent": "aggregation_db", "metrics": ["co2"')
        self.assertTrue(plan.fallback_used)


class RouterClarificationPromptTests(unittest.TestCase):
    """Rules the prompt must keep stating, since deleting the regex clarify gate made
    the prompt the only thing preventing these behaviours."""

    def test_prompt_treats_contrastive_place_references_as_unresolvable(self):
        # "over there" points at a DIFFERENT space than the one under discussion, so a
        # lab in prior conversation must not silently satisfy it.
        prompt = planner._SYSTEM_PROMPT
        self.assertIn("over there", prompt)
        self.assertIn("CONTRASTIVE", prompt)

    def test_prompt_keeps_unit_conversion_followups_on_the_prior_intent(self):
        prompt = planner._SYSTEM_PROMPT
        self.assertIn("give me that in centimeters", prompt.lower())
        self.assertIn("RESTATES", prompt)


class RouterClarificationParsingTests(unittest.TestCase):
    def test_clarification_is_parsed(self):
        raw = json.dumps(
            {
                "intent": "current_status_db",
                "confidence": 0.4,
                "needs_clarification": True,
                "clarification_question": "Which space did you mean?",
            }
        )
        plan = planner._parse_llm_response(raw, "how is it over there?", None)
        self.assertTrue(plan.needs_clarification)
        self.assertEqual(plan.clarification_question, "Which space did you mean?")

    def test_flag_without_a_question_is_ignored(self):
        # A clarification with nothing to ask would dead-end the turn.
        raw = json.dumps({"intent": "current_status_db", "needs_clarification": True})
        plan = planner._parse_llm_response(raw, "how is it?", None)
        self.assertFalse(plan.needs_clarification)

    def test_absent_field_defaults_to_no_clarification(self):
        plan = planner._parse_llm_response(_VALID_PLAN, "average co2 last week", None)
        self.assertFalse(plan.needs_clarification)
        self.assertIsNone(plan.clarification_question)

    def test_regex_fallback_never_asks_for_clarification(self):
        # The emergency planner has no language understanding to justify a question.
        plan = planner._fallback_plan("average co2 last week", None)
        self.assertFalse(plan.needs_clarification)


if __name__ == "__main__":
    unittest.main()
