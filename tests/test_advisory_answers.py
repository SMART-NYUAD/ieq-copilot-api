"""Advice questions must be answered as advice, not as a status report.

Reported symptom: "How can I improve the air quality?" and "What would you
recommend to improve VOC?" came back as metric-by-metric readings, the second one
closing with "No immediate action is needed" — a status dump where the user asked
what to do.

The cause was not a missing instruction. Four separate prompts already said the
model MUST give recommendations when asked. The problem was that nothing upstream
ever recognised the question as advisory, so the directive chosen for it was one
of the status directives, whose *structure* ("provide an overall status", "include
metric-by-metric interpretation") is what the model followed. Probing the router
before the fix, the five advice questions below scattered across
``definition_explanation`` (an advice question read as a glossary lookup),
``current_status_db`` + ``analysis_mode=diagnostic``, and one unparseable plan.

So the fix is a route, not louder prose: the router sets
``analysis_mode="advisory"``, that reaches the executor as a resolved flag on the
payload, and ``db_response_directive`` returns an advisory directive that leads
with actions. These tests pin the three links of that chain that do not need the
network — the router's own decision is measured by ``tests/router_eval.py``
(group ``advice``), since the suite here stubs the router out.
"""

import os
import sys
import unittest

TEST_DIR = os.path.dirname(__file__)
SERVER_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

from fake_sensor_api import FakeSensorApiMixin
from executors import db_query_executor
from executors.db_support import response_helpers as db_helpers
from query_routing.intent_classifier import IntentType
from prompting.db_prompts import (
    DB_TOOL_RESPONSE_DIRECTIVE_ADVISORY,
    DB_TOOL_RESPONSE_DIRECTIVE_DIAGNOSTIC,
    DB_TOOL_RESPONSE_DIRECTIVE_POINT_LOOKUP,
)
from query_routing.llm_router_planner import _VALID_ANALYSIS_MODES


ADVISORY_QUESTIONS = [
    "How can I improve the air quality?",
    "What would you recommend to improve VOC?",
    "What should I do about the CO2?",
    "Any advice on making the space more comfortable?",
    "How do I lower PM2.5?",
    "what are the next steps?",
]

STATUS_QUESTIONS = [
    "what is the CO2?",
    "how is the air quality?",
    "what was the average temperature last week?",
    "is the humidity too high?",
]


class AdvisoryDirectiveSelectionTests(unittest.TestCase):
    def test_router_accepts_advisory_analysis_mode(self):
        self.assertIn("advisory", _VALID_ANALYSIS_MODES)

    def test_advisory_flag_selects_the_advisory_directive(self):
        # The resolved router signal wins regardless of intent or question wording.
        for intent in ("current_status_db", "point_lookup_db", "aggregation_db"):
            with self.subTest(intent=intent):
                self.assertEqual(
                    db_helpers.db_response_directive(
                        intent, question="what is the CO2?", advisory=True
                    ),
                    DB_TOOL_RESPONSE_DIRECTIVE_ADVISORY,
                )

    def test_advisory_beats_diagnostic(self):
        # Asked why AND what to do, the action is the thing they wanted.
        directive = db_helpers.db_response_directive(
            "current_status_db",
            question="why is the IEQ low and what should I do about it?",
            diagnostic=True,
            advisory=True,
        )
        self.assertEqual(directive, DB_TOOL_RESPONSE_DIRECTIVE_ADVISORY)

    def test_status_questions_keep_their_directive(self):
        # The guard in the other direction: adding advisory must not make every
        # reading grow a recommendations section.
        directive = db_helpers.db_response_directive(
            "current_status_db", question="what is the CO2?", advisory=False
        )
        self.assertEqual(directive, DB_TOOL_RESPONSE_DIRECTIVE_POINT_LOOKUP)

    def test_diagnostic_still_reachable_when_not_advisory(self):
        directive = db_helpers.db_response_directive(
            "current_status_db",
            question="what is driving the IEQ down?",
            diagnostic=True,
            advisory=False,
        )
        self.assertEqual(directive, DB_TOOL_RESPONSE_DIRECTIVE_DIAGNOSTIC)


class AdvisoryKeywordFallbackTests(unittest.TestCase):
    """The emergency path only — the router decides this when it is reachable."""

    def test_advisory_phrasings_detected(self):
        for question in ADVISORY_QUESTIONS:
            with self.subTest(question=question):
                self.assertTrue(db_helpers.is_advisory_query_text(question))

    def test_status_phrasings_not_detected(self):
        for question in STATUS_QUESTIONS:
            with self.subTest(question=question):
                self.assertFalse(db_helpers.is_advisory_query_text(question))

    def test_keyword_fallback_selects_advisory_when_caller_passes_nothing(self):
        directive = db_helpers.db_response_directive(
            "current_status_db", question="How can I improve the air quality?"
        )
        self.assertEqual(directive, DB_TOOL_RESPONSE_DIRECTIVE_ADVISORY)


class AdvisoryPayloadPlumbingTests(FakeSensorApiMixin, unittest.TestCase):
    """``response_mode`` rides the payload so both renderers read one resolved value.

    render_sync and render_stream each build their own prompt text. If they derived
    "is this advisory?" independently they could disagree, which is the drift that
    has already cost this codebase its streamed citations once.
    """

    def _prepare(self, question, planner_hints):
        return db_query_executor.prepare_db_query(
            question=question,
            intent=IntentType.CURRENT_STATUS_DB,
            lab_name=None,
            planner_hints=planner_hints,
        )

    def test_router_advisory_mode_reaches_the_payload(self):
        context = self._prepare(
            "How can I improve the air quality?", {"analysis_mode": "advisory"}
        )
        self.assertEqual(context["payload"].get("response_mode"), "advisory")

    def test_plain_status_question_carries_no_response_mode(self):
        context = self._prepare("what is the CO2?", {"analysis_mode": None})
        self.assertIsNone(context["payload"].get("response_mode"))

    def test_keyword_fallback_applies_when_router_supplied_no_mode(self):
        # Router unreachable: planner_hints carry no analysis_mode at all.
        context = self._prepare("How can I improve the air quality?", {})
        self.assertEqual(context["payload"].get("response_mode"), "advisory")

    def test_keyword_heuristic_never_overrules_the_router(self):
        # "how can I improve" trips the keyword list, but the router said diagnostic.
        # LLM-primary: the heuristic only speaks when the router said nothing.
        context = self._prepare(
            "why is the IEQ low - how can I improve it?", {"analysis_mode": "diagnostic"}
        )
        self.assertIsNone(context["payload"].get("response_mode"))


class AdvisoryDirectiveContentTests(unittest.TestCase):
    """What the directive tells the model, phrased as behaviour we depend on."""

    def test_directive_demands_actions_over_status(self):
        text = DB_TOOL_RESPONSE_DIRECTIVE_ADVISORY.lower()
        self.assertIn("recommendations are the answer", text)
        self.assertIn("do not open with a status report", text)

    def test_directive_forbids_the_no_action_non_answer(self):
        # The reported VOC answer ended on "No immediate action is needed", which
        # does not answer "what would you recommend?".
        self.assertIn(
            '"no action needed" does not answer',
            DB_TOOL_RESPONSE_DIRECTIVE_ADVISORY.lower(),
        )

    def test_directive_suppresses_the_metric_rundown(self):
        text = DB_TOOL_RESPONSE_DIRECTIVE_ADVISORY.lower()
        self.assertIn("metric-by-metric rundown", text)
        self.assertIn("missing-metric disclaimer", text)


if __name__ == "__main__":
    unittest.main()
