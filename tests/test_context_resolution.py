"""Deterministic coverage for LLM context resolution (Phase 1).

Verifies the mechanics without hitting a live LLM:
  * the router parses `resolved_question` and falls back to the raw question,
  * the emergency regex fallback still yields a usable question,
  * the orchestrator routes on the raw question but executes the resolved one,
  * the resolved question is surfaced in metadata only when it actually changed.
"""

import json
import os
import sys
import unittest
from unittest.mock import patch

TEST_DIR = os.path.dirname(__file__)
SERVER_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

from query_routing import query_orchestrator as qo
from query_routing.intent_classifier import IntentType
from query_routing.llm_router_planner import _fallback_plan, _parse_llm_response
from query_routing.router_types import RoutePlan
from storage.conversation_context import ConversationContext


def _router_json(**overrides):
    payload = {
        "intent": "aggregation_db",
        "lab": "smart_lab",
        "metrics": ["humidity"],
        "time_phrase": "first week of May",
        "confidence": 0.9,
    }
    payload.update(overrides)
    return json.dumps(payload)


class RouterResolvedQuestionParsing(unittest.TestCase):
    def test_resolved_question_is_parsed(self):
        raw = _router_json(resolved_question="what was the humidity in smart_lab in the first week of May?")
        plan = _parse_llm_response(raw, "what about humidity?", "smart_lab")
        self.assertEqual(
            plan.resolved_question,
            "what was the humidity in smart_lab in the first week of May?",
        )

    def test_missing_resolved_question_falls_back_to_raw(self):
        plan = _parse_llm_response(_router_json(), "what about humidity?", "smart_lab")
        self.assertEqual(plan.resolved_question, "what about humidity?")

    def test_empty_resolved_question_falls_back_to_raw(self):
        plan = _parse_llm_response(_router_json(resolved_question="   "), "raw q", None)
        self.assertEqual(plan.resolved_question, "raw q")

    def test_null_resolved_question_falls_back_to_raw(self):
        plan = _parse_llm_response(_router_json(resolved_question=None), "raw q", None)
        self.assertEqual(plan.resolved_question, "raw q")

    def test_fallback_plan_sets_resolved_question(self):
        plan = _fallback_plan("average co2 last week", None)
        self.assertTrue(plan.fallback_used)
        self.assertEqual(plan.resolved_question, "average co2 last week")


def _ctx(original, effective=None):
    return ConversationContext(
        conversation_id="c1",
        original_question=original,
        raw_block="Previous conversation context (most recent last):\nUser: earlier",
        effective_question=effective or original,
        effective_lab="smart_lab",
        routing_snippet="User: earlier",
        llm_history="User: earlier",
        carried_metric=None,
        carried_time_phrase=None,
    )


def _route(resolved_question):
    return RoutePlan(
        intent=IntentType.AGGREGATION_DB,
        confidence=0.9,
        lab_name="smart_lab",
        time_phrase=None,
        model="test",
        fallback_used=False,
        metrics=["humidity"],
        resolved_question=resolved_question,
    )


class OrchestratorUsesResolvedQuestion(unittest.TestCase):
    def _run(self, ctx, route):
        captured = {}

        def fake_run_db_query(question, intent, lab_name, planner_hints=None, conversation_context=""):
            captured["question"] = question
            return {"answer": "ok", "timescale": "1hour", "metrics_used": ["humidity"]}

        with patch.object(qo, "plan_route", return_value=route), \
             patch.object(qo, "run_db_query", side_effect=fake_run_db_query):
            result = qo.execute_query(ctx, k=5)
        return captured, result

    def test_executor_receives_resolved_question(self):
        ctx = _ctx("what about humidity?")
        route = _route("what was the humidity in smart_lab in the first week of May?")
        captured, result = self._run(ctx, route)
        self.assertEqual(captured["question"], route.resolved_question)
        # And it is surfaced in metadata because it changed.
        self.assertEqual(result["metadata"]["resolved_question"], route.resolved_question)

    def test_routing_still_uses_raw_question(self):
        # plan_route must be called with the clean current-turn question, never the resolved one.
        ctx = _ctx("what about humidity?")
        route = _route("what was the humidity in smart_lab in the first week of May?")
        with patch.object(qo, "plan_route", return_value=route) as mock_plan, \
             patch.object(qo, "run_db_query", return_value={"answer": "ok", "timescale": "1hour"}):
            qo.execute_query(ctx, k=5)
        mock_plan.assert_called_once()
        self.assertEqual(mock_plan.call_args.args[0], "what about humidity?")

    def test_no_resolution_uses_effective_question_and_omits_metadata(self):
        # When the router returns no rewrite (already self-contained), the executor gets the
        # effective question and no resolved_question key is added to metadata.
        ctx = _ctx("what is the co2 in smart_lab?")
        route = _route(None)
        captured, result = self._run(ctx, route)
        self.assertEqual(captured["question"], "what is the co2 in smart_lab?")
        self.assertNotIn("resolved_question", result["metadata"])

    def test_stream_surfaces_the_resolved_question_in_meta(self):
        # The stream renderer carries the same observability field as the sync body.
        import asyncio

        ctx = _ctx("what about humidity?")
        route = _route("what was the humidity in smart_lab in the first week of May?")

        async def _fake_plan(*a, **kw):
            return route

        async def _fake_tokens(*a, **kw):
            yield 'data: {"event": "done"}\n\n'

        async def _run():
            return [c async for c in qo.stream_query(ctx, k=5)]

        with patch.object(qo, "plan_route_async", side_effect=_fake_plan), \
             patch.object(qo, "prepare_db_query", return_value={"metrics_used": ["humidity"]}), \
             patch.object(qo, "stream_db_tokens", side_effect=_fake_tokens):
            chunks = asyncio.new_event_loop().run_until_complete(_run())

        metas = [
            json.loads(c.removeprefix("data: ").strip())
            for c in chunks
            if '"event": "meta"' in c
        ]
        self.assertEqual(metas[0]["resolved_question"], route.resolved_question)


if __name__ == "__main__":
    unittest.main()
