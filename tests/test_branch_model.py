"""One branch ladder, two renderers.

The sync and stream paths used to carry independent intent ladders, and they drifted —
the stream stopped reporting citations while the sync body kept returning them. These
tests hold the two renderers to the same branch: whatever ``plan_branch`` decides, both
must describe it identically.
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

from query_routing import query_orchestrator as qo
from query_routing.intent_classifier import IntentType
from query_routing.router_types import RoutePlan
from storage.conversation_context import ConversationContext


def _ctx(question="what is the co2 in smart_lab?"):
    return ConversationContext(
        conversation_id="c1",
        original_question=question,
        raw_block="",
        effective_question=question,
        effective_lab="smart_lab",
        routing_snippet="",
        llm_history="",
    )


def _route(intent=IntentType.CURRENT_STATUS_DB, **kwargs):
    params = dict(
        intent=intent,
        confidence=0.9,
        lab_name="smart_lab",
        time_phrase=None,
        model="test-model",
        fallback_used=False,
        metrics=["co2"],
    )
    params.update(kwargs)
    return RoutePlan(**params)


def _events(chunks):
    out = []
    for chunk in chunks:
        raw = str(chunk).removeprefix("data: ").strip()
        if raw:
            out.append(json.loads(raw))
    return out


def _drain(agen):
    async def _run():
        return [chunk async for chunk in agen]

    return asyncio.new_event_loop().run_until_complete(_run())


# Every instant branch: (label, route kwargs, expected executor).
_INSTANT_BRANCHES = [
    ("viewer", dict(intent=IntentType.VIEWER_CONTROL, viewer_type="splat"), "viewer_control"),
    ("heatmap", dict(intent=IntentType.HEATMAP_CONTROL, heatmap_action="on"), "heatmap_control"),
    ("download", dict(intent=IntentType.DOWNLOAD_DATA, download_metric="co2"), "download_data"),
    ("download_no_metric", dict(intent=IntentType.DOWNLOAD_DATA), "download_data"),
    ("guardrail", dict(intent=IntentType.UNKNOWN_FALLBACK), "guardrail"),
    (
        "clarify",
        dict(needs_clarification=True, clarification_question="Which space did you mean?"),
        "clarify_gate",
    ),
]


class InstantBranchParityTests(unittest.TestCase):
    """Branches that answer without a model call must render identically both ways."""

    def _both(self, route):
        branch = qo.plan_branch(_ctx(), route)
        sync = qo.render_sync(branch)
        events = _events(_drain(qo.render_stream(branch)))
        return sync, events

    def test_every_instant_branch_agrees_across_renderers(self):
        for label, kwargs, expected_executor in _INSTANT_BRANCHES:
            with self.subTest(branch=label):
                sync, events = self._both(_route(**kwargs))
                meta = next(e for e in events if e["event"] == "meta")
                tokens = "".join(e.get("text", "") for e in events if e["event"] == "token")

                self.assertEqual(sync["metadata"]["executor"], expected_executor)
                self.assertEqual(meta["executor"], expected_executor)
                self.assertEqual(sync["answer"], tokens)
                self.assertEqual(sync["timescale"], meta["timescale"])
                self.assertEqual(sync["metadata"]["ui"], meta["ui"])
                self.assertEqual(sync["metadata"]["intent"], meta["intent"])
                self.assertEqual(events[-1]["event"], "done")

    def test_planner_model_and_confidence_reach_both_renderers(self):
        sync, events = self._both(_route(intent=IntentType.VIEWER_CONTROL, viewer_type="pc"))
        meta = next(e for e in events if e["event"] == "meta")
        for key in ("planner_model", "route_confidence", "fallback_used"):
            self.assertEqual(sync["metadata"][key], meta[key], key)


class ClarifyBranchTests(unittest.TestCase):
    def test_router_clarification_becomes_the_answer(self):
        route = _route(needs_clarification=True, clarification_question="Which space did you mean?")
        result = qo.render_sync(qo.plan_branch(_ctx(), route))
        self.assertEqual(result["answer"], "Which space did you mean?")
        self.assertEqual(result["timescale"], "clarify")
        self.assertEqual(result["metadata"]["ui"]["mode"], "clarify")

    def test_allow_clarify_false_executes_instead_of_asking(self):
        route = _route(needs_clarification=True, clarification_question="Which space?")
        branch = qo.plan_branch(_ctx(), route, allow_clarify=False)
        self.assertEqual(branch.name, "db_query")

    def test_clarification_is_skipped_when_the_router_did_not_ask(self):
        branch = qo.plan_branch(_ctx(), _route())
        self.assertEqual(branch.name, "db_query")

    def test_clarify_outranks_the_executor_choice(self):
        # Even a UI-control intent defers to a clarification the router asked for.
        route = _route(
            intent=IntentType.VIEWER_CONTROL,
            viewer_type="splat",
            needs_clarification=True,
            clarification_question="Which view?",
        )
        self.assertEqual(qo.plan_branch(_ctx(), route).name, "clarify_gate")


class GeneratedBranchTests(unittest.TestCase):
    """Branches that call a model: prepared work is shared, citations survive."""

    def test_db_branch_prepares_once_and_reuses_it_for_the_stream(self):
        route = _route(intent=IntentType.AGGREGATION_DB)
        branch = qo.plan_branch(_ctx(), route)
        prepared = {"timescale": "1week", "metrics_used": ["co2"], "time_window": {}, "resolved_lab_name": "smart_lab"}
        captured = {}

        async def _fake_tokens(*_args, **kwargs):
            captured["query_context"] = kwargs.get("query_context")
            yield 'data: {"event": "done"}\n\n'

        with patch.object(qo, "prepare_db_query", return_value=prepared) as prep, \
             patch.object(qo, "stream_db_tokens", side_effect=_fake_tokens):
            events = _events(_drain(qo.render_stream(branch)))

        prep.assert_called_once()
        self.assertIs(captured["query_context"], prepared)
        # The placeholder meta is corrected once the query has run.
        meta = next(e for e in events if e["event"] == "meta")
        update = next(e for e in events if e["event"] == "meta_update")
        self.assertEqual(meta["timescale"], "pending")
        self.assertEqual(update["timescale"], "1week")

    def test_knowledge_branch_grounds_both_renderers_in_the_same_snapshot(self):
        route = _route(intent=IntentType.DEFINITION_EXPLANATION)
        branch = qo.plan_branch(_ctx("what is co2?"), route)
        snapshot = {"rows": [{"co2": 430}]}
        seen = []

        def _fake_answer(**kwargs):
            seen.append(kwargs.get("live_sensor_data"))
            return {"answer": "CO2 is carbon dioxide.", "llm_used": True}

        async def _fake_stream(**kwargs):
            seen.append(kwargs.get("live_sensor_data"))
            yield 'data: {"event": "done"}\n\n'

        with patch.object(qo, "_fetch_live_sensor_data", return_value=snapshot), \
             patch.object(qo, "answer_env_question_with_metadata", side_effect=_fake_answer), \
             patch.object(qo, "stream_knowledge_tokens", side_effect=_fake_stream):
            qo.render_sync(branch)
            _drain(qo.render_stream(branch))

        self.assertEqual(seen, [snapshot, snapshot])

    def test_knowledge_prefetch_keeps_the_routers_scope(self):
        """The grounding pre-fetch must not re-decide what the question was about.

        It built planner_hints by hand and omitted `metric_scope`, so plan_metrics fell
        back to inferring the scope from question text — discarding a decision the router
        had already made correctly. A follow-up routed at scope `air_quality` arrived here
        with the scope dropped, was re-inferred as `named` over the router's metrics
        (["ieq", "co2"]), and `named` expands IEQ to ALL FOUR sub-indices: an air question
        came back leading with the lighting score.
        """
        route = _route(
            intent=IntentType.DEFINITION_EXPLANATION,
            metrics=["ieq", "co2"],
            metric_scope="air_quality",
            analysis_mode="advisory",
        )
        with patch.object(qo, "prepare_db_query", return_value={"rows": [], "payload": {}}) as prep:
            qo._fetch_live_sensor_data("what should I do next?", "smart_lab", route, "researcher")
        hints = prep.call_args.kwargs["planner_hints"]
        self.assertEqual(hints["metric_scope"], "air_quality")
        self.assertEqual(hints["analysis_mode"], "advisory")
        self.assertEqual(prep.call_args.kwargs["role"], "researcher")

    def test_db_sync_reports_resolved_timescale_and_metrics(self):
        route = _route(intent=IntentType.AGGREGATION_DB)
        branch = qo.plan_branch(_ctx(), route)
        with patch.object(qo, "run_db_query", return_value={
            "answer": "Average CO2 was 430 ppm.",
            "timescale": "1week",
            "metrics_used": ["co2"],
            "llm_used": True,
            "indexed_sources": [{"index": 1}],
            "footnotes": [{"index": 1}],
        }):
            result = qo.render_sync(branch)

        self.assertEqual(result["timescale"], "1week")
        self.assertEqual(result["metadata"]["ui"]["primary_metric"], "co2")
        self.assertEqual(result["citation_sources"], [{"index": 1}])
        self.assertEqual(result["footnotes"], [{"index": 1}])

    def test_llm_failure_is_reported_not_hidden(self):
        route = _route(intent=IntentType.AGGREGATION_DB)
        branch = qo.plan_branch(_ctx(), route)
        with patch.object(qo, "run_db_query", return_value={
            "answer": "Average CO2 was 430 ppm.", "timescale": "1hour", "llm_used": False,
        }):
            result = qo.render_sync(branch)
        self.assertFalse(result["metadata"]["llm_used"])


class BranchCoverageTests(unittest.TestCase):
    def test_every_intent_plans_a_branch(self):
        # A new intent that nobody wired up must not fall through to a crash.
        for intent in IntentType:
            with self.subTest(intent=intent.value):
                route = _route(intent=intent, viewer_type="splat", download_metric="co2")
                branch = qo.plan_branch(_ctx(), route)
                self.assertTrue(branch.name)
                self.assertTrue(branch.answer is not None or branch.run_sync is not None)
                self.assertTrue(branch.answer is not None or branch.open_stream is not None)


if __name__ == "__main__":
    unittest.main()
