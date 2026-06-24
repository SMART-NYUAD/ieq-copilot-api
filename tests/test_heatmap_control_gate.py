"""Tests for the heatmap_control gate: intent classification, route plan, and stream/sync output."""

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

from query_routing.intent_classifier import IntentType
from query_routing.router_types import RoutePlan, RouteExecutor
from query_routing.llm_router_planner import (
    _infer_heatmap_action,
    _infer_heatmap_metric,
    _parse_llm_response,
)
from query_routing.query_orchestrator import (
    _choose_executor,
    _execute_heatmap_control,
)


def _make_route(action="on", metric=None, confidence=0.95, model="test"):
    return RoutePlan(
        intent=IntentType.HEATMAP_CONTROL,
        confidence=confidence,
        lab_name=None,
        time_phrase=None,
        model=model,
        fallback_used=False,
        heatmap_action=action,
        heatmap_metric=metric,
    )


class TestInferHeatmap(unittest.TestCase):
    def test_metric_aliases(self):
        self.assertEqual(_infer_heatmap_metric("color the model by temperature"), "temperature")
        self.assertEqual(_infer_heatmap_metric("use the humidity overlay"), "humidity")
        self.assertEqual(_infer_heatmap_metric("switch to voc"), "voc")
        self.assertEqual(_infer_heatmap_metric("show the pm2.5 heatmap"), "pm25")
        self.assertEqual(_infer_heatmap_metric("show the pm 2.5 heatmap"), "pm25")

    def test_metric_none_when_unsupported_or_absent(self):
        self.assertIsNone(_infer_heatmap_metric("turn on the heatmap"))
        # co2 is not a supported heatmap metric per product decision
        self.assertIsNone(_infer_heatmap_metric("color the model by co2"))

    def test_action_defaults_to_on(self):
        self.assertEqual(_infer_heatmap_action("turn on the heatmap"), "on")
        self.assertEqual(_infer_heatmap_action("color the model by temperature"), "on")

    def test_action_off_hints(self):
        self.assertEqual(_infer_heatmap_action("turn off the heatmap"), "off")
        self.assertEqual(_infer_heatmap_action("hide the heatmap"), "off")
        self.assertEqual(_infer_heatmap_action("disable the overlay"), "off")


class TestParseHeatmap(unittest.TestCase):
    def _build_raw(self, intent, action=None, metric=None, confidence=0.9):
        payload = {"intent": intent, "lab": None, "second_lab": None,
                   "metrics": [], "time_phrase": None, "confidence": confidence}
        if action is not None:
            payload["heatmap_action"] = action
        if metric is not None:
            payload["heatmap_metric"] = metric
        return json.dumps(payload)

    def test_explicit_action_and_metric(self):
        raw = self._build_raw("heatmap_control", action="on", metric="temperature")
        plan = _parse_llm_response(raw, "turn on the heatmap and use temperature", None)
        self.assertIsNotNone(plan)
        self.assertEqual(plan.intent, IntentType.HEATMAP_CONTROL)
        self.assertEqual(plan.heatmap_action, "on")
        self.assertEqual(plan.heatmap_metric, "temperature")

    def test_off_action(self):
        raw = self._build_raw("heatmap_control", action="off")
        plan = _parse_llm_response(raw, "turn off the heatmap", None)
        self.assertEqual(plan.heatmap_action, "off")
        self.assertIsNone(plan.heatmap_metric)

    def test_pm25_metric_canonicalized(self):
        raw = self._build_raw("heatmap_control", action="on", metric="pm2.5")
        plan = _parse_llm_response(raw, "show the pm2.5 heatmap", None)
        self.assertEqual(plan.heatmap_metric, "pm25")

    def test_invalid_metric_falls_back_to_inference(self):
        # LLM names an unsupported metric, but the question text mentions humidity.
        raw = self._build_raw("heatmap_control", action="on", metric="co2")
        plan = _parse_llm_response(raw, "color the model by humidity", None)
        self.assertEqual(plan.heatmap_metric, "humidity")

    def test_missing_fields_fall_back_to_inference(self):
        raw = self._build_raw("heatmap_control")
        plan = _parse_llm_response(raw, "turn on the heatmap and use the metric temperature", None)
        self.assertEqual(plan.heatmap_action, "on")
        self.assertEqual(plan.heatmap_metric, "temperature")

    def test_non_heatmap_intent_has_no_heatmap_fields(self):
        raw = self._build_raw("current_status_db")
        plan = _parse_llm_response(raw, "what is the co2?", None)
        self.assertIsNone(plan.heatmap_action)
        self.assertIsNone(plan.heatmap_metric)


class TestChooseExecutor(unittest.TestCase):
    def test_heatmap_control_routes_to_heatmap_executor(self):
        self.assertEqual(_choose_executor(_make_route()), RouteExecutor.HEATMAP_CONTROL)


class TestExecuteHeatmapControl(unittest.TestCase):
    def test_on_without_metric(self):
        result = _execute_heatmap_control(_make_route("on", None))
        self.assertEqual(result["metadata"]["executor"], "heatmap_control")
        self.assertEqual(result["metadata"]["ui"], {"heatmap_action": "on", "heatmap_metric": None})
        self.assertEqual(result["timescale"], "instant")
        self.assertFalse(result["metadata"]["llm_used"])
        self.assertIn("heatmap", result["answer"].lower())

    def test_on_with_metric_label(self):
        result = _execute_heatmap_control(_make_route("on", "pm25"))
        self.assertEqual(result["metadata"]["ui"]["heatmap_metric"], "pm25")
        self.assertIn("PM2.5", result["answer"])

    def test_off(self):
        result = _execute_heatmap_control(_make_route("off", None))
        self.assertEqual(result["metadata"]["ui"]["heatmap_action"], "off")
        self.assertIn("off", result["answer"].lower())

    def test_missing_action_defaults_to_on(self):
        route = RoutePlan(intent=IntentType.HEATMAP_CONTROL, confidence=0.9,
                          lab_name=None, time_phrase=None, model="test",
                          heatmap_action=None, heatmap_metric=None)
        result = _execute_heatmap_control(route)
        self.assertEqual(result["metadata"]["ui"]["heatmap_action"], "on")


class TestStreamHeatmapControl(unittest.TestCase):
    def _collect_stream(self, route):
        from query_routing.query_orchestrator import stream_query
        from storage.conversation_context import ConversationContext

        ctx = ConversationContext(
            original_question="turn on the heatmap",
            effective_question="turn on the heatmap",
            effective_lab=None,
            routing_snippet="",
            llm_history="",
            carried_metric=None,
            carried_time_phrase=None,
            conversation_id="test-conv",
            raw_block="",
        )

        async def _run():
            chunks = []
            async for chunk in stream_query(ctx, k=5):
                chunks.append(chunk)
            return chunks

        with patch("query_routing.query_orchestrator.plan_route_async") as mock_plan:
            async def _fake_plan(*a, **kw):
                return route

            mock_plan.side_effect = _fake_plan
            chunks = asyncio.get_event_loop().run_until_complete(_run())
        return chunks

    def _parse_events(self, chunks):
        events = []
        for chunk in chunks:
            raw = chunk.removeprefix("data: ").strip()
            if raw:
                events.append(json.loads(raw))
        return events

    def test_stream_emits_meta_token_done_in_order(self):
        events = self._parse_events(self._collect_stream(_make_route("on", "temperature")))
        event_types = [e.get("event") for e in events]
        self.assertIn("meta", event_types)
        self.assertIn("token", event_types)
        self.assertIn("done", event_types)
        self.assertLess(event_types.index("meta"), event_types.index("token"))
        self.assertLess(event_types.index("token"), event_types.index("done"))

    def test_stream_meta_contains_heatmap_ui(self):
        events = self._parse_events(self._collect_stream(_make_route("on", "humidity")))
        meta_events = [e for e in events if e.get("event") == "meta"]
        self.assertEqual(len(meta_events), 1)
        ui = meta_events[0].get("ui", {})
        self.assertEqual(ui.get("heatmap_action"), "on")
        self.assertEqual(ui.get("heatmap_metric"), "humidity")
        self.assertEqual(meta_events[0].get("executor"), "heatmap_control")

    def test_stream_token_contains_confirmation(self):
        events = self._parse_events(self._collect_stream(_make_route("off", None)))
        token_events = [e for e in events if e.get("event") == "token"]
        self.assertEqual(len(token_events), 1)
        self.assertIn("off", token_events[0].get("text", "").lower())


if __name__ == "__main__":
    unittest.main()
