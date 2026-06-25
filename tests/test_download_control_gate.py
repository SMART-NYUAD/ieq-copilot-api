"""Tests for the download_data gate: intent parsing, parameter building, and stream/sync output.

The download path targets ``/spaces/{slug}/metrics/{metric_type}/download-agg-summary`` and hands
the frontend the discrete parameters (slug, metric_type, start, end, interval, format) rather than a
pre-built URL. A metric is required: when it is missing the gate asks a follow-up question instead.
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

from query_routing.intent_classifier import IntentType
from query_routing.router_types import RoutePlan, RouteExecutor
from query_routing.llm_router_planner import (
    _infer_download_format,
    _infer_download_metric,
    _infer_download_interval,
    _parse_llm_response,
)
from query_routing.query_orchestrator import (
    _build_download,
    _choose_executor,
    _execute_download_control,
)


def _make_route(fmt="csv", metric="pm25", interval=None, confidence=0.95, model="test", lab=None):
    return RoutePlan(
        intent=IntentType.DOWNLOAD_DATA,
        confidence=confidence,
        lab_name=lab,
        time_phrase=None,
        model=model,
        fallback_used=False,
        download_format=fmt,
        download_metric=metric,
        download_interval=interval,
    )


class TestInferDownload(unittest.TestCase):
    def test_format_defaults_to_csv(self):
        self.assertEqual(_infer_download_format("download the data"), "csv")
        self.assertEqual(_infer_download_format("export last 7 days"), "csv")

    def test_format_json_when_named(self):
        self.assertEqual(_infer_download_format("export the readings as JSON"), "json")

    def test_metric_inferred_from_question(self):
        self.assertEqual(_infer_download_metric("download the temperature data"), "temperature")
        self.assertEqual(_infer_download_metric("export pm2.5 readings"), "pm25")
        self.assertEqual(_infer_download_metric("save the co2 measurements"), "co2")

    def test_metric_none_when_absent(self):
        self.assertIsNone(_infer_download_metric("download the data"))

    def test_interval_inferred_from_granularity(self):
        self.assertEqual(_infer_download_interval("export hourly temperature"), "1h")
        self.assertEqual(_infer_download_interval("download daily humidity"), "1d")
        self.assertIsNone(_infer_download_interval("download the data"))


class TestParseDownload(unittest.TestCase):
    def _build_raw(self, intent, fmt=None, metric=None, interval=None, time_phrase=None, confidence=0.9):
        payload = {"intent": intent, "lab": None, "second_lab": None,
                   "metrics": [], "time_phrase": time_phrase, "confidence": confidence}
        if fmt is not None:
            payload["download_format"] = fmt
        if metric is not None:
            payload["download_metric"] = metric
        if interval is not None:
            payload["download_interval"] = interval
        return json.dumps(payload)

    def test_explicit_format_and_metric(self):
        raw = self._build_raw("download_data", fmt="JSON", metric="pm2.5", interval="1h")
        plan = _parse_llm_response(raw, "export pm2.5 readings as json hourly", None)
        self.assertIsNotNone(plan)
        self.assertEqual(plan.intent, IntentType.DOWNLOAD_DATA)
        self.assertEqual(plan.download_format, "json")
        self.assertEqual(plan.download_metric, "pm25")
        self.assertEqual(plan.download_interval, "1h")

    def test_defaults_when_fields_missing(self):
        raw = self._build_raw("download_data")
        plan = _parse_llm_response(raw, "download the data", None)
        self.assertEqual(plan.download_format, "csv")
        self.assertIsNone(plan.download_metric)
        self.assertIsNone(plan.download_interval)

    def test_metric_falls_back_to_inference(self):
        raw = self._build_raw("download_data")
        plan = _parse_llm_response(raw, "export the temperature readings", None)
        self.assertEqual(plan.download_metric, "temperature")

    def test_non_download_intent_has_no_download_fields(self):
        raw = self._build_raw("current_status_db")
        plan = _parse_llm_response(raw, "what is the co2?", None)
        self.assertIsNone(plan.download_format)
        self.assertIsNone(plan.download_metric)
        self.assertIsNone(plan.download_interval)


class TestBuildDownloadParams(unittest.TestCase):
    def test_params_carry_required_fields(self):
        dl = _build_download(_make_route(fmt="csv", metric="pm25"), "export last 7 days")
        self.assertEqual(dl["metric_type"], "pm2.5")
        self.assertEqual(dl["format"], "csv")
        self.assertEqual(dl["slug"], "smart_lab")
        self.assertIn("T", dl["start"])
        self.assertIn("T", dl["end"])
        self.assertTrue(dl["interval"])

    def test_interval_defaults_when_absent(self):
        dl = _build_download(_make_route(metric="temperature", interval=None), "download the data")
        self.assertEqual(dl["interval"], "1m")

    def test_lab_becomes_slug(self):
        dl = _build_download(_make_route(metric="co2", lab="Smart Lab"), "download co2")
        self.assertEqual(dl["slug"], "smart_lab")


class TestChooseExecutor(unittest.TestCase):
    def test_download_routes_to_download_executor(self):
        self.assertEqual(_choose_executor(_make_route()), RouteExecutor.DOWNLOAD_DATA)


class TestExecuteDownloadControl(unittest.TestCase):
    def test_metadata_and_ui(self):
        result = _execute_download_control(_make_route(fmt="csv", metric="pm25"), "download last 7 days")
        meta = result["metadata"]
        self.assertEqual(meta["executor"], "download_data")
        self.assertFalse(meta["llm_used"])
        self.assertEqual(result["timescale"], "instant")
        ui = meta["ui"]
        self.assertFalse(ui["download_needs_metric"])
        self.assertEqual(ui["download_slug"], "smart_lab")
        self.assertEqual(ui["download_metric_type"], "pm2.5")
        self.assertEqual(ui["download_format"], "csv")
        self.assertIn("T", ui["download_start"])
        self.assertIn("download_interval", ui)
        self.assertNotIn("download_url", ui)  # parameters only, never a pre-built URL
        self.assertIn("download", result["answer"].lower())

    def test_missing_metric_asks_followup(self):
        result = _execute_download_control(_make_route(metric=None), "download the data")
        ui = result["metadata"]["ui"]
        self.assertTrue(ui["download_needs_metric"])
        self.assertNotIn("download_metric_type", ui)
        self.assertIn("which metric", result["answer"].lower())


class TestStreamDownloadControl(unittest.TestCase):
    def _collect_stream(self, route):
        from query_routing.query_orchestrator import stream_query
        from storage.conversation_context import ConversationContext

        ctx = ConversationContext(
            original_question="download the data",
            effective_question="download the temperature data",
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
        events = self._parse_events(self._collect_stream(_make_route(metric="temperature")))
        event_types = [e.get("event") for e in events]
        self.assertIn("meta", event_types)
        self.assertIn("token", event_types)
        self.assertIn("done", event_types)
        self.assertLess(event_types.index("meta"), event_types.index("token"))
        self.assertLess(event_types.index("token"), event_types.index("done"))

    def test_stream_meta_contains_download_params(self):
        events = self._parse_events(self._collect_stream(_make_route(metric="temperature")))
        meta_events = [e for e in events if e.get("event") == "meta"]
        self.assertEqual(len(meta_events), 1)
        ui = meta_events[0].get("ui", {})
        self.assertEqual(ui.get("download_metric_type"), "temperature")
        self.assertNotIn("download_url", ui)
        self.assertEqual(meta_events[0].get("executor"), "download_data")

    def test_stream_missing_metric_asks_followup(self):
        events = self._parse_events(self._collect_stream(_make_route(metric=None)))
        meta_events = [e for e in events if e.get("event") == "meta"]
        self.assertEqual(len(meta_events), 1)
        self.assertTrue(meta_events[0].get("ui", {}).get("download_needs_metric"))


if __name__ == "__main__":
    unittest.main()
