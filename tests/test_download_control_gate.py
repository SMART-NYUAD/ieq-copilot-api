"""Tests for the download_data gate: intent parsing, URL building, and stream/sync output."""

import asyncio
import json
import os
import sys
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

TEST_DIR = os.path.dirname(__file__)
SERVER_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

from query_routing.intent_classifier import IntentType
from query_routing.router_types import RoutePlan, RouteExecutor
from query_routing.llm_router_planner import (
    _infer_download_format,
    _infer_download_type,
    _parse_llm_response,
)
from query_routing.query_orchestrator import (
    _build_download,
    _choose_executor,
    _execute_download_control,
)


def _make_route(fmt="csv", dtype="aggregated", confidence=0.95, model="test"):
    return RoutePlan(
        intent=IntentType.DOWNLOAD_DATA,
        confidence=confidence,
        lab_name=None,
        time_phrase=None,
        model=model,
        fallback_used=False,
        download_format=fmt,
        download_type=dtype,
    )


class TestInferDownload(unittest.TestCase):
    def test_format_defaults_to_csv(self):
        self.assertEqual(_infer_download_format("download the data"), "csv")
        self.assertEqual(_infer_download_format("export last 7 days"), "csv")

    def test_format_json_when_named(self):
        self.assertEqual(_infer_download_format("export the readings as JSON"), "json")

    def test_type_defaults_to_aggregated(self):
        self.assertEqual(_infer_download_type("download the data"), "aggregated")

    def test_type_raw_when_named(self):
        self.assertEqual(_infer_download_type("download the raw readings"), "raw")
        self.assertEqual(_infer_download_type("give me the unaggregated data"), "raw")


class TestParseDownload(unittest.TestCase):
    def _build_raw(self, intent, fmt=None, dtype=None, time_phrase=None, confidence=0.9):
        payload = {"intent": intent, "lab": None, "second_lab": None,
                   "metrics": [], "time_phrase": time_phrase, "confidence": confidence}
        if fmt is not None:
            payload["download_format"] = fmt
        if dtype is not None:
            payload["download_type"] = dtype
        return json.dumps(payload)

    def test_explicit_format_and_type(self):
        raw = self._build_raw("download_data", fmt="JSON", dtype="raw")
        plan = _parse_llm_response(raw, "export raw readings as json", None)
        self.assertIsNotNone(plan)
        self.assertEqual(plan.intent, IntentType.DOWNLOAD_DATA)
        self.assertEqual(plan.download_format, "json")
        self.assertEqual(plan.download_type, "raw")

    def test_defaults_when_fields_missing(self):
        raw = self._build_raw("download_data")
        plan = _parse_llm_response(raw, "download the data", None)
        self.assertEqual(plan.download_format, "csv")
        self.assertEqual(plan.download_type, "aggregated")

    def test_invalid_format_falls_back_to_inference(self):
        raw = self._build_raw("download_data", fmt="pdf")
        plan = _parse_llm_response(raw, "export the readings as json", None)
        self.assertEqual(plan.download_format, "json")

    def test_non_download_intent_has_no_download_fields(self):
        raw = self._build_raw("current_status_db")
        plan = _parse_llm_response(raw, "what is the co2?", None)
        self.assertIsNone(plan.download_format)
        self.assertIsNone(plan.download_type)


class TestBuildDownloadUrl(unittest.TestCase):
    def test_url_carries_required_params(self):
        dl = _build_download(_make_route("csv", "aggregated"), "export last 7 days")
        parsed = urlparse(dl["url"])
        qs = parse_qs(parsed.query)
        self.assertEqual(parsed.path, "/download/sensor-readings")
        self.assertEqual(qs["sensor_alias"], ["Atmocube Sensor 02"])
        self.assertEqual(qs["format"], ["csv"])
        self.assertEqual(qs["type"], ["aggregated"])
        self.assertIn("start", qs)
        self.assertIn("end", qs)

    def test_explicit_window_is_narrower_than_default(self):
        narrow = _build_download(_make_route(), "download the last 7 days")
        wide = _build_download(_make_route(), "download the data")
        # No explicit window → wide (one-year) default starts earlier than a 7-day window.
        self.assertLess(wide["start"], narrow["start"])

    def test_json_raw_propagate_to_url(self):
        dl = _build_download(_make_route("json", "raw"), "export raw data as json")
        qs = parse_qs(urlparse(dl["url"]).query)
        self.assertEqual(qs["format"], ["json"])
        self.assertEqual(qs["type"], ["raw"])


class TestChooseExecutor(unittest.TestCase):
    def test_download_routes_to_download_executor(self):
        self.assertEqual(_choose_executor(_make_route()), RouteExecutor.DOWNLOAD_DATA)


class TestExecuteDownloadControl(unittest.TestCase):
    def test_metadata_and_ui(self):
        result = _execute_download_control(_make_route("csv", "aggregated"), "download last 7 days")
        meta = result["metadata"]
        self.assertEqual(meta["executor"], "download_data")
        self.assertFalse(meta["llm_used"])
        self.assertEqual(result["timescale"], "instant")
        ui = meta["ui"]
        self.assertTrue(ui["download_url"].startswith("https://"))
        self.assertEqual(ui["download_format"], "csv")
        self.assertEqual(ui["download_type"], "aggregated")
        self.assertIn("T", ui["download_start"])  # ISO-8601 datetime
        self.assertIn("download", result["answer"].lower())


class TestStreamDownloadControl(unittest.TestCase):
    def _collect_stream(self, route):
        from query_routing.query_orchestrator import stream_query
        from storage.conversation_context import ConversationContext

        ctx = ConversationContext(
            original_question="download the data",
            effective_question="download the data",
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
        events = self._parse_events(self._collect_stream(_make_route()))
        event_types = [e.get("event") for e in events]
        self.assertIn("meta", event_types)
        self.assertIn("token", event_types)
        self.assertIn("done", event_types)
        self.assertLess(event_types.index("meta"), event_types.index("token"))
        self.assertLess(event_types.index("token"), event_types.index("done"))

    def test_stream_meta_contains_download_url(self):
        events = self._parse_events(self._collect_stream(_make_route()))
        meta_events = [e for e in events if e.get("event") == "meta"]
        self.assertEqual(len(meta_events), 1)
        ui = meta_events[0].get("ui", {})
        self.assertTrue(ui.get("download_url", "").startswith("https://"))
        self.assertEqual(meta_events[0].get("executor"), "download_data")


if __name__ == "__main__":
    unittest.main()
