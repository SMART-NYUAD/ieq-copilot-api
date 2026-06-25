"""Tests for the sensor_inspection intent: routing, staleness, ranking, executor, stream."""

import asyncio
import json
import os
import sys
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

TEST_DIR = os.path.dirname(__file__)
SERVER_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

from query_routing.intent_classifier import IntentType
from query_routing.router_types import RoutePlan, RouteExecutor
from query_routing.llm_router_planner import _parse_llm_response, _fallback_plan
from query_routing.query_orchestrator import _choose_executor, _execute_sensor_inspection
from executors import sensor_inspection_executor as sie


# A frozen "now" close to the fresh readings in the fixture (21:30 +04:00 == 17:30 UTC).
_NOW = datetime(2026, 6, 25, 17, 30, tzinfo=timezone.utc)


def _fixture_devices():
    """The per-device payload shape returned by /spaces/{slug}/heatmap/metrics."""
    return [
        {"device_id": 2, "device_name": "all_sensor_02", "device_alias": "SMART ESP32 Sensor 02",
         "metrics": [
             {"type": "temperature", "metric_name": "all_sensor_02_temperature", "unit": "C",
              "latest_value": 26.895, "latest_timestamp": "2026-06-25T21:17:25+04:00"},
             {"type": "pm25", "metric_name": "all_sensor_02_pm_2_5", "unit": "ug/m3",
              "latest_value": 15.4, "latest_timestamp": "2026-06-25T21:17:25+04:00"},
             {"type": "humidity", "metric_name": "all_sensor_02_humidity", "unit": "%",
              "latest_value": 50.869999, "latest_timestamp": "2026-06-25T21:17:25+04:00"},
             # Stale: last reported a year ago.
             {"type": "voc", "metric_name": "all_sensor_02_voc", "unit": "ppb",
              "latest_value": 101, "latest_timestamp": "2025-06-17T16:19:52+04:00"},
         ]},
        # Whole device offline: last reported Nov 2025.
        {"device_id": 23, "device_name": "all_sensor_67", "device_alias": "SMART ESP32 Sensor 08",
         "metrics": [
             {"type": "temperature", "metric_name": "all_sensor_67_temperature", "unit": "C",
              "latest_value": 24.682148, "latest_timestamp": "2025-11-14T03:05:26+04:00"},
             {"type": "humidity", "metric_name": "all_sensor_67_humidity", "unit": "%",
              "latest_value": 49.644531, "latest_timestamp": "2025-11-14T03:05:26+04:00"},
         ]},
        {"device_id": 24, "device_name": "all_sensor_68", "device_alias": "SMART ESP32 Sensor 06",
         "metrics": [
             {"type": "temperature", "metric_name": "all_sensor_68_temperature", "unit": "C",
              "latest_value": 27.318281, "latest_timestamp": "2026-06-25T21:11:23+04:00"},
             {"type": "humidity", "metric_name": "all_sensor_68_humidity", "unit": "%",
              "latest_value": 45.727539, "latest_timestamp": "2026-06-25T21:11:23+04:00"},
         ]},
        {"device_id": 27, "device_name": "all_sensor_71", "device_alias": "SMART ESP32 Sensor 11",
         "metrics": [
             {"type": "temperature", "metric_name": "all_sensor_71_temperature", "unit": "C",
              "latest_value": 27.139999, "latest_timestamp": "2026-06-25T21:17:36+04:00"},
             {"type": "humidity", "metric_name": "all_sensor_71_humidity", "unit": "%",
              "latest_value": 50.400002, "latest_timestamp": "2026-06-25T21:17:36+04:00"},
         ]},
        {"device_id": 29, "device_name": "all_sensor_73", "device_alias": "Atmocube Sensor 02",
         "metrics": [
             {"type": "temperature", "metric_name": "all_sensor_73_temperature", "unit": "C",
              "latest_value": 24.36, "latest_timestamp": "2026-06-25T21:17:18+04:00"},
             {"type": "humidity", "metric_name": "all_sensor_73_humidity", "unit": "%",
              "latest_value": 57.78, "latest_timestamp": "2026-06-25T21:17:18+04:00"},
         ]},
    ]


def _facts():
    return sie._build_device_facts(_fixture_devices(), _NOW, threshold_hours=24)


def _make_route(confidence=0.95):
    return RoutePlan(
        intent=IntentType.SENSOR_INSPECTION,
        confidence=confidence,
        lab_name=None,
        time_phrase=None,
        model="test",
        fallback_used=False,
    )


class TestRouting(unittest.TestCase):
    def test_parse_llm_response_accepts_sensor_intent(self):
        raw = json.dumps({"intent": "sensor_inspection", "lab": None, "second_lab": None,
                          "metrics": ["temperature"], "time_phrase": None, "confidence": 0.9})
        plan = _parse_llm_response(raw, "which sensor has the highest temperature?", None)
        self.assertIsNotNone(plan)
        self.assertEqual(plan.intent, IntentType.SENSOR_INSPECTION)

    def test_choose_executor_routes_sensor(self):
        self.assertEqual(_choose_executor(_make_route()), RouteExecutor.SENSOR_INSPECTION)

    def test_emergency_fallback_routes_ranking_question(self):
        plan = _fallback_plan("which sensor has the highest temperature?", None)
        self.assertEqual(plan.intent, IntentType.SENSOR_INSPECTION)

    def test_emergency_fallback_routes_health_question(self):
        plan = _fallback_plan("are any sensors offline?", None)
        self.assertEqual(plan.intent, IntentType.SENSOR_INSPECTION)

    def test_emergency_fallback_ignores_plain_status_question(self):
        # No device noun + signal pairing → must not hijack ordinary status questions.
        plan = _fallback_plan("what is the temperature?", None)
        self.assertNotEqual(plan.intent, IntentType.SENSOR_INSPECTION)


class TestStaleDetection(unittest.TestCase):
    def _stale_pairs(self):
        return {
            (d["alias"], m["type"])
            for d in _facts() for m in d["metrics"] if m["stale"]
        }

    def test_stale_readings_flagged(self):
        stale = self._stale_pairs()
        self.assertIn(("SMART ESP32 Sensor 08", "temperature"), stale)
        self.assertIn(("SMART ESP32 Sensor 08", "humidity"), stale)
        self.assertIn(("SMART ESP32 Sensor 02", "voc"), stale)

    def test_fresh_readings_not_flagged(self):
        stale = self._stale_pairs()
        self.assertNotIn(("SMART ESP32 Sensor 06", "temperature"), stale)
        self.assertNotIn(("SMART ESP32 Sensor 02", "temperature"), stale)


class TestDeterministicFallback(unittest.TestCase):
    def test_highest_temperature(self):
        ans = sie._deterministic_fallback("which sensor has the highest temperature?", _facts(), 24)
        self.assertIn("Sensor 06", ans)  # all_sensor_68 at 27.32 C
        self.assertIn("highest", ans.lower())

    def test_lowest_humidity(self):
        ans = sie._deterministic_fallback("which sensor reads the lowest humidity?", _facts(), 24)
        self.assertIn("Sensor 06", ans)  # all_sensor_68 at 45.73 %
        self.assertIn("lowest", ans.lower())

    def test_offline_sensors_listed(self):
        ans = sie._deterministic_fallback("which sensors are offline?", _facts(), 24)
        self.assertIn("Sensor 08", ans)   # all_sensor_67 (whole device stale)
        self.assertIn("voc", ans.lower())  # all_sensor_02 voc stale

    def test_no_stale_reports_all_healthy(self):
        # Threshold large enough that nothing is stale (must match the facts build).
        big = 24 * 400
        facts = sie._build_device_facts(_fixture_devices(), _NOW, threshold_hours=big)
        ans = sie._deterministic_fallback("any faulty sensors?", facts, threshold_hours=big)
        self.assertIn("none", ans.lower())

    def test_empty_devices(self):
        ans = sie._deterministic_fallback("highest temperature?", [], 24)
        self.assertIn("no sensors", ans.lower())


class TestExecuteSensorInspection(unittest.TestCase):
    def test_executor_uses_deterministic_fallback_when_llm_down(self):
        with patch.object(sie.api_client, "fetch_heatmap_metrics", return_value=_fixture_devices()), \
             patch.object(sie, "sensor_stale_hours", return_value=24), \
             patch("executors.sensor_inspection_executor.httpx.Client", side_effect=RuntimeError("ollama down")):
            result = _execute_sensor_inspection("which sensor has the highest temperature?", None, _make_route())
        self.assertEqual(result["metadata"]["executor"], "sensor_inspection")
        self.assertEqual(result["timescale"], "sensors")
        self.assertFalse(result["metadata"]["llm_used"])
        self.assertTrue(result["answer"])
        self.assertEqual(result["metadata"]["ui"]["panel"], "sensors")
        self.assertEqual(result["data"], None)

    def test_executor_graceful_when_no_devices(self):
        with patch.object(sie.api_client, "fetch_heatmap_metrics", return_value=[]):
            result = _execute_sensor_inspection("are any sensors offline?", None, _make_route())
        self.assertFalse(result["metadata"]["llm_used"])
        self.assertTrue(result["answer"])


class TestStream(unittest.TestCase):
    def test_stream_emits_meta_token_done(self):
        from query_routing.query_orchestrator import stream_query
        from storage.conversation_context import ConversationContext

        ctx = ConversationContext(
            original_question="which sensors are offline?",
            effective_question="which sensors are offline?",
            effective_lab=None,
            routing_snippet="",
            llm_history="",
            carried_metric=None,
            carried_time_phrase=None,
            conversation_id="test-conv",
            raw_block="",
        )
        route = _make_route()

        async def _fake_plan(*a, **kw):
            return route

        async def _fake_tokens(user_question, space=None):
            yield f"data: {json.dumps({'event': 'token', 'text': 'Sensor 08 is offline.'})}\n\n"
            yield f"data: {json.dumps({'event': 'done'})}\n\n"

        async def _run():
            chunks = []
            async for chunk in stream_query(ctx, k=5):
                chunks.append(chunk)
            return chunks

        with patch("query_routing.query_orchestrator.plan_route_async", side_effect=_fake_plan), \
             patch("query_routing.query_orchestrator.stream_sensor_tokens", side_effect=_fake_tokens):
            chunks = asyncio.new_event_loop().run_until_complete(_run())

        events = [json.loads(c.removeprefix("data: ").strip()) for c in chunks if c.strip()]
        types = [e.get("event") for e in events]
        self.assertIn("meta", types)
        self.assertIn("token", types)
        self.assertIn("done", types)
        meta = next(e for e in events if e.get("event") == "meta")
        self.assertEqual(meta["executor"], "sensor_inspection")


if __name__ == "__main__":
    unittest.main()
