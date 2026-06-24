"""Tests for the viewer_control gate: intent classification, route plan, and stream/sync output."""

import asyncio
import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

TEST_DIR = os.path.dirname(__file__)
SERVER_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

from query_routing.intent_classifier import IntentType
from query_routing.router_types import RoutePlan, RouteExecutor
from query_routing.llm_router_planner import _infer_viewer_type, _parse_llm_response
from query_routing.query_orchestrator import (
    _choose_executor,
    _execute_unknown_fallback,
    _execute_viewer_control,
    _VIEWER_CONFIRMATIONS,
)


def _make_route(viewer_type="splat", confidence=0.95, model="test"):
    return RoutePlan(
        intent=IntentType.VIEWER_CONTROL,
        confidence=confidence,
        lab_name=None,
        time_phrase=None,
        model=model,
        fallback_used=False,
        viewer_type=viewer_type,
    )


class TestInferViewerType(unittest.TestCase):
    def test_splat_keywords(self):
        self.assertEqual(_infer_viewer_type("show me the splat"), "splat")
        self.assertEqual(_infer_viewer_type("open the gaussian splat"), "splat")
        self.assertEqual(_infer_viewer_type("switch to gaussian view"), "splat")

    def test_ifc_keywords(self):
        self.assertEqual(_infer_viewer_type("switch to IFC"), "ifc")
        self.assertEqual(_infer_viewer_type("open the floor plan"), "ifc")
        self.assertEqual(_infer_viewer_type("show me the floorplan"), "ifc")
        self.assertEqual(_infer_viewer_type("open bim model"), "ifc")

    def test_pc_keywords(self):
        self.assertEqual(_infer_viewer_type("open the point cloud"), "pc")
        self.assertEqual(_infer_viewer_type("switch to point cloud mode"), "pc")

    def test_pano_keywords(self):
        self.assertEqual(_infer_viewer_type("show me the panorama"), "pano")
        self.assertEqual(_infer_viewer_type("open pano view"), "pano")

    def test_unknown_defaults_to_splat(self):
        self.assertEqual(_infer_viewer_type("open the 3D viewer please"), "splat")


class TestParseLlmResponse(unittest.TestCase):
    def _build_raw(self, intent, viewer_type=None, confidence=0.9):
        payload = {"intent": intent, "lab": None, "second_lab": None,
                   "metrics": [], "time_phrase": None, "confidence": confidence}
        if viewer_type is not None:
            payload["viewer_type"] = viewer_type
        return json.dumps(payload)

    def test_viewer_control_valid_viewer_type(self):
        raw = self._build_raw("viewer_control", viewer_type="pano")
        plan = _parse_llm_response(raw, "show me the panorama", None)
        self.assertIsNotNone(plan)
        self.assertEqual(plan.intent, IntentType.VIEWER_CONTROL)
        self.assertEqual(plan.viewer_type, "pano")

    def test_viewer_control_invalid_viewer_type_falls_back_to_inference(self):
        raw = self._build_raw("viewer_control", viewer_type="unknown_thing")
        plan = _parse_llm_response(raw, "open the point cloud", None)
        self.assertIsNotNone(plan)
        self.assertEqual(plan.intent, IntentType.VIEWER_CONTROL)
        self.assertEqual(plan.viewer_type, "pc")

    def test_viewer_control_missing_viewer_type_falls_back_to_inference(self):
        raw = self._build_raw("viewer_control")
        plan = _parse_llm_response(raw, "switch to ifc", None)
        self.assertIsNotNone(plan)
        self.assertEqual(plan.viewer_type, "ifc")

    def test_non_viewer_intent_has_no_viewer_type(self):
        raw = self._build_raw("current_status_db")
        plan = _parse_llm_response(raw, "what is the co2?", None)
        self.assertIsNotNone(plan)
        self.assertIsNone(plan.viewer_type)

    def test_unknown_fallback_intent_is_accepted(self):
        raw = self._build_raw("unknown_fallback", confidence=0.95)
        plan = _parse_llm_response(raw, "who won the football match?", None)
        self.assertIsNotNone(plan)
        self.assertEqual(plan.intent, IntentType.UNKNOWN_FALLBACK)
        self.assertEqual(plan.confidence, 0.95)

    def test_all_valid_viewer_types_accepted(self):
        for vt in ("splat", "ifc", "pc", "pano"):
            raw = self._build_raw("viewer_control", viewer_type=vt)
            plan = _parse_llm_response(raw, "switch viewer", None)
            self.assertEqual(plan.viewer_type, vt)


class TestChooseExecutor(unittest.TestCase):
    def test_viewer_control_routes_to_viewer_executor(self):
        route = _make_route()
        self.assertEqual(_choose_executor(route), RouteExecutor.VIEWER_CONTROL)

    def test_knowledge_intent_routes_to_knowledge_executor(self):
        route = RoutePlan(intent=IntentType.DEFINITION_EXPLANATION, confidence=0.9,
                          lab_name=None, time_phrase=None, model="test")
        self.assertEqual(_choose_executor(route), RouteExecutor.KNOWLEDGE_QA)

    def test_db_intent_routes_to_db_executor(self):
        route = RoutePlan(intent=IntentType.CURRENT_STATUS_DB, confidence=0.9,
                          lab_name=None, time_phrase=None, model="test")
        self.assertEqual(_choose_executor(route), RouteExecutor.DB_QUERY)

    def test_unknown_fallback_uses_knowledge_executor_bucket(self):
        route = RoutePlan(intent=IntentType.UNKNOWN_FALLBACK, confidence=0.9,
                          lab_name=None, time_phrase=None, model="test")
        self.assertEqual(_choose_executor(route), RouteExecutor.KNOWLEDGE_QA)


class TestExecuteViewerControl(unittest.TestCase):
    def test_result_structure_for_splat(self):
        result = _execute_viewer_control(_make_route("splat"))
        self.assertEqual(result["metadata"]["ui"]["viewer_type"], "splat")
        self.assertEqual(result["metadata"]["executor"], "viewer_control")
        self.assertEqual(result["timescale"], "instant")
        self.assertIn("Gaussian Splat", result["answer"])
        self.assertEqual(result["cards_retrieved"], 0)
        self.assertFalse(result["metadata"]["llm_used"])

    def test_result_structure_for_ifc(self):
        result = _execute_viewer_control(_make_route("ifc"))
        self.assertEqual(result["metadata"]["ui"]["viewer_type"], "ifc")
        self.assertIn("IFC", result["answer"])

    def test_result_structure_for_pc(self):
        result = _execute_viewer_control(_make_route("pc"))
        self.assertEqual(result["metadata"]["ui"]["viewer_type"], "pc")
        self.assertIn("Point Cloud", result["answer"])

    def test_result_structure_for_pano(self):
        result = _execute_viewer_control(_make_route("pano"))
        self.assertEqual(result["metadata"]["ui"]["viewer_type"], "pano")
        self.assertIn("Panorama", result["answer"])

    def test_missing_viewer_type_defaults_to_splat(self):
        route = RoutePlan(intent=IntentType.VIEWER_CONTROL, confidence=0.9,
                          lab_name=None, time_phrase=None, model="test", viewer_type=None)
        result = _execute_viewer_control(route)
        self.assertEqual(result["metadata"]["ui"]["viewer_type"], "splat")

    def test_unknown_fallback_response_is_concise_guardrail(self):
        route = RoutePlan(intent=IntentType.UNKNOWN_FALLBACK, confidence=0.95,
                          lab_name=None, time_phrase=None, model="test")
        result = _execute_unknown_fallback(route)
        self.assertEqual(result["metadata"]["executor"], "guardrail")
        self.assertFalse(result["metadata"]["llm_used"])
        self.assertLessEqual(len(result["answer"].split()), 25)
        self.assertIn("indoor environmental quality", result["answer"])


class TestStreamViewerControl(unittest.TestCase):
    def _collect_stream(self, route):
        from query_routing.query_orchestrator import stream_query
        from storage.conversation_context import ConversationContext

        ctx = ConversationContext(
            original_question="show me the splat",
            effective_question="show me the splat",
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
            mock_plan.return_value = asyncio.coroutine(lambda: route)() if False else route

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
        route = _make_route("pc")
        chunks = self._collect_stream(route)
        events = self._parse_events(chunks)
        event_types = [e.get("event") for e in events]
        self.assertIn("meta", event_types)
        self.assertIn("token", event_types)
        self.assertIn("done", event_types)
        meta_idx = event_types.index("meta")
        token_idx = event_types.index("token")
        done_idx = event_types.index("done")
        self.assertLess(meta_idx, token_idx)
        self.assertLess(token_idx, done_idx)

    def test_stream_meta_contains_viewer_type_in_ui(self):
        route = _make_route("pano")
        chunks = self._collect_stream(route)
        events = self._parse_events(chunks)
        meta_events = [e for e in events if e.get("event") == "meta"]
        self.assertEqual(len(meta_events), 1)
        ui = meta_events[0].get("ui", {})
        self.assertEqual(ui.get("viewer_type"), "pano")

    def test_stream_token_contains_confirmation(self):
        route = _make_route("ifc")
        chunks = self._collect_stream(route)
        events = self._parse_events(chunks)
        token_events = [e for e in events if e.get("event") == "token"]
        self.assertEqual(len(token_events), 1)
        self.assertIn("IFC", token_events[0].get("text", ""))


if __name__ == "__main__":
    unittest.main()
