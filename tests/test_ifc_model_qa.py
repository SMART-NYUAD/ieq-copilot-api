"""Tests for IFC-model Q&A: parsing, routing, executor wiring, and streaming."""

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
from query_routing.llm_router_planner import _parse_llm_response
from query_routing.query_orchestrator import _choose_executor, _ifc_branch, render_sync
from ifc_model.ifc_store import get_ifc_facts, get_ifc_summary, build_ifc_context_text

IFC_PATH = os.path.join(SERVER_DIR, "smart.ifc")
_HAS_MODEL = os.path.exists(IFC_PATH)


def _make_ifc_route(confidence=0.95):
    return RoutePlan(
        intent=IntentType.IFC_MODEL_QA,
        confidence=confidence,
        lab_name=None,
        time_phrase=None,
        model="test",
        fallback_used=False,
    )


@unittest.skipUnless(_HAS_MODEL, "smart.ifc not present")
class TestIfcParsing(unittest.TestCase):
    def test_summary_basic_structure(self):
        s = get_ifc_summary(IFC_PATH)
        self.assertEqual(s["schema"], "IFC2X3")
        self.assertGreater(s["total_elements"], 0)
        self.assertIsInstance(s["element_counts"], dict)
        self.assertIsInstance(s["materials"], list)

    def test_storeys_parsed_with_elevations(self):
        s = get_ifc_summary(IFC_PATH)
        names = {st["name"] for st in s["storeys"]}
        self.assertIn("B2", names)
        self.assertIn("Level 1", names)
        for st in s["storeys"]:
            self.assertIsNotNone(st["elevation"])

    def test_length_unit_is_millimetre(self):
        facts = get_ifc_facts(IFC_PATH)
        self.assertEqual(facts.length_unit, "mm")

    def test_element_counts_match_known_model(self):
        s = get_ifc_summary(IFC_PATH)
        counts = s["element_counts"]
        self.assertEqual(counts.get("column"), 6)
        self.assertEqual(counts.get("door"), 2)
        self.assertEqual(counts.get("wall"), 4)

    def test_context_text_grounded_and_compact(self):
        text = build_ifc_context_text(IFC_PATH)
        self.assertIn("Element Inventory", text)
        self.assertIn("Levels / Storeys", text)
        self.assertIn("Materials", text)
        # Door dimensions surfaced in mm, not metres.
        self.assertIn("mm", text)

    def test_overall_dimensions_computed_in_world_space(self):
        facts = get_ifc_facts(IFC_PATH)
        self.assertIsNotNone(facts.dimensions)
        d = facts.dimensions
        # Positive, physically plausible extents for an office-scale floor (in mm).
        self.assertGreater(d["length_x"], 1000)
        self.assertGreater(d["width_y"], 1000)
        self.assertGreater(d["height_z"], 0)
        # World-resolved bbox must be far smaller than the contaminated raw-point
        # span (~20 m × 32 m); footprint axes should be under 25 m.
        self.assertLess(d["length_x"], 25000)
        self.assertLess(d["width_y"], 25000)
        self.assertEqual(d["unit"], "mm")
        self.assertIn("Overall Model Dimensions", build_ifc_context_text(IFC_PATH))

    def test_architectural_metrics_gia_and_derived(self):
        facts = get_ifc_facts(IFC_PATH)
        m = facts.architectural_metrics
        self.assertIsNotNone(m)
        # GIA from the slab plate (~14.74 m × 10.11 m ≈ 149 m²).
        self.assertGreater(m["gross_internal_area_m2"], 100)
        self.assertLess(m["gross_internal_area_m2"], 200)
        self.assertEqual(m["number_of_storeys"], 2)
        self.assertEqual(m["floor_to_floor_height_m"], 2.5)
        self.assertIn(200.0, m["wall_thickness_mm"])
        self.assertGreater(m["gross_internal_volume_m3"], 0)
        self.assertEqual(m["column_count"], 6)
        self.assertIn("Architectural Metrics", build_ifc_context_text(IFC_PATH))
        self.assertIn("Gross Internal Area", build_ifc_context_text(IFC_PATH))

    def test_caching_returns_same_object(self):
        a = get_ifc_facts(IFC_PATH)
        b = get_ifc_facts(IFC_PATH)
        self.assertIs(a, b)


class TestIfcRouting(unittest.TestCase):
    def test_choose_executor_routes_ifc(self):
        self.assertEqual(_choose_executor(_make_ifc_route()), RouteExecutor.IFC_QA)

    def test_parse_llm_response_accepts_ifc_intent(self):
        raw = json.dumps(
            {"intent": "ifc_model_qa", "lab": None, "second_lab": None,
             "metrics": [], "time_phrase": None, "confidence": 0.9}
        )
        plan = _parse_llm_response(raw, "how many columns are in the building?", None)
        self.assertIsNotNone(plan)
        self.assertEqual(plan.intent, IntentType.IFC_MODEL_QA)
        self.assertIsNone(plan.viewer_type)

    def test_ifc_distinct_from_viewer_control(self):
        raw = json.dumps(
            {"intent": "viewer_control", "viewer_type": "ifc", "lab": None,
             "second_lab": None, "metrics": [], "time_phrase": None, "confidence": 0.9}
        )
        plan = _parse_llm_response(raw, "open the IFC view", None)
        self.assertEqual(plan.intent, IntentType.VIEWER_CONTROL)
        self.assertEqual(_choose_executor(plan), RouteExecutor.VIEWER_CONTROL)


class TestIfcExecutor(unittest.TestCase):
    @unittest.skipUnless(_HAS_MODEL, "smart.ifc not present")
    def test_execute_ifc_structure_with_mocked_llm(self):
        with patch(
            "executors.ifc_executor._coerce_text", side_effect=lambda v: "There are 6 columns."
        ), patch("httpx.Client") as mock_client:
            mock_ctx = mock_client.return_value.__enter__.return_value
            mock_resp = mock_ctx.post.return_value
            mock_resp.json.return_value = {"response": "There are 6 columns."}
            mock_resp.raise_for_status.return_value = None
            result = render_sync(_ifc_branch(_make_ifc_route(), "how many columns are there?"))

        self.assertEqual(result["metadata"]["executor"], "ifc_qa")
        self.assertEqual(result["timescale"], "model")
        self.assertEqual(result["metadata"]["intent"], "ifc_model_qa")
        self.assertTrue(result["answer"])
        self.assertEqual(result["metadata"]["ui"]["panel"], "ifc")

    @unittest.skipUnless(_HAS_MODEL, "smart.ifc not present")
    def test_deterministic_fallback_used_when_llm_fails(self):
        with patch("httpx.Client", side_effect=RuntimeError("ollama down")):
            result = render_sync(_ifc_branch(_make_ifc_route(), "describe the building"))
        # Fallback never fabricates: it reports real counts from the model.
        self.assertIn("element", result["answer"].lower())
        self.assertFalse(result["metadata"]["llm_used"])


class TestIfcStream(unittest.TestCase):
    @unittest.skipUnless(_HAS_MODEL, "smart.ifc not present")
    def test_stream_emits_meta_and_done(self):
        from query_routing.query_orchestrator import stream_query
        from storage.conversation_context import ConversationContext

        ctx = ConversationContext(
            original_question="how many doors are there?",
            effective_question="how many doors are there?",
            effective_lab=None,
            routing_snippet="",
            llm_history="",
            carried_metric=None,
            carried_time_phrase=None,
            conversation_id="test-conv",
            raw_block="",
        )
        route = _make_ifc_route()

        async def _fake_plan(*a, **kw):
            return route

        async def _fake_tokens(user_question):
            yield f"data: {json.dumps({'event': 'token', 'text': '2 doors.'})}\n\n"
            yield f"data: {json.dumps({'event': 'done'})}\n\n"

        async def _run():
            chunks = []
            async for chunk in stream_query(ctx, k=5):
                chunks.append(chunk)
            return chunks

        with patch("query_routing.query_orchestrator.plan_route_async", side_effect=_fake_plan), \
             patch("query_routing.query_orchestrator.stream_ifc_tokens", side_effect=_fake_tokens):
            chunks = asyncio.new_event_loop().run_until_complete(_run())

        events = [json.loads(c.removeprefix("data: ").strip()) for c in chunks if c.strip()]
        types = [e.get("event") for e in events]
        self.assertIn("meta", types)
        self.assertIn("token", types)
        self.assertIn("done", types)
        meta = next(e for e in events if e.get("event") == "meta")
        self.assertEqual(meta["executor"], "ifc_qa")


class TestIfcStreamFallback(unittest.TestCase):
    """An unreachable answer LLM must not turn into an empty stream.

    The sync path already answered from the parsed model facts; the stream swallowed the
    exception and emitted only ``done``, so the user saw nothing and the blank turn was
    skipped by persistence.
    """

    def _stream_with_llm_down(self, summary_patch):
        """Stream with a parsed model but an unreachable answer LLM."""
        from contextlib import ExitStack

        from executors import ifc_executor

        async def _run():
            return [chunk async for chunk in ifc_executor.stream_ifc_tokens("describe the building")]

        with ExitStack() as stack:
            stack.enter_context(
                patch.object(ifc_executor, "build_ifc_context_text", return_value="MODEL CONTEXT")
            )
            stack.enter_context(
                patch.object(ifc_executor.httpx, "AsyncClient", side_effect=RuntimeError("ollama down"))
            )
            stack.enter_context(summary_patch)
            chunks = asyncio.new_event_loop().run_until_complete(_run())
        return [json.loads(c.removeprefix("data: ").strip()) for c in chunks if c.strip()]

    _SUMMARY = {
        "project": "Smart Building",
        "schema": "IFC4",
        "total_elements": 42,
        "element_counts": {"walls": 30, "doors": 12},
        "storeys": [{"name": "Level 1"}],
    }

    def test_llm_outage_falls_back_to_parsed_model_facts(self):
        from executors import ifc_executor

        events = self._stream_with_llm_down(
            patch.object(ifc_executor, "get_ifc_summary", return_value=self._SUMMARY)
        )
        tokens = [e for e in events if e["event"] == "token"]
        self.assertEqual(len(tokens), 1)
        # Grounded in the parsed model, never invented.
        self.assertIn("42 physical elements", tokens[0]["text"])
        self.assertIn("Level 1", tokens[0]["text"])
        self.assertEqual(events[-1]["event"], "done")

    def test_unreadable_model_still_terminates_with_an_explanation(self):
        from executors import ifc_executor

        events = self._stream_with_llm_down(
            patch.object(ifc_executor, "get_ifc_summary", side_effect=RuntimeError("bad file"))
        )
        tokens = [e for e in events if e["event"] == "token"]
        self.assertEqual(len(tokens), 1)
        self.assertIn("not available", tokens[0]["text"])
        self.assertEqual(events[-1]["event"], "done")

    def test_missing_model_file_reports_cleanly(self):
        from executors import ifc_executor

        async def _run():
            return [chunk async for chunk in ifc_executor.stream_ifc_tokens("describe the building")]

        with patch.object(ifc_executor, "build_ifc_context_text", side_effect=FileNotFoundError):
            chunks = asyncio.new_event_loop().run_until_complete(_run())

        events = [json.loads(c.removeprefix("data: ").strip()) for c in chunks if c.strip()]
        self.assertIn("not available", events[0]["text"])
        self.assertEqual(events[-1]["event"], "done")

    def test_successful_stream_does_not_append_a_fallback(self):
        from executors import ifc_executor

        class _Response:
            def raise_for_status(self):
                return None

            async def aiter_lines(self):
                yield json.dumps({"response": "The building has 12 doors."})

        class _Stream:
            async def __aenter__(self):
                return _Response()

            async def __aexit__(self, *exc):
                return False

        class _Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            def stream(self, *a, **kw):
                return _Stream()

        async def _run():
            return [chunk async for chunk in ifc_executor.stream_ifc_tokens("how many doors?")]

        with patch.object(ifc_executor, "build_ifc_context_text", return_value="MODEL CONTEXT"), \
             patch.object(ifc_executor.httpx, "AsyncClient", return_value=_Client()), \
             patch.object(ifc_executor, "get_ifc_summary", return_value=self._SUMMARY):
            chunks = asyncio.new_event_loop().run_until_complete(_run())

        events = [json.loads(c.removeprefix("data: ").strip()) for c in chunks if c.strip()]
        tokens = [e for e in events if e["event"] == "token"]
        self.assertEqual(len(tokens), 1)
        self.assertEqual(tokens[0]["text"], "The building has 12 doors.")


if __name__ == "__main__":
    unittest.main()
