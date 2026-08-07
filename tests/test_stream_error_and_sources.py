"""The stream must terminate cleanly on error and report its citations.

Two regressions this locks down:
  * ``stream_error_payload`` was called without its required ``scope`` argument and
    returned a dict, so a mid-stream failure raised a second exception inside the
    except block and the client never received a terminating frame.
  * The streamed DB/knowledge answers were generated with a numbered sources block
    (so the model emits ``[N]`` markers) but the sources were never sent, leaving
    dangling markers while the sync ``/query`` response returned full footnotes.
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

from executors import sse
from query_routing.intent_classifier import IntentType
from runtime_errors import stream_error_payload


def _events(chunks):
    out = []
    for chunk in chunks:
        for frame in str(chunk).split("data: "):
            raw = frame.strip()
            if raw:
                out.append(json.loads(raw))
    return out


_GUIDELINE_RECORDS = [
    {
        "source_key": "RESET_AIR_A",
        "source_label": "RESET Air Grade A",
        "section_ref": "CO2",
        "citation_tier": "regulatory",
        "source_url": "https://example.org/reset",
        "threshold_value": 1000,
        "threshold_unit": "ppm",
        "metric": "co2",
    },
    {
        "source_key": "STUDY_2019",
        "source_label": "Cognitive study",
        "citation_tier": "research",
        "metric": "co2",
    },
]


class StreamErrorPayloadTests(unittest.TestCase):
    def test_returns_error_then_done_frames(self):
        payload = stream_error_payload(ValueError("boom"), scope="query.stream")
        self.assertIsInstance(payload, str)
        events = _events([payload])
        self.assertEqual([e["event"] for e in events], ["error", "done"])
        self.assertEqual(events[0]["detail"], "boom")
        self.assertEqual(events[0]["scope"], "query.stream")
        self.assertTrue(events[0]["code"])

    def test_call_shape_used_by_the_stream_route_does_not_raise(self):
        # The route yields this directly from an except block; a TypeError here
        # would kill the connection instead of reporting the error.
        frames = stream_error_payload(RuntimeError("upstream down"), scope="query.stream")
        self.assertTrue(frames.endswith("\n\n"))
        self.assertIn('"event": "done"', frames)


class SourcesEventTests(unittest.TestCase):
    def test_footnotes_cover_only_the_markers_the_answer_emitted(self):
        from evidence.citation_processor import build_numbered_sources_block

        _, indexed_sources = build_numbered_sources_block(_GUIDELINE_RECORDS)
        frame = sse.sources_event_for_answer(
            ["CO2 is above the threshold [1].", " Nothing cites source two."],
            _GUIDELINE_RECORDS,
            indexed_sources,
        )
        event = _events([frame])[0]
        self.assertEqual(event["event"], "sources")
        # Every offered source is advertised, mirroring the sync response contract.
        self.assertEqual([s["index"] for s in event["citation_sources"]], [1, 2])
        # Only the cited one becomes a footnote.
        self.assertEqual([f["index"] for f in event["footnotes"]], [1])
        self.assertEqual(event["footnotes"][0]["source_label"], "RESET Air Grade A")

    def test_no_citations_yields_empty_footnotes(self):
        frame = sse.sources_event_for_answer(["Plain answer with no markers."], [], [])
        event = _events([frame])[0]
        self.assertEqual(event["citation_sources"], [])
        self.assertEqual(event["footnotes"], [])


class DbStreamSourcesTests(unittest.TestCase):
    """The DB stream emits sources between the last token and ``done``."""

    def _collect(self, *, llm_fails: bool):
        from executors import db_query_executor as dbx

        # Shaped like a prepare_db_query result: the guideline records are resolved once,
        # during preparation, and the stream reuses them rather than re-querying.
        context = {
            # A real prepare_db_query payload names the metric the row's generic
            # `avg_value` column belongs to; the threshold assessment needs it to know
            # what 600 is a reading OF.
            "payload": {"operation_type": "aggregation", "metric": "co2"},
            "fallback_answer": "Average CO2 was 600 ppm.",
            "backend_semantic_state": None,
            "knowledge_cards": [],
            "rows": [{"avg_value": 600}],
            "metric_alias": "co2",
            "sources": [],
            "guideline_records": _GUIDELINE_RECORDS,
        }

        class _FakeResponse:
            def raise_for_status(self):
                if llm_fails:
                    raise RuntimeError("ollama unreachable")

            async def aiter_lines(self):
                yield json.dumps({"response": "Average CO2 was 600 ppm [1]."})

        class _FakeStream:
            async def __aenter__(self):
                return _FakeResponse()

            async def __aexit__(self, *exc):
                return False

        class _FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            def stream(self, *a, **kw):
                return _FakeStream()

        async def _run():
            chunks = []
            async for chunk in dbx.stream_db_tokens(
                question="average co2 last week",
                intent=IntentType.AGGREGATION_DB,
                lab_name="smart_lab",
                planner_hints=None,
                query_context=context,
            ):
                chunks.append(chunk)
            return chunks

        with patch.object(dbx, "get_thresholds_for_metrics") as threshold_lookup, \
             patch.object(dbx.httpx, "AsyncClient", return_value=_FakeClient()):
            chunks = asyncio.new_event_loop().run_until_complete(_run())
        # The streamed turn must not re-query the guideline store.
        threshold_lookup.assert_not_called()
        return chunks

    def test_sources_event_precedes_done(self):
        events = _events(self._collect(llm_fails=False))
        names = [e["event"] for e in events]
        self.assertEqual(names[-2:], ["sources", "done"])
        sources = events[-2]
        # Only RESET_AIR_A is advertised: it carries the threshold the assessment graded
        # 600 ppm against. STUDY_2019 has no threshold_value, so it governs nothing and is
        # not a source the answer may cite a limit from — see governing_records.
        self.assertEqual([s["index"] for s in sources["citation_sources"]], [1])
        self.assertEqual([f["index"] for f in sources["footnotes"]], [1])

    def test_llm_failure_still_terminates_with_done(self):
        events = _events(self._collect(llm_fails=True))
        self.assertEqual(events[-1]["event"], "done")
        self.assertTrue(any(e["event"] == "token" for e in events))


if __name__ == "__main__":
    unittest.main()
