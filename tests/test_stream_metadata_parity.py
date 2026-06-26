"""The streamed DB path must reconcile its placeholder meta with the resolved facts.

Before the fix, /query/stream emitted `timescale: "pending"` in the meta event and never
corrected it, and derived its UI from the (pre-DB) route metrics — diverging from the sync
/query response. The stream now emits a `meta_update` carrying the resolved timescale and a
UI contract derived from the executed query's metrics_used.
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
from query_routing.router_types import RoutePlan
from query_routing import query_orchestrator as qo
from storage.conversation_context import ConversationContext


def _ctx():
    return ConversationContext(
        original_question="what was the average co2 last week?",
        effective_question="what was the average co2 last week?",
        effective_lab="smart_lab",
        routing_snippet="",
        llm_history="",
        carried_metric=None,
        carried_time_phrase=None,
        conversation_id="test-conv",
        raw_block="",
    )


def _route():
    return RoutePlan(
        intent=IntentType.AGGREGATION_DB,
        confidence=0.9,
        lab_name="smart_lab",
        time_phrase="last week",
        model="test",
        fallback_used=False,
        metrics=["ieq", "co2"],  # pre-DB default; resolved query narrows to a single metric
    )


def _collect(ctx):
    async def _fake_plan(*a, **kw):
        return _route()

    def _fake_prepare(*a, **kw):
        return {
            "timescale": "1week",
            "time_window": {"label": "last week", "start": "x", "end": "y"},
            "resolved_lab_name": "smart_lab",
            "metrics_used": ["co2"],
        }

    async def _fake_tokens(*a, **kw):
        yield f"data: {json.dumps({'event': 'token', 'text': 'avg co2 was 600ppm'})}\n\n"
        yield f"data: {json.dumps({'event': 'done'})}\n\n"

    async def _run():
        chunks = []
        async for chunk in qo.stream_query(ctx, k=5):
            chunks.append(chunk)
        return chunks

    with patch.object(qo, "plan_route_async", side_effect=_fake_plan), \
         patch.object(qo, "prepare_db_query", side_effect=_fake_prepare), \
         patch.object(qo, "stream_db_tokens", side_effect=_fake_tokens):
        return asyncio.new_event_loop().run_until_complete(_run())


def _events(chunks):
    out = []
    for chunk in chunks:
        raw = chunk.removeprefix("data: ").strip()
        if raw:
            out.append(json.loads(raw))
    return out


class TestStreamMetadataParity(unittest.TestCase):
    def test_meta_update_carries_resolved_timescale_and_ui(self):
        events = _events(_collect(_ctx()))
        meta_updates = [e for e in events if e.get("event") == "meta_update"]
        self.assertEqual(len(meta_updates), 1)
        mu = meta_updates[0]
        # Resolved timescale replaces the initial "pending".
        self.assertEqual(mu["timescale"], "1week")
        self.assertEqual(mu["metrics_used"], ["co2"])
        # UI is derived from the executed query's single metric, not the route's 2-metric default.
        self.assertEqual(mu["ui"]["panel"], "single_metric")
        self.assertEqual(mu["ui"]["primary_metric"], "co2")

    def test_initial_meta_is_pending_then_corrected(self):
        events = _events(_collect(_ctx()))
        metas = [e for e in events if e.get("event") == "meta"]
        self.assertEqual(len(metas), 1)
        self.assertEqual(metas[0]["timescale"], "pending")
        # The later meta_update supersedes it with the resolved value.
        mu = [e for e in events if e.get("event") == "meta_update"][0]
        self.assertNotEqual(mu["timescale"], "pending")


if __name__ == "__main__":
    unittest.main()
