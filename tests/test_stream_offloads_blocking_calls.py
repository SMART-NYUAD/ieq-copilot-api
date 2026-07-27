"""Streaming executors must not run blocking work on the event loop.

``stream_ifc_tokens`` parsed a multi-megabyte IFC file inline and
``stream_sensor_tokens`` made a synchronous HTTP call inline. Inside an async
generator either one stalls every concurrent request, not just its own, so both
are offloaded with ``run_in_threadpool``.

Each test replaces the blocking function with a mock and the threadpool with a
recorder that does *not* call through. The offloaded function therefore appears in
``pool.calls`` and is never invoked directly — running it inline would show up as
``mock.called`` instead.
"""

import asyncio
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

TEST_DIR = os.path.dirname(__file__)
SERVER_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

from executors import ifc_executor as ifc
from executors import sensor_inspection_executor as sensors


class _RecordingThreadpool:
    """Stands in for run_in_threadpool: records the callable, returns a canned result."""

    def __init__(self, result=None, raises=None):
        self.calls = []
        self._result = result
        self._raises = raises

    async def __call__(self, func, *args, **kwargs):
        self.calls.append(func)
        if self._raises is not None:
            raise self._raises
        return self._result


def _drain(agen_factory):
    async def _run():
        chunks = []
        async for chunk in agen_factory():
            chunks.append(chunk)
        return chunks

    return asyncio.new_event_loop().run_until_complete(_run())


class IfcStreamOffloadTests(unittest.TestCase):
    def test_ifc_parse_is_offloaded_and_never_run_inline(self):
        pool = _RecordingThreadpool(result="MODEL CONTEXT")
        fake_parse = MagicMock(name="build_ifc_context_text")

        with patch.object(ifc, "run_in_threadpool", pool), \
             patch.object(ifc, "build_ifc_context_text", fake_parse), \
             patch.object(ifc.httpx, "AsyncClient", side_effect=RuntimeError("llm down")):
            chunks = _drain(lambda: ifc.stream_ifc_tokens(user_question="how many columns?"))

        self.assertIn(fake_parse, pool.calls)
        self.assertFalse(fake_parse.called, "the IFC parse ran on the event loop")
        self.assertIn('"event": "done"', "".join(chunks))

    def test_missing_model_file_still_reports_cleanly(self):
        pool = _RecordingThreadpool(raises=FileNotFoundError())

        with patch.object(ifc, "run_in_threadpool", pool), \
             patch.object(ifc, "build_ifc_context_text", MagicMock()):
            chunks = _drain(lambda: ifc.stream_ifc_tokens(user_question="how many columns?"))

        joined = "".join(chunks)
        self.assertIn("not available", joined)
        self.assertIn('"event": "done"', joined)


class SensorStreamOffloadTests(unittest.TestCase):
    def test_heatmap_fetch_is_offloaded_and_never_run_inline(self):
        pool = _RecordingThreadpool(result=[])
        fake_fetch = MagicMock(name="fetch_heatmap_metrics")

        with patch.object(sensors, "run_in_threadpool", pool), \
             patch.object(sensors.api_client, "fetch_heatmap_metrics", fake_fetch):
            chunks = _drain(lambda: sensors.stream_sensor_tokens(user_question="any faulty sensors?"))

        self.assertIn(fake_fetch, pool.calls)
        self.assertFalse(fake_fetch.called, "the sensor API call ran on the event loop")
        # No devices came back, so the stream still terminates with a grounded answer.
        self.assertIn('"event": "done"', "".join(chunks))


if __name__ == "__main__":
    unittest.main()
