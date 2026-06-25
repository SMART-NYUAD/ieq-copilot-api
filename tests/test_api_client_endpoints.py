"""Tests for the REST API client's agg-summary / indoor-data parameter wiring.

The refactored endpoints take explicit ``window_start``/``window_end`` bounds plus a
range-derived granularity (``interval_hours`` for agg-summary, ``interval`` for
indoor-data) rather than a hours-back-from-now lookback.
"""

import os
import sys
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

TEST_DIR = os.path.dirname(__file__)
SERVER_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

from executors.db_support import api_client


def _ok_response(payload):
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"success": True, "data": payload}
    return resp


class AggSummaryParamTests(unittest.TestCase):
    def setUp(self):
        api_client._RESPONSE_CACHE.clear()

    def test_agg_summary_sends_window_bounds_and_derived_interval(self):
        end = datetime(2026, 6, 10, tzinfo=timezone.utc)
        start = end - timedelta(days=10)  # a week-to-a-month span → 6h granularity
        client = MagicMock()
        client.get.return_value = _ok_response({"aggregate_readings": []})
        with patch.object(api_client, "_get_client", return_value=client):
            api_client.fetch_metric_agg_summary("smart_lab", "pm25", start, end)
        _, kwargs = client.get.call_args
        params = kwargs["params"]
        self.assertEqual(params["window_start"], start.isoformat())
        self.assertEqual(params["window_end"], end.isoformat())
        self.assertEqual(params["interval_hours"], 6)
        self.assertNotIn("window_hours", params)

    def test_agg_summary_explicit_interval_overrides_derivation(self):
        end = datetime(2026, 6, 10, tzinfo=timezone.utc)
        start = end - timedelta(days=10)
        client = MagicMock()
        client.get.return_value = _ok_response({"aggregate_readings": []})
        with patch.object(api_client, "_get_client", return_value=client):
            api_client.fetch_metric_agg_summary("smart_lab", "pm25", start, end, interval_hours=1)
        self.assertEqual(client.get.call_args.kwargs["params"]["interval_hours"], 1)


class IndoorDataParamTests(unittest.TestCase):
    def setUp(self):
        api_client._RESPONSE_CACHE.clear()

    def test_indoor_data_sends_window_bounds_and_type(self):
        end = datetime(2026, 6, 10, tzinfo=timezone.utc)
        start = end - timedelta(days=40)  # a month-plus span → 12h granularity
        client = MagicMock()
        client.get.return_value = _ok_response({"readings": []})
        with patch.object(api_client, "_get_client", return_value=client):
            api_client.fetch_indoor_data("smart_lab", "IEQ", start, end)
        params = client.get.call_args.kwargs["params"]
        self.assertEqual(params["type"], "IEQ")
        self.assertEqual(params["window_start"], start.isoformat())
        self.assertEqual(params["window_end"], end.isoformat())
        self.assertEqual(params["interval"], 12)
        self.assertNotIn("timeframe", params)


if __name__ == "__main__":
    unittest.main()
