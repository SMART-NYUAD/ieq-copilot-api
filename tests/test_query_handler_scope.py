"""Handler-level correctness for space scope and period partitioning.

Two defects found auditing ``query_handlers``:

  * a current-status question naming no space answered from whichever space the API
    listed first, while the response reported no resolved space — one room's air
    presented as the whole site's. This became reachable when the regex gate that used
    to block unscoped queries was replaced by the router's clarify decision.
  * the baseline comparison split "current" from "baseline" rows by comparing ISO
    strings. Window bounds are built in the display timezone and buckets come back in
    the API's, so the comparison mis-partitioned rows near the boundary. The sibling
    temporal-comparison handler already parsed instants properly.
"""

import os
import sys
import unittest
from datetime import datetime, timedelta, timezone

TEST_DIR = os.path.dirname(__file__)
SERVER_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

from fake_sensor_api import SPACES, FakeSensorApiMixin
from executors.db_support import api_client, query_handlers
from executors.db_support.time_windows import parse_bucket_utc
from query_routing.intent_classifier import IntentType


class UnscopedPointLookupTests(FakeSensorApiMixin, unittest.TestCase):
    def _run(self, question="how is the air quality?"):
        return query_handlers._handle_point_lookup(
            question=question,
            intent=IntentType.CURRENT_STATUS_DB,
            metric_alias="co2",
            unit="ppm",
            requested_metrics=["co2", "pm25", "voc", "humidity", "ieq"],
            window_start=datetime(2026, 6, 1, tzinfo=timezone.utc),
            window_end=datetime(2026, 6, 1, 1, tzinfo=timezone.utc),
            window_label="last 1 hours",
            resolved_lab_name=None,
        )

    def test_no_space_named_averages_across_spaces(self):
        result = self._run()
        rows = result["rows"]
        self.assertEqual(len(rows), 1)
        # Not a single arbitrary space's slug.
        self.assertEqual(rows[0]["lab_space"], "all_labs")
        self.assertNotIn(rows[0]["lab_space"], [s["slug"] for s in SPACES])

    def test_named_space_still_answers_for_that_space(self):
        result = query_handlers._handle_point_lookup(
            question="how is the air quality?",
            intent=IntentType.CURRENT_STATUS_DB,
            metric_alias="co2",
            unit="ppm",
            requested_metrics=["co2", "pm25", "voc", "humidity", "ieq"],
            window_start=datetime(2026, 6, 1, tzinfo=timezone.utc),
            window_end=datetime(2026, 6, 1, 1, tzinfo=timezone.utc),
            window_label="last 1 hours",
            resolved_lab_name="concrete_lab",
        )
        self.assertEqual(result["rows"][0]["lab_space"], "concrete_lab")


class AllSpacesPointRowTests(FakeSensorApiMixin, unittest.TestCase):
    def test_averages_every_space(self):
        row = api_client.fetch_all_spaces_point_row(["co2", "temperature"])
        self.assertEqual(row["lab_space"], "all_labs")
        # The stub returns identical readings per space, so the mean equals the value.
        self.assertAlmostEqual(row["co2"], 430.0)
        self.assertAlmostEqual(row["temperature"], 22.5)

    def test_survives_an_empty_space_list(self):
        self.fake_api.empty = True
        api_client._RESPONSE_CACHE.clear()
        row = api_client.fetch_all_spaces_point_row(["co2"])
        self.assertEqual(row["lab_space"], "all_labs")
        self.assertIsNone(row["co2"])


class BucketParsingTests(unittest.TestCase):
    def test_offsets_compare_by_instant_not_string(self):
        # Same instant, different offset spellings: string comparison gets this wrong.
        utc_text, gulf_text = "2026-06-01T00:00:00+00:00", "2026-06-01T04:00:00+04:00"
        # The two spellings denote the same moment...
        self.assertEqual(parse_bucket_utc(utc_text), parse_bucket_utc(gulf_text))
        # ...but sort differently as strings, which is what broke the partition.
        self.assertGreater(gulf_text, utc_text)

    def test_zulu_suffix_is_accepted(self):
        self.assertEqual(
            parse_bucket_utc("2026-06-01T00:00:00Z"),
            datetime(2026, 6, 1, tzinfo=timezone.utc),
        )

    def test_naive_timestamps_are_treated_as_utc(self):
        self.assertEqual(
            parse_bucket_utc("2026-06-01T00:00:00"),
            datetime(2026, 6, 1, tzinfo=timezone.utc),
        )

    def test_garbage_returns_none(self):
        for value in ("", None, "not a date", 12345):
            self.assertIsNone(parse_bucket_utc(value), value)


class BaselinePartitionTests(FakeSensorApiMixin, unittest.TestCase):
    def test_boundary_rows_are_partitioned_by_instant(self):
        """Window bounds in the display timezone must still split UTC buckets correctly."""
        gulf = timezone(timedelta(hours=4))
        window_start = datetime(2026, 6, 1, 12, 0, tzinfo=gulf)
        window_end = datetime(2026, 6, 1, 18, 0, tzinfo=gulf)

        # Buckets straddling the boundary, expressed in UTC as the API returns them.
        rows = [
            {"lab_space": "smart_lab", "bucket": "2026-06-01T06:00:00+00:00", "value": 10.0},
            {"lab_space": "smart_lab", "bucket": "2026-06-01T07:00:00+00:00", "value": 12.0},
            {"lab_space": "smart_lab", "bucket": "2026-06-01T09:00:00+00:00", "value": 20.0},
            {"lab_space": "smart_lab", "bucket": "2026-06-01T10:00:00+00:00", "value": 22.0},
        ]
        original = api_client.fetch_timeseries_rows
        api_client.fetch_timeseries_rows = lambda *a, **kw: rows
        self.addCleanup(setattr, api_client, "fetch_timeseries_rows", original)

        result = query_handlers._handle_baseline_reference_comparison(
            question="is co2 higher than normal in smart_lab?",
            intent=IntentType.COMPARISON_DB,
            metric_alias="co2",
            unit="ppm",
            window_start=window_start,
            window_end=window_end,
            window_label="today",
            resolved_lab_name="smart_lab",
        )

        row = result["rows"][0]
        # 08:00Z is the boundary: 06/07Z are baseline, 09/10Z are current.
        self.assertAlmostEqual(row["baseline_avg"], 11.0)
        self.assertAlmostEqual(row["current_avg"], 21.0)
        self.assertEqual(row["baseline_count"], 2)
        self.assertEqual(row["current_count"], 2)


if __name__ == "__main__":
    unittest.main()
