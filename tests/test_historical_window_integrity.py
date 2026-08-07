"""A reading may never be presented under a window it does not fall in.

Three consecutive turns produced three different PM2.5 figures for the same day, and two
of them were fabrications:

  * "How was the air quality on that date?" -> PM2.5 18.0 ug/m3, VOC 0.06 ppm. Those were
    that afternoon's LIVE readings, three months after the day asked about. The
    multi-metric branch called fetch_multi_metric_point_row -- which always returns the
    latest reading -- and returned it under the label "May 7, 2026".
  * "How was the pm2.5 on may 7th?" -> "was 14.0 ug/m3". That is the 23:00 bucket, the last
    hour of the day, reported as the day's level. The day's mean was 9.19.

Both trace to one gate: the historical-window path was conditioned on
``intent == POINT_LOOKUP_DB``, and the router answers "how was X on <date>?" with
CURRENT_STATUS_DB about as often. The window was resolved correctly the whole way; only
the branch that read it was wrong. It is now conditioned on the window being CLOSED, which
is a property of the resolved window rather than of a router coin-flip.
"""

import os
import sys
import unittest
from unittest.mock import patch
from datetime import datetime, timedelta, timezone

TEST_DIR = os.path.dirname(__file__)
SERVER_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)
if TEST_DIR not in sys.path:
    sys.path.insert(0, TEST_DIR)

from executors.db_support import query_handlers as qh
from query_routing.intent_classifier import IntentType
from fake_sensor_api import FakeSensorApiMixin


def _closed_window():
    """A finished day, well in the past."""
    start = datetime(2026, 5, 7, tzinfo=timezone.utc)
    return start, start + timedelta(days=1)


def _live_window():
    # A live status question resolves to the one-hour default, not a 24-hour span: a wide
    # window is itself evidence that a period was named.
    now = datetime.now(timezone.utc)
    return now - timedelta(hours=1), now


class ClosedWindowNeverTakesTheLiveSnapshot(FakeSensorApiMixin, unittest.TestCase):
    def _run(self, question, intent, metrics, window):
        start, end = window
        self.point_row_calls = []
        real = qh.api_client.fetch_multi_metric_point_row

        def _spy(slug, metric_names):
            self.point_row_calls.append(list(metric_names))
            return real(slug, metric_names)

        qh.api_client.fetch_multi_metric_point_row = _spy
        try:
            return qh.execute_intent_query(
                question=question,
                intent=intent,
                metric_alias=metrics[0],
                metric_column="value",
                unit="ug/m3",
                window_start=start,
                window_end=end,
                window_label="May 07, 2026",
                resolved_lab_name="smart_lab",
                compared_spaces=[],
                explicit_metrics=metrics,
                hinted_metrics=metrics,
            )
        finally:
            qh.api_client.fetch_multi_metric_point_row = real

    def test_multi_metric_history_does_not_call_the_latest_reading_endpoint(self):
        # The exact defect: CURRENT_STATUS_DB + a closed window used to reach the
        # "latest reading" endpoint and label its result with the historical window.
        result = self._run(
            "How was the air quality in smart_lab on May 7, 2026?",
            IntentType.CURRENT_STATUS_DB,
            ["co2", "pm25", "voc"],
            _closed_window(),
        )
        self.assertEqual(self.point_row_calls, [], "live snapshot fetched for a past day")
        self.assertEqual(result["operation_type"], "aggregation_multi_metric")

    def test_both_status_intents_behave_the_same_on_a_closed_window(self):
        # The two intents are near synonyms to the router, so they must not disagree
        # about which endpoint answers a question about a finished day.
        ops = {
            intent: self._run(
                "How was the air quality in smart_lab on May 7, 2026?",
                intent,
                ["co2", "pm25", "voc"],
                _closed_window(),
            )["operation_type"]
            for intent in (IntentType.CURRENT_STATUS_DB, IntentType.POINT_LOOKUP_DB)
        }
        self.assertEqual(len(set(ops.values())), 1, ops)

    def test_single_metric_history_summarises_the_window_not_its_last_bucket(self):
        result = self._run(
            "How was the pm2.5 in smart_lab on May 7, 2026?",
            IntentType.CURRENT_STATUS_DB,
            ["pm25"],
            _closed_window(),
        )
        self.assertEqual(result["operation_type"], "aggregation")
        row = result["rows"][0]
        # An aggregate describes the window; a bucket describes one hour of it. Reporting
        # the final bucket as "the level on May 7" is what produced 14.0 against a daily
        # mean of 9.19.
        self.assertIn("avg_value", row)
        self.assertNotIn("bucket", row)

    def test_a_live_question_still_gets_the_live_snapshot(self):
        # The guard must not turn every status question into a historical aggregate.
        # The stub timestamps its readings at a fixed base time, so pin the window around
        # that moment rather than around wall-clock now: otherwise the window-integrity
        # check correctly drops the stub's rows and the assertion below tests nothing.
        import fake_sensor_api

        base = fake_sensor_api._BASE_TIME
        result = self._run(
            "How is the air quality in smart_lab?",
            IntentType.CURRENT_STATUS_DB,
            ["co2", "pm25", "voc"],
            (base - timedelta(hours=1), base),
        )
        self.assertEqual(result["operation_type"], "point_lookup_multi_metric")
        self.assertTrue(self.point_row_calls, "live question did not use the snapshot")
        self.assertTrue(result["rows"], "the snapshot row was dropped")

    def test_the_two_status_intents_never_disagree(self):
        """The invariant that replaces the coin-flip.

        `point_lookup_db` and `current_status_db` describe the same thing to the router,
        which picks between them roughly at random. Neither may change which data the
        answer is built from -- for ANY window, not only a closed one.
        """
        import fake_sensor_api

        base = fake_sensor_api._BASE_TIME
        cases = [
            ("How is the air quality in smart_lab?", (base - timedelta(hours=1), base)),
            ("How was the air quality in smart_lab on May 7, 2026?", _closed_window()),
        ]
        for question, window in cases:
            ops = {
                intent: self._run(question, intent, ["co2", "pm25", "voc"], window)[
                    "operation_type"
                ]
                for intent in (IntentType.CURRENT_STATUS_DB, IntentType.POINT_LOOKUP_DB)
            }
            with self.subTest(question=question):
                self.assertEqual(len(set(ops.values())), 1, ops)


class RowsOutsideTheWindowAreDropped(unittest.TestCase):
    """The backstop, tested directly: it must hold for paths that do not exist yet."""

    def test_a_reading_from_outside_the_window_is_removed(self):
        start, end = _closed_window()
        rows = [{"bucket": "2026-08-07T17:40:33+04:00", "pm25": 17.9}]
        self.assertEqual(qh._rows_within_window(rows, start, end), [])

    def test_a_reading_inside_the_window_is_kept(self):
        start, end = _closed_window()
        rows = [{"bucket": "2026-05-07T22:00:00+00:00", "pm25": 14.1}]
        self.assertEqual(len(qh._rows_within_window(rows, start, end)), 1)

    def test_an_aggregate_row_has_no_moment_to_contradict(self):
        start, end = _closed_window()
        rows = [{"lab_space": "smart_lab", "avg_value": 9.19, "reading_count": 24}]
        self.assertEqual(len(qh._rows_within_window(rows, start, end)), 1)

    def test_boundary_readings_survive_clock_skew(self):
        # The sensor API and this process need not agree to the second.
        start, end = _closed_window()
        rows = [{"bucket": (end + timedelta(minutes=20)).isoformat(), "pm25": 14.0}]
        self.assertEqual(len(qh._rows_within_window(rows, start, end)), 1)

    def test_an_unparseable_timestamp_is_not_silently_dropped(self):
        # Dropping on a parse failure would hide data for a formatting change; the
        # timestamp check only removes rows it can positively place outside the window.
        start, end = _closed_window()
        rows = [{"bucket": "not a date", "pm25": 14.0}]
        self.assertEqual(len(qh._rows_within_window(rows, start, end)), 1)

    def test_a_closed_window_is_recognised_and_a_live_one_is_not(self):
        self.assertTrue(qh._window_is_closed(_closed_window()[1]))
        self.assertFalse(qh._window_is_closed(_live_window()[1]))


class MultiMetricWindowsCarryTheirShape(unittest.TestCase):
    """A multi-metric window summary is scalars; a shape has to come from somewhere.

    Only the FIRST metric of a multi-metric answer ever got a series, so PM2.5 and VOC
    arrived as an average and a max with no times attached. That is where an invented
    narrative gets in: one answer described VOC "rising gradually through occupied hours,
    peaking in the afternoon" on a day whose peak was at 8 PM and whose quietest stretch
    was 9 AM-5 PM. The aggregate could not contradict it, because an aggregate has no
    shape in it.
    """

    SERIES = [
        {"bucket": f"2026-05-07T{hour:02d}:00:00+04:00", "voc": voc, "pm25": pm}
        for hour, voc, pm in [
            (9, 0.048, 8.8), (12, 0.050, 6.8), (15, 0.094, 8.2),
            (20, 0.196, 10.3), (22, 0.098, 14.1),
        ]
    ]

    def _trends(self, series=None, metrics=("voc", "pm25")):
        from executors.db_query_executor import _per_metric_trends

        return {
            t["metric"]: t
            for t in _per_metric_trends(self.SERIES if series is None else series, list(metrics))
        }

    def test_every_metric_gets_its_own_extrema_not_just_the_first(self):
        trends = self._trends()
        self.assertEqual(set(trends), {"voc", "pm25"})
        self.assertEqual(trends["voc"]["peak_at"][11:16], "20:00")
        self.assertEqual(trends["pm25"]["peak_at"][11:16], "22:00")

    def test_the_real_shape_contradicts_the_invented_one(self):
        # VOC's trough is inside occupied hours and its peak is after them — the reverse
        # of the narrative that was produced when this evidence was absent.
        voc = self._trends()["voc"]
        self.assertEqual(voc["trough_at"][11:16], "09:00")
        self.assertEqual(voc["peak_value"], 0.196)

    def test_each_metric_carries_its_own_unit(self):
        trends = self._trends()
        self.assertEqual(trends["voc"]["unit"], "ppm")
        self.assertNotEqual(trends["pm25"]["unit"], trends["voc"]["unit"])

    def test_a_flat_series_is_not_described_as_a_trend(self):
        flat = [{"bucket": f"2026-05-07T{h:02d}:00:00+04:00", "voc": 0.05} for h in (9, 12, 15)]
        self.assertEqual(self._trends(flat, ["voc", "pm25"])["voc"]["direction_over_window"],
                         "steady")

    def test_a_single_metric_answer_gets_no_redundant_trend_block(self):
        # The single-metric path already ships a full series; duplicating it as a summary
        # would give the model two shapes for one metric to choose between.
        self.assertEqual(self._trends(metrics=["voc"]), {})

    def test_a_metric_with_too_few_points_is_omitted_rather_than_guessed(self):
        sparse = [{"bucket": "2026-05-07T09:00:00+04:00", "voc": 0.048, "pm25": 8.8},
                  {"bucket": "2026-05-07T12:00:00+04:00", "pm25": 6.8}]
        trends = self._trends(sparse)
        self.assertIn("pm25", trends)
        self.assertNotIn("voc", trends)

    def test_the_prompt_forbids_borrowing_another_metrics_timing(self):
        from prompting.db_prompts import DB_TOOL_RESPONSE_DIRECTIVE

        self.assertIn("metric_trends", DB_TOOL_RESPONSE_DIRECTIVE)
        self.assertIn("never from a neighbouring", DB_TOOL_RESPONSE_DIRECTIVE)



if __name__ == "__main__":
    unittest.main()
