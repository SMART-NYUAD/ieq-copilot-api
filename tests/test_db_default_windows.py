import os
import sys
import unittest
from datetime import datetime, timedelta, timezone


TEST_DIR = os.path.dirname(__file__)
SERVER_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)
REPO_DIR = os.path.abspath(os.path.join(SERVER_DIR, ".."))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from query_routing.intent_classifier import IntentType
from executors.db_query_executor import _build_db_payload, _default_window_hours_for_intent
from executors.db_support.query_parsing import (
    extract_metric_aliases,
    extract_time_window,
    pick_metric,
    strip_conversation_context,
)


class DbDefaultWindowTests(unittest.TestCase):
    def test_point_lookup_defaults_to_last_hour(self):
        self.assertEqual(_default_window_hours_for_intent(IntentType.POINT_LOOKUP_DB), 1)
        self.assertEqual(_default_window_hours_for_intent(IntentType.CURRENT_STATUS_DB), 1)

    def test_aggregation_like_defaults_to_last_day(self):
        self.assertEqual(_default_window_hours_for_intent(IntentType.AGGREGATION_DB), 24)
        self.assertEqual(_default_window_hours_for_intent(IntentType.COMPARISON_DB), 24)
        self.assertEqual(_default_window_hours_for_intent(IntentType.ANOMALY_ANALYSIS_DB), 24)

    def test_forecast_keeps_existing_default(self):
        self.assertEqual(_default_window_hours_for_intent(IntentType.FORECAST_DB), 24)

    def test_db_payload_includes_deterministic_display_window(self):
        payload = _build_db_payload(
            intent=IntentType.AGGREGATION_DB,
            metric_alias="co2",
            window_label="last 24 hours",
            rows=[],
            window_start="2026-03-27T10:15:28+00:00",
            window_end="2026-03-28T10:15:28+00:00",
            display_start="Mar 27, 2026, 2:15 PM GMT+4",
            display_end="Mar 28, 2026, 2:15 PM GMT+4",
        )
        self.assertEqual(payload.get("display_start"), "Mar 27, 2026, 2:15 PM GMT+4")
        self.assertEqual(payload.get("display_end"), "Mar 28, 2026, 2:15 PM GMT+4")

    def test_extract_time_window_understands_last_hour_phrase(self):
        start, end, label = extract_time_window("pm2.5 in smart lab for the last hour", default_hours=24)
        self.assertEqual(label, "last 1 hour")
        self.assertAlmostEqual((end - start).total_seconds(), 3600.0, delta=2.0)

    def test_extract_time_window_ignores_appended_conversation_context(self):
        question = (
            "pm2.5 in smart lab for the last hour\n\n"
            "Previous conversation context (most recent last):\n"
            "User: show March 28 report\n"
            "Assistant: done"
        )
        start, end, label = extract_time_window(question, default_hours=24)
        self.assertEqual(label, "last 1 hour")
        self.assertAlmostEqual((end - start).total_seconds(), 3600.0, delta=2.0)

    def test_current_day_window_is_capped_to_now(self):
        target_tz = timezone(timedelta(hours=4))
        now = datetime.now(target_tz)
        day_question = f"pm2.5 in smart lab on {now.strftime('%B')} {now.day}"
        start, end, _ = extract_time_window(day_question, default_hours=24)
        self.assertLessEqual(end, datetime.now(target_tz) + timedelta(seconds=1))
        self.assertLess(end, start + timedelta(days=1))

    def test_metric_parsing_ignores_appended_conversation_context(self):
        question = (
            "find anomalies in smart lab last week\n\n"
            "Previous conversation context (most recent last):\n"
            "User: show PM2.5 in shores_office for last 24 hours\n"
            "Assistant: PM2.5 was low"
        )
        cleaned = strip_conversation_context(question)
        metric_alias, _ = pick_metric(cleaned)
        self.assertEqual(metric_alias, "ieq")
        self.assertEqual(extract_metric_aliases(cleaned), [])


if __name__ == "__main__":
    unittest.main()
