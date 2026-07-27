"""Display timezone is configured in one place, and windows are bounded.

Two separate hazards:
  * the parser used a hardcoded +4 constant while the serializers read
    ``DISPLAY_UTC_OFFSET_HOURS``, so changing the setting moved some timestamps and not
    others — a silent, near-undebuggable inconsistency;
  * aggregation is always hourly, so an unbounded window turns one question into tens of
    thousands of upstream buckets.
"""

import os
import sys
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

TEST_DIR = os.path.dirname(__file__)
SERVER_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

from executors.db_support import query_parsing, time_windows


class DisplayTimezoneTests(unittest.TestCase):
    def test_parser_and_display_share_one_setting(self):
        with patch.dict(os.environ, {"DISPLAY_UTC_OFFSET_HOURS": "-5"}):
            self.assertEqual(time_windows.target_tz(), timezone(timedelta(hours=-5)))
            self.assertEqual(time_windows.target_tz_label(), "GMT-5")
            # The window parser anchors "today" in the same zone it labels with.
            start, _end, _label = query_parsing.extract_time_window("co2 today")
            self.assertEqual(start.utcoffset(), timedelta(hours=-5))

    def test_default_offset_is_gulf_standard_time(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DISPLAY_UTC_OFFSET_HOURS", None)
            self.assertEqual(time_windows.target_tz(), timezone(timedelta(hours=4)))
            self.assertEqual(time_windows.target_tz_label(), "GMT+4")

    def test_zero_offset_is_labelled_utc(self):
        with patch.dict(os.environ, {"DISPLAY_UTC_OFFSET_HOURS": "0"}):
            self.assertEqual(time_windows.target_tz_label(), "UTC")

    def test_display_string_follows_the_configured_offset(self):
        moment = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
        with patch.dict(os.environ, {"DISPLAY_UTC_OFFSET_HOURS": "2"}):
            rendered = time_windows.format_display_datetime(moment)
        self.assertIn("GMT+2", rendered)
        self.assertIn("2:00 PM", rendered)


class MaxWindowGuardTests(unittest.TestCase):
    def test_absurd_window_is_clamped(self):
        with patch.dict(os.environ, {"MAX_QUERY_WINDOW_DAYS": "366"}):
            start, end, _label = query_parsing.extract_time_window("average co2 last 5000 days")
        self.assertLessEqual((end - start).days, 366)

    def test_normal_window_is_untouched(self):
        start, end, label = query_parsing.extract_time_window("average co2 last 30 days")
        self.assertEqual((end - start).days, 30)
        self.assertEqual(label, "last 30 days")

    def test_clamped_label_says_the_window_was_trimmed(self):
        # The label reaches the answer LLM and the response metadata; leaving the
        # original phrasing on a trimmed window makes the answer claim a range it
        # never read.
        with patch.dict(os.environ, {"MAX_QUERY_WINDOW_DAYS": "366"}):
            _start, _end, label = query_parsing.extract_time_window(
                "average co2 from january 1 2015 to june 1 2026"
            )
        self.assertIn("limited to the last 366 days", label)

    def test_cap_is_configurable(self):
        with patch.dict(os.environ, {"MAX_QUERY_WINDOW_DAYS": "7"}):
            start, end, _label = query_parsing.extract_time_window("average co2 last 90 days")
        self.assertEqual((end - start).days, 7)

    def test_clamp_keeps_the_window_end(self):
        # Clamping trims the start so the answer still covers the most recent data.
        with patch.dict(os.environ, {"MAX_QUERY_WINDOW_DAYS": "2"}):
            uncapped_start, uncapped_end, _ = query_parsing._resolve_time_window("average co2 last 90 days")
            start, end, _ = query_parsing.extract_time_window("average co2 last 90 days")
        # Both calls anchor on "now", so compare within a tolerance rather than exactly.
        self.assertLess(abs(end - uncapped_end), timedelta(seconds=5))
        self.assertGreater(start, uncapped_start)


if __name__ == "__main__":
    unittest.main()
