"""Deterministic coverage for date-range parsing in extract_time_window.

Regression for the reported bug: ranges without a "from"/"between" lead
("May 1-7", "1 May - 7 May", "May 1 to May 7") collapsed to a single day
(May 1). These assert on the resolved window's span and month/day rather than
the absolute year, so they are independent of the system clock.
"""

import os
import sys
import unittest
from datetime import timedelta

TEST_DIR = os.path.dirname(__file__)
SERVER_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

from executors.db_support.query_parsing import extract_time_window


def _span_days(start, end):
    return (end - start).total_seconds() / 86400.0


def _inclusive_last_day(end):
    """The last fully-covered calendar day of a window (window end is exclusive)."""
    return (end - timedelta(seconds=1)).date()


class DateRangeParsingTests(unittest.TestCase):
    # --- the forms confirmed broken: no "from"/"between" lead ---------------

    def test_month_day_dash_day(self):
        start, end, _ = extract_time_window("May 1-7")
        self.assertEqual((start.month, start.day), (5, 1))
        self.assertEqual((_inclusive_last_day(end).month, _inclusive_last_day(end).day), (5, 7))
        self.assertEqual(_span_days(start, end), 7.0)

    def test_month_day_to_day(self):
        start, end, _ = extract_time_window("May 1 to 7")
        self.assertEqual((start.month, start.day), (5, 1))
        self.assertEqual(_span_days(start, end), 7.0)

    def test_day_month_dash_day_month(self):
        start, end, _ = extract_time_window("1 May - 7 May")
        self.assertEqual((start.month, start.day), (5, 1))
        self.assertEqual(_span_days(start, end), 7.0)

    def test_day_month_to_day_month(self):
        start, end, _ = extract_time_window("1 May to 7 May")
        self.assertEqual((start.month, start.day), (5, 1))
        self.assertEqual(_span_days(start, end), 7.0)

    def test_month_day_to_month_day(self):
        start, end, _ = extract_time_window("May 1 to May 7")
        self.assertEqual((start.month, start.day), (5, 1))
        self.assertEqual(_span_days(start, end), 7.0)

    def test_day_dash_day_month(self):
        start, end, _ = extract_time_window("1-7 May")
        self.assertEqual((start.month, start.day), (5, 1))
        self.assertEqual(_span_days(start, end), 7.0)

    def test_iso_range(self):
        start, end, _ = extract_time_window("2026-05-01 to 2026-05-07")
        self.assertEqual((start.year, start.month, start.day), (2026, 5, 1))
        self.assertEqual(_span_days(start, end), 7.0)

    def test_cross_month_range(self):
        start, end, _ = extract_time_window("Apr 28 - May 3")
        self.assertEqual((start.month, start.day), (4, 28))
        self.assertEqual((_inclusive_last_day(end).month, _inclusive_last_day(end).day), (5, 3))
        self.assertEqual(_span_days(start, end), 6.0)

    def test_range_embedded_in_question(self):
        start, end, _ = extract_time_window("what was the average CO2 in smart_lab for May 1-7?")
        self.assertEqual((start.month, start.day), (5, 1))
        self.assertEqual(_span_days(start, end), 7.0)

    def test_ordinal_days_in_range(self):
        start, end, _ = extract_time_window("May 1st to 7th")
        self.assertEqual((start.month, start.day), (5, 1))
        self.assertEqual(_span_days(start, end), 7.0)

    # --- regression: the existing "from"/"between" forms still work ----------

    def test_from_to_still_works(self):
        start, end, _ = extract_time_window("from May 1 to May 7")
        self.assertEqual((start.month, start.day), (5, 1))
        self.assertEqual(_span_days(start, end), 7.0)

    def test_between_and_still_works(self):
        start, end, _ = extract_time_window("between May 1 and May 8")
        self.assertEqual((start.month, start.day), (5, 1))
        self.assertEqual(_span_days(start, end), 8.0)

    # --- non-range inputs must be untouched ---------------------------------

    def test_single_date_unchanged(self):
        start, end, _ = extract_time_window("May 5")
        self.assertEqual((start.month, start.day), (5, 5))
        self.assertEqual(_span_days(start, end), 1.0)

    def test_last_n_days_not_hijacked(self):
        _, _, label = extract_time_window("last 7 days")
        self.assertEqual(label, "last 7 days")

    def test_week_of_month_not_hijacked(self):
        _, _, label = extract_time_window("first week of May")
        self.assertIn("first week", label.lower())

    def test_reversed_range_degrades_to_single_day(self):
        # A nonsensical reversed range must not produce a broken/negative window;
        # it falls through to the single-date parser (a single day), never < 0 span.
        start, end, _ = extract_time_window("May 7 to May 1")
        self.assertGreaterEqual(_span_days(start, end), 0.0)
        self.assertEqual(_span_days(start, end), 1.0)


if __name__ == "__main__":
    unittest.main()
