"""Each bound of a date range carries its own year.

The question-level year (the first one found anywhere in the text) was applied to both
bounds, so "from January 1 2015 to June 1 2026" resolved to a range ending in 2015 — the
answer covered a window the user never asked for, with nothing in the response to show it.

The same year loss existed in the follow-up carry-over: the phrase extracted for the next
turn dropped the year, so a carried date silently re-resolved against the current year.
"""

import os
import sys
import unittest

TEST_DIR = os.path.dirname(__file__)
SERVER_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

from executors.db_support.query_parsing import _resolve_time_window
from storage.conversation_memory import _explicit_date_phrase, _explicit_date_range_phrase


class RangeBoundYearTests(unittest.TestCase):
    """``_resolve_time_window`` is the uncapped resolver — the max-window clamp is
    tested separately, and would otherwise mask a multi-year range."""

    def test_each_bound_keeps_its_own_year(self):
        start, end, _label = _resolve_time_window("from january 1 2015 to june 1 2026")
        self.assertEqual((start.year, start.month, start.day), (2015, 1, 1))
        self.assertEqual((end.year, end.month), (2026, 6))

    def test_label_reports_both_years(self):
        _start, _end, label = _resolve_time_window("from january 1 2015 to june 1 2026")
        self.assertIn("2015", label)
        self.assertIn("2026", label)

    def test_matching_years_still_resolve(self):
        start, end, _label = _resolve_time_window("from may 1 2026 to may 20 2026")
        self.assertEqual(start.year, 2026)
        self.assertEqual(end.year, 2026)
        self.assertEqual((end - start).days, 20)

    def test_year_on_one_bound_applies_to_the_other(self):
        # "from may 1 to may 20 2026" — the bound without a year falls back to the
        # question-level year rather than drifting to the current one.
        start, end, _label = _resolve_time_window("from may 1 to may 20 2026")
        self.assertEqual(start.year, 2026)
        self.assertEqual(end.year, 2026)

    def test_day_first_bounds_keep_their_years(self):
        start, end, _label = _resolve_time_window("co2 from 1 may 2024 to 10 may 2025")
        self.assertEqual(start.year, 2024)
        self.assertEqual(end.year, 2025)

    def test_iso_bounds_are_unaffected(self):
        start, end, _label = _resolve_time_window("co2 from 2024-05-01 to 2025-05-10")
        self.assertEqual((start.year, start.month, start.day), (2024, 5, 1))
        self.assertEqual((end.year, end.month, end.day), (2025, 5, 11))  # end day inclusive

    def test_single_date_with_a_year_is_unaffected(self):
        start, _end, _label = _resolve_time_window("what was the pm25 on may 3 2025?")
        self.assertEqual((start.year, start.month, start.day), (2025, 5, 3))


class CarryOverYearTests(unittest.TestCase):
    def test_carried_range_phrase_keeps_both_years(self):
        phrase = _explicit_date_range_phrase("co2 from january 1 2015 to june 1 2026")
        self.assertEqual(phrase, "from january 1 2015 to june 1 2026")

    def test_carried_single_date_keeps_its_year(self):
        self.assertEqual(_explicit_date_phrase("pm25 on may 3 2025"), "may 3 2025")

    def test_carried_bare_month_keeps_its_year(self):
        self.assertEqual(_explicit_date_phrase("co2 in june 2024"), "june 2024")

    def test_carried_week_of_month_keeps_its_year(self):
        self.assertEqual(
            _explicit_date_phrase("pm25 in the first week of may 2025"), "first week of may 2025"
        )

    def test_carried_phrase_without_a_year_is_unchanged(self):
        self.assertEqual(_explicit_date_phrase("pm25 on may 3"), "may 3")

    def test_carried_phrase_round_trips_to_the_same_window(self):
        # A carried phrase is re-parsed on the next turn; it must resolve to the window
        # the original question described, not to the current year.
        original = "what was the pm25 on may 3 2025?"
        carried = _explicit_date_phrase(original)
        want_start, want_end, _ = _resolve_time_window(original)
        got_start, got_end, _ = _resolve_time_window(carried)
        self.assertEqual((got_start, got_end), (want_start, want_end))

    def test_carried_range_round_trips_to_the_same_window(self):
        original = "co2 from 1 may 2024 to 10 may 2025"
        carried = _explicit_date_range_phrase(original)
        want_start, want_end, _ = _resolve_time_window(original)
        got_start, got_end, _ = _resolve_time_window(carried)
        self.assertEqual((got_start, got_end), (want_start, want_end))


if __name__ == "__main__":
    unittest.main()
