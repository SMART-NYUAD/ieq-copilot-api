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
from executors.db_query_executor import prepare_db_query
from executors.db_support.response_helpers import build_db_payload as _build_db_payload
from executors.db_support.query_parsing import default_window_hours_for_intent as _default_window_hours_for_intent
from executors.db_support.query_handlers import execute_intent_query
from executors.db_support.query_parsing import (
    extract_metric_aliases,
    extract_time_window,
    has_explicit_time_hint,
    pick_metric,
    strip_conversation_context,
    validate_db_execution_invariants,
)
from executors.db_support.time_windows import (
    granularity_hours_for_window,
    widen_window_to_min_span,
)


class GranularityRuleTests(unittest.TestCase):
    """The aggregation granularity (interval_hours) derived from the window span."""

    def _gran(self, hours):
        end = datetime(2026, 6, 1, tzinfo=timezone.utc)
        start = end - timedelta(hours=hours)
        return granularity_hours_for_window(start, end)

    def test_less_than_a_week_is_hourly(self):
        self.assertEqual(self._gran(1), 1)
        self.assertEqual(self._gran(24), 1)
        self.assertEqual(self._gran(6 * 24), 1)

    def test_a_week_up_to_a_month_is_six_hours(self):
        self.assertEqual(self._gran(7 * 24), 6)
        self.assertEqual(self._gran(20 * 24), 6)

    def test_a_month_or_more_is_twelve_hours(self):
        self.assertEqual(self._gran(28 * 24), 12)
        self.assertEqual(self._gran(31 * 24), 12)
        self.assertEqual(self._gran(120 * 24), 12)

    def test_widen_window_extends_short_spans_only(self):
        end = datetime(2026, 6, 1, tzinfo=timezone.utc)
        start = end - timedelta(hours=1)
        ws, we = widen_window_to_min_span(start, end, 6)
        self.assertEqual(we, end)
        self.assertAlmostEqual((we - ws).total_seconds(), 6 * 3600.0, delta=1.0)
        # A window already wider than the minimum is left untouched.
        wide_start = end - timedelta(hours=48)
        ws2, we2 = widen_window_to_min_span(wide_start, end, 6)
        self.assertEqual((ws2, we2), (wide_start, end))


class TimeRangeParsingTests(unittest.TestCase):
    """Calendar-week and explicit 'from X to Y' range parsing for the data endpoints."""

    def test_first_week_of_month_resolves_to_seven_day_window(self):
        start, end, label = extract_time_window("get the pm2.5 from the first week of July")
        self.assertEqual(start.month, 7)
        self.assertEqual(start.day, 1)
        self.assertEqual((end - start).days, 7)
        self.assertIn("first week of July", label)

    def test_second_week_of_month_offsets_by_seven_days(self):
        start, end, _ = extract_time_window("voc in the second week of August")
        self.assertEqual(start.month, 8)
        self.assertEqual(start.day, 8)
        self.assertEqual((end - start).days, 7)

    def test_last_week_of_month_ends_at_month_boundary(self):
        start, end, _ = extract_time_window("co2 in the last week of January 2025")
        self.assertEqual(start.month, 1)
        self.assertEqual(start.day, 25)
        self.assertEqual(end.month, 2)
        self.assertEqual(end.day, 1)

    def test_explicit_from_to_date_range(self):
        start, end, label = extract_time_window("pm2.5 data from July 1 to July 7 2025")
        self.assertEqual((start.month, start.day, start.year), (7, 1, 2025))
        # End day is inclusive — window covers through the whole of July 7.
        self.assertEqual((end.month, end.day), (7, 8))
        self.assertIn("–", label)

    def test_explicit_iso_from_to_range(self):
        start, end, _ = extract_time_window("temperature from 2025-07-01 to 2025-07-08")
        self.assertEqual((start.month, start.day, start.year), (7, 1, 2025))
        self.assertEqual((end.month, end.day, end.year), (7, 9, 2025))

    def test_between_and_range(self):
        start, end, _ = extract_time_window("show humidity between july 3 and july 10 2025")
        self.assertEqual((start.month, start.day), (7, 3))
        self.assertEqual((end.month, end.day), (7, 11))

    def test_range_phrases_count_as_explicit_time_hint(self):
        self.assertTrue(has_explicit_time_hint("pm2.5 from July 1 to July 7"))
        self.assertTrue(has_explicit_time_hint("co2 in the first week of July"))


class DbDefaultWindowTests(unittest.TestCase):
    def test_point_lookup_defaults_to_last_hour(self):
        self.assertEqual(_default_window_hours_for_intent(IntentType.POINT_LOOKUP_DB), 1)
        self.assertEqual(_default_window_hours_for_intent(IntentType.CURRENT_STATUS_DB), 1)

    def test_aggregation_like_defaults_to_last_day(self):
        self.assertEqual(_default_window_hours_for_intent(IntentType.AGGREGATION_DB), 24)
        self.assertEqual(_default_window_hours_for_intent(IntentType.COMPARISON_DB), 24)
        self.assertEqual(_default_window_hours_for_intent(IntentType.ANOMALY_ANALYSIS_DB), 24)

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

    def test_extract_time_window_uses_current_question_scope(self):
        start, end, label = extract_time_window("pm2.5 in smart lab for the last hour", default_hours=24)
        self.assertEqual(label, "last 1 hour")
        self.assertAlmostEqual((end - start).total_seconds(), 3600.0, delta=2.0)

    def test_generic_week_phrase_widens_window_instead_of_default(self):
        # "show me for the week" must resolve to a week-long window, not silently
        # collapse to the 24h default (which looks like the prior turn carried over).
        start, end, label = extract_time_window("Show me now for the week", default_hours=24)
        self.assertEqual(label, "last 7 days")
        self.assertAlmostEqual((end - start).total_seconds(), 7 * 86400.0, delta=2.0)

    def test_generic_month_phrase_resolves_to_month_window(self):
        start, end, label = extract_time_window("temperature over the past month", default_hours=24)
        self.assertEqual(label, "last 30 days")
        self.assertAlmostEqual((end - start).total_seconds(), 30 * 86400.0, delta=2.0)

    def test_generic_relative_phrase_counts_as_explicit_time_hint(self):
        # Without this, the carried time phrase from a prior turn would override
        # the window the current question actually asked for.
        self.assertTrue(has_explicit_time_hint("Show me for the week"))
        self.assertTrue(has_explicit_time_hint("over the past month"))

    def test_calendar_anchored_week_phrases_are_unchanged(self):
        # The generic catch must not shadow the calendar-anchored phrases.
        self.assertEqual(extract_time_window("how was it this week", default_hours=24)[2], "this week")
        self.assertEqual(extract_time_window("how was it last week", default_hours=24)[2], "last week")

    def test_current_day_window_is_capped_to_now(self):
        target_tz = timezone(timedelta(hours=4))
        now = datetime.now(target_tz)
        day_question = f"pm2.5 in smart lab on {now.strftime('%B')} {now.day}"
        start, end, _ = extract_time_window(day_question, default_hours=24)
        self.assertLessEqual(end, datetime.now(target_tz) + timedelta(seconds=1))
        self.assertLess(end, start + timedelta(days=1))

    def test_metric_parsing_uses_current_question_scope(self):
        question = "find anomalies in smart lab last week"
        metric_alias, _ = pick_metric(question)
        self.assertEqual(metric_alias, "ieq")
        self.assertEqual(extract_metric_aliases(question), [])

    def test_pick_metric_prefers_first_mention_in_question(self):
        metric_alias, _ = pick_metric("What is the temperature and CO2 in smart_lab?")
        self.assertEqual(metric_alias, "temperature")

    def test_invariants_block_deictic_room_reference_without_explicit_lab(self):
        # "in the room" with no explicit lab in the question and no request_lab_name
        # should trigger lab_scope_not_justified, even if resolved from conversation context.
        result = validate_db_execution_invariants(
            question="Is there any anomaly in the room?",
            intent=IntentType.ANOMALY_ANALYSIS_DB,
            selected_metric="ieq",
            resolved_lab_name="smart_lab",  # carried over from context, NOT from this question
            request_lab_name=None,
            explicit_metrics=[],
            hinted_metrics=[],
            planner_hints={},
        )
        self.assertFalse(result["allowed"])
        self.assertIn("lab_scope_not_justified", result["violations"])

    def test_invariants_allow_deictic_when_request_lab_name_provided(self):
        # When the API caller explicitly sends lab_name, deictic reference is fine.
        result = validate_db_execution_invariants(
            question="Is there any anomaly in the room?",
            intent=IntentType.ANOMALY_ANALYSIS_DB,
            selected_metric="ieq",
            resolved_lab_name="smart_lab",
            request_lab_name="smart_lab",  # explicit API parameter
            explicit_metrics=[],
            hinted_metrics=[],
            planner_hints={},
        )
        self.assertTrue(result["allowed"])

    def test_invariants_allow_within_space_comparison(self):
        # Within-space metric comparisons no longer require a second space.
        result = validate_db_execution_invariants(
            question="Compare humidity in smart_lab this morning",
            intent=IntentType.COMPARISON_DB,
            selected_metric="ieq",
            resolved_lab_name="smart_lab",
            request_lab_name="smart_lab",
            explicit_metrics=[],
            hinted_metrics=[],
            planner_hints={
                "query_signals": {
                    "asks_for_db_facts": True,
                    "has_db_scope_phrase": True,
                    "has_metric_reference": False,
                    "has_time_window_hint": False,
                    "has_lab_reference": True,
                    "is_baseline_reference_query": False,
                }
            },
        )
        self.assertTrue(result["allowed"])
        self.assertNotIn("comparison_second_space_not_justified", result["violations"])

    def test_invariants_allow_single_lab_baseline_reference_comparison(self):
        result = validate_db_execution_invariants(
            question="Compare humidity in concrete_lab against its baseline for this morning",
            intent=IntentType.COMPARISON_DB,
            selected_metric="humidity",
            resolved_lab_name="concrete_lab",
            request_lab_name=None,
            explicit_metrics=["humidity"],
            hinted_metrics=[],
            planner_hints={
                "query_signals": {
                    "asks_for_db_facts": True,
                    "has_db_scope_phrase": True,
                    "has_metric_reference": True,
                    "has_time_window_hint": True,
                    "has_lab_reference": True,
                    "is_baseline_reference_query": True,
                }
            },
        )
        self.assertTrue(result["allowed"])

    def test_invariants_allow_single_lab_metric_vs_metric_comparison(self):
        result = validate_db_execution_invariants(
            question="Is PM2.5 or CO2 the bigger issue in smart_lab this month?",
            intent=IntentType.COMPARISON_DB,
            selected_metric="pm25",
            resolved_lab_name="smart_lab",
            request_lab_name=None,
            explicit_metrics=["pm25", "co2"],
            hinted_metrics=[],
            planner_hints={
                "query_signals": {
                    "asks_for_db_facts": True,
                    "has_db_scope_phrase": True,
                    "has_metric_reference": True,
                    "has_time_window_hint": True,
                    "has_lab_reference": True,
                    "is_baseline_reference_query": False,
                }
            },
        )
        self.assertTrue(result["allowed"])
        self.assertNotIn("comparison_second_space_not_justified", result["violations"])

    def test_invariants_allow_two_explicit_labs_without_planner_signals(self):
        result = validate_db_execution_invariants(
            question=(
                "Compare smart_lab and concrete_lab for CO2, PM2.5, VOC, "
                "temperature, and humidity for the last 7 days."
            ),
            intent=IntentType.COMPARISON_DB,
            selected_metric="co2",
            resolved_lab_name="smart_lab",
            request_lab_name=None,
            explicit_metrics=["co2", "pm25", "voc", "temperature", "humidity"],
            hinted_metrics=[],
            planner_hints={"query_signals": {}},
        )
        self.assertTrue(result["allowed"])
        self.assertNotIn("lab_scope_not_justified", result["violations"])
        self.assertNotIn("comparison_second_space_not_justified", result["violations"])

    def test_invariants_block_current_status_without_lab_scope(self):
        result = validate_db_execution_invariants(
            question="Would you say it's good the air quality there?",
            intent=IntentType.CURRENT_STATUS_DB,
            selected_metric="ieq",
            resolved_lab_name=None,
            request_lab_name=None,
            explicit_metrics=[],
            hinted_metrics=[],
            planner_hints={
                "query_signals": {
                    "asks_for_db_facts": True,
                    "has_db_scope_phrase": False,
                    "has_metric_reference": False,
                    "has_time_window_hint": False,
                    "has_lab_reference": False,
                }
            },
        )
        self.assertFalse(result["allowed"])
        self.assertIn("lab_scope_not_justified", result["violations"])

    def test_invariants_block_aggregation_without_lab_scope_even_with_time_window(self):
        result = validate_db_execution_invariants(
            question="What changed in indoor air quality over the last 6 hours?",
            intent=IntentType.AGGREGATION_DB,
            selected_metric="ieq",
            resolved_lab_name=None,
            request_lab_name=None,
            explicit_metrics=[],
            hinted_metrics=["ieq", "co2", "pm25"],
            planner_hints={
                "query_signals": {
                    "asks_for_db_facts": True,
                    "has_db_scope_phrase": True,
                    "has_metric_reference": False,
                    "has_time_window_hint": True,
                    "has_lab_reference": False,
                }
            },
        )
        self.assertFalse(result["allowed"])
        self.assertIn("lab_scope_not_justified", result["violations"])

    def test_invariants_allow_aggregation_when_global_scope_explicit(self):
        result = validate_db_execution_invariants(
            question="What changed in indoor air quality across all labs over the last 6 hours?",
            intent=IntentType.AGGREGATION_DB,
            selected_metric="ieq",
            resolved_lab_name=None,
            request_lab_name=None,
            explicit_metrics=[],
            hinted_metrics=["ieq", "co2", "pm25"],
            planner_hints={
                "query_signals": {
                    "asks_for_db_facts": True,
                    "has_db_scope_phrase": True,
                    "has_metric_reference": False,
                    "has_time_window_hint": True,
                    "has_lab_reference": False,
                }
            },
        )
        self.assertTrue(result["allowed"])

    def test_prepare_db_query_returns_lab_first_clarification_when_lab_missing(self):
        result = prepare_db_query(
            question="Would you say it's good the air quality there?",
            intent=IntentType.CURRENT_STATUS_DB,
            lab_name=None,
            planner_hints={
                "query_signals": {
                    "asks_for_db_facts": True,
                    "has_db_scope_phrase": False,
                    "has_metric_reference": False,
                    "has_time_window_hint": False,
                    "has_lab_reference": False,
                }
            },
        )
        self.assertEqual(result.get("timescale"), "clarify")
        self.assertIn("need the lab first", str(result.get("fallback_answer") or "").lower())

    def test_prepare_db_query_returns_clarification_for_missing_lab_on_air_quality_window(self):
        result = prepare_db_query(
            question="What changed in indoor air quality over the last 6 hours?",
            intent=IntentType.AGGREGATION_DB,
            lab_name=None,
            planner_hints={
                "query_signals": {
                    "asks_for_db_facts": True,
                    "has_db_scope_phrase": True,
                    "has_metric_reference": False,
                    "has_time_window_hint": True,
                    "has_lab_reference": False,
                }
            },
        )
        self.assertEqual(result.get("timescale"), "clarify")
        self.assertIn("need the lab first", str(result.get("fallback_answer") or "").lower())

    def test_invariants_allow_prepositional_scope_for_non_lab_suffix_names(self):
        result = validate_db_execution_invariants(
            question="Which metric is driving poor IEQ in shores_office this week?",
            intent=IntentType.AGGREGATION_DB,
            selected_metric="ieq",
            resolved_lab_name="shores_office",
            request_lab_name=None,
            explicit_metrics=["ieq"],
            hinted_metrics=[],
            planner_hints={
                "query_signals": {
                    "asks_for_db_facts": True,
                    "has_db_scope_phrase": True,
                    "has_metric_reference": True,
                    "has_time_window_hint": True,
                    "has_lab_reference": False,
                }
            },
        )
        self.assertTrue(result["allowed"])

    def test_comparison_co2_query_expands_to_multi_metric_air_quality_pack(self):
        class _Cursor:
            def execute(self, _sql, _params):
                return None

            def fetchall(self):
                return [
                    {
                        "lab_space": "smart_lab",
                        "co2": 415.3,
                        "pm25": 1.2,
                        "voc": 0.08,
                        "humidity": 44.2,
                    },
                    {
                        "lab_space": "concrete_lab",
                        "co2": 418.1,
                        "pm25": 1.4,
                        "voc": 0.09,
                        "humidity": 45.1,
                    },
                ]

            def fetchone(self):
                return None

        result = execute_intent_query(
            cur=_Cursor(),
            question="Compare CO2 levels in smart_lab vs concrete_lab in the last 24 hours",
            intent=IntentType.COMPARISON_DB,
            metric_alias="co2",
            metric_column="co2_avg",
            unit="ppm",
            window_start=datetime(2026, 3, 28, 0, 0, tzinfo=timezone.utc),
            window_end=datetime(2026, 3, 29, 0, 0, tzinfo=timezone.utc),
            window_label="last 24 hours",
            resolved_lab_name=None,
            compared_spaces=[],
            explicit_metrics=["co2"],
            hinted_metrics=[],
        )

        self.assertEqual(result.get("operation_type"), "comparison_multi_metric")
        metrics_used = list(result.get("metrics_used") or [])
        self.assertIn("co2", metrics_used)
        self.assertIn("pm25", metrics_used)
        self.assertIn("voc", metrics_used)

    def test_aggregation_single_air_metric_trend_expands_context_pack(self):
        class _Cursor:
            def execute(self, _sql, _params):
                return None

            def fetchall(self):
                return []

            def fetchone(self):
                return {
                    "lab_space": "smart_lab",
                    "co2": 430.0,
                    "pm25": 3.2,
                    "voc": 0.09,
                    "humidity": 44.8,
                    "ieq": 82.0,
                    "reading_count": 120,
                }

        result = execute_intent_query(
            cur=_Cursor(),
            question="How has CO2 trended this week in smart_lab?",
            intent=IntentType.AGGREGATION_DB,
            metric_alias="co2",
            metric_column="co2_avg",
            unit="ppm",
            window_start=datetime(2026, 3, 22, 0, 0, tzinfo=timezone.utc),
            window_end=datetime(2026, 3, 29, 0, 0, tzinfo=timezone.utc),
            window_label="this week",
            resolved_lab_name="smart_lab",
            compared_spaces=[],
            explicit_metrics=["co2"],
            hinted_metrics=[],
        )
        self.assertEqual(result.get("operation_type"), "aggregation_multi_metric")
        metrics_used = list(result.get("metrics_used") or [])
        self.assertIn("co2", metrics_used)
        self.assertIn("pm25", metrics_used)
        self.assertIn("voc", metrics_used)

    def test_aggregation_multi_metric_without_lab_uses_all_labs_scope(self):
        result = execute_intent_query(
            question="What changed in indoor air quality over the last 6 hours?",
            intent=IntentType.AGGREGATION_DB,
            metric_alias="co2",
            metric_column="co2_avg",
            unit="ppm",
            window_start=datetime(2026, 3, 22, 0, 0, tzinfo=timezone.utc),
            window_end=datetime(2026, 3, 22, 6, 0, tzinfo=timezone.utc),
            window_label="last 6 hours",
            resolved_lab_name=None,
            compared_spaces=[],
            explicit_metrics=["co2"],
            hinted_metrics=[],
        )
        self.assertEqual(result.get("operation_type"), "aggregation_multi_metric")
        self.assertIn("all_labs", str(result.get("fallback_answer") or ""))

    def test_point_lookup_historical_multi_metric_without_lab_uses_all_labs_scope(self):
        result = execute_intent_query(
            question="How was indoor air quality over the last 6 hours?",
            intent=IntentType.POINT_LOOKUP_DB,
            metric_alias="co2",
            metric_column="co2_avg",
            unit="ppm",
            window_start=datetime(2026, 3, 22, 0, 0, tzinfo=timezone.utc),
            window_end=datetime(2026, 3, 22, 6, 0, tzinfo=timezone.utc),
            window_label="last 6 hours",
            resolved_lab_name=None,
            compared_spaces=[],
            explicit_metrics=["co2", "pm25"],
            hinted_metrics=[],
        )
        self.assertEqual(result.get("operation_type"), "aggregation_multi_metric")
        self.assertIn("all_labs", str(result.get("fallback_answer") or ""))

    def test_current_status_co2_returns_point_lookup(self):
        end = datetime(2026, 3, 29, 12, 0, tzinfo=timezone.utc)
        start = end - timedelta(hours=1)
        result = execute_intent_query(
            question="What is the CO2 now in smart_lab?",
            intent=IntentType.CURRENT_STATUS_DB,
            metric_alias="co2",
            metric_column="co2_avg",
            unit="ppm",
            window_start=start,
            window_end=end,
            window_label="last 1 hour",
            resolved_lab_name="smart_lab",
            compared_spaces=[],
            explicit_metrics=["co2"],
            hinted_metrics=[],
        )
        self.assertEqual(result.get("operation_type"), "point_lookup")

    def test_point_lookup_explicit_multi_metric_latest_uses_multi_snapshot(self):
        class _Cursor:
            def execute(self, _sql, _params):
                return None

            def fetchall(self):
                return []

            def fetchone(self):
                return {
                    "lab_space": "smart_lab",
                    "bucket": datetime(2026, 3, 29, 11, 55, tzinfo=timezone.utc),
                    "co2": 422.0,
                    "pm25": 2.1,
                    "voc": 0.08,
                }

        end = datetime(2026, 3, 29, 12, 0, tzinfo=timezone.utc)
        start = end - timedelta(hours=1)
        result = execute_intent_query(
            cur=_Cursor(),
            question="What are the latest readings for CO2, PM2.5, and VOC in smart_lab?",
            intent=IntentType.POINT_LOOKUP_DB,
            metric_alias="co2",
            metric_column="co2_avg",
            unit="ppm",
            window_start=start,
            window_end=end,
            window_label="last 1 hour",
            resolved_lab_name="smart_lab",
            compared_spaces=[],
            explicit_metrics=["co2", "pm25", "voc"],
            hinted_metrics=[],
        )
        self.assertEqual(result.get("operation_type"), "point_lookup_multi_metric")
        metrics_used = list(result.get("metrics_used") or [])
        self.assertIn("co2", metrics_used)
        self.assertIn("pm25", metrics_used)
        self.assertIn("voc", metrics_used)

    def test_point_lookup_last_week_single_metric_returns_window_aggregation(self):
        class _Cursor:
            def execute(self, _sql, _params):
                return None

            def fetchall(self):
                return [
                    {
                        "lab_space": "smart_lab",
                        "avg_value": 6.5,
                        "min_value": 1.1,
                        "max_value": 12.2,
                        "reading_count": 240,
                    }
                ]

            def fetchone(self):
                return None

        start = datetime(2026, 3, 22, 0, 0, tzinfo=timezone.utc)
        end = datetime(2026, 3, 29, 0, 0, tzinfo=timezone.utc)
        result = execute_intent_query(
            cur=_Cursor(),
            question="How was PM2.5 in smart_lab last week?",
            intent=IntentType.POINT_LOOKUP_DB,
            metric_alias="pm25",
            metric_column="pm25_avg",
            unit="ug/m3",
            window_start=start,
            window_end=end,
            window_label="last week",
            resolved_lab_name="smart_lab",
            compared_spaces=[],
            explicit_metrics=["pm25"],
            hinted_metrics=[],
        )
        self.assertEqual(result.get("operation_type"), "aggregation")
        self.assertIn("average", str(result.get("fallback_answer") or "").lower())

    def test_complete_assessment_request_uses_full_environment_metric_pack(self):
        class _Cursor:
            def execute(self, _sql, _params):
                return None

            def fetchall(self):
                return []

            def fetchone(self):
                return {
                    "lab_space": "smart_lab",
                    "ieq": 79.4,
                    "co2": 413.6,
                    "pm25": 2.0,
                    "voc": 0.09,
                    "humidity": 42.0,
                    "temperature": 23.4,
                    "sound": 46.0,
                    "light": 380.0,
                    "reading_count": 24,
                }

        start = datetime(2026, 4, 8, 0, 0, tzinfo=timezone.utc)
        end = datetime(2026, 4, 9, 0, 0, tzinfo=timezone.utc)
        result = execute_intent_query(
            cur=_Cursor(),
            question="give me a complete assessment of the smart lab",
            intent=IntentType.AGGREGATION_DB,
            metric_alias="ieq",
            metric_column="index_value",
            unit="index",
            window_start=start,
            window_end=end,
            window_label="last 24 hours",
            resolved_lab_name="smart_lab",
            compared_spaces=[],
            explicit_metrics=[],
            hinted_metrics=[],
        )
        self.assertEqual(result.get("operation_type"), "aggregation_multi_metric")
        metrics_used = list(result.get("metrics_used") or [])
        self.assertEqual(
            metrics_used[:8],
            ["ieq", "co2", "pm25", "voc", "humidity", "temperature", "sound", "light"],
        )

    def test_comfort_comparison_expands_metrics_beyond_humidity(self):
        from executors.db_support.query_handlers import _requested_metrics

        metrics = _requested_metrics(
            "How does humidity compare with comfort levels today?",
            explicit_metrics=["humidity"],
            hinted_metrics=[],
            intent=IntentType.COMPARISON_DB,
        )
        self.assertIn("humidity", metrics)
        self.assertIn("ieq", metrics)
        self.assertIn("itc", metrics)
        self.assertGreaterEqual(len(metrics), 3)

    def test_comfort_assessment_pack_includes_sound_and_light(self):
        class _Cursor:
            def execute(self, _sql, _params):
                return None

            def fetchall(self):
                return []

            def fetchone(self):
                return {
                    "lab_space": "smart_lab",
                    "ieq": 80.1,
                    "temperature": 23.7,
                    "humidity": 43.4,
                    "co2": 420.2,
                    "pm25": 2.4,
                    "voc": 0.09,
                    "sound": 47.2,
                    "light": 410.0,
                    "reading_count": 24,
                }

        start = datetime(2026, 4, 8, 0, 0, tzinfo=timezone.utc)
        end = datetime(2026, 4, 9, 0, 0, tzinfo=timezone.utc)
        result = execute_intent_query(
            cur=_Cursor(),
            question="Is smart_lab comfortable right now?",
            intent=IntentType.AGGREGATION_DB,
            metric_alias="ieq",
            metric_column="index_value",
            unit="index",
            window_start=start,
            window_end=end,
            window_label="last 24 hours",
            resolved_lab_name="smart_lab",
            compared_spaces=[],
            explicit_metrics=[],
            hinted_metrics=[],
        )
        self.assertEqual(result.get("operation_type"), "aggregation_multi_metric")
        metrics_used = list(result.get("metrics_used") or [])
        self.assertIn("sound", metrics_used)
        self.assertIn("light", metrics_used)

    def test_comparison_handler_runs_single_lab_aggregation(self):
        # Within-space comparison with a single metric falls back to aggregation for that lab.
        class _Cursor:
            def execute(self, _sql, _params):
                return None

            def fetchall(self):
                return []

            def fetchone(self):
                return None

        result = execute_intent_query(
            cur=_Cursor(),
            question="Compare humidity in smart_lab this morning",
            intent=IntentType.COMPARISON_DB,
            metric_alias="humidity",
            metric_column="humidity_avg",
            unit="%",
            window_start=datetime(2026, 3, 28, 0, 0, tzinfo=timezone.utc),
            window_end=datetime(2026, 3, 29, 0, 0, tzinfo=timezone.utc),
            window_label="this morning",
            resolved_lab_name="smart_lab",
            compared_spaces=[],
            explicit_metrics=["humidity"],
            hinted_metrics=[],
        )
        # Should return an aggregation result, not a cross-space error
        self.assertNotIn("need two explicit spaces", str(result.get("fallback_answer") or "").lower())
        self.assertIn("humidity", str(result.get("metrics_used") or "").lower())

    def test_baseline_reference_comparison_runs_single_lab_path(self):
        class _Cursor:
            def execute(self, _sql, _params):
                return None

            def fetchall(self):
                return []

            def fetchone(self):
                return {
                    "current_avg": 51.2,
                    "baseline_avg": 45.0,
                    "baseline_stddev": 2.1,
                    "current_count": 12,
                    "baseline_count": 12,
                }

        result = execute_intent_query(
            cur=_Cursor(),
            question="Compare humidity in concrete_lab against its baseline for this morning",
            intent=IntentType.COMPARISON_DB,
            metric_alias="humidity",
            metric_column="humidity_avg",
            unit="%",
            window_start=datetime(2026, 3, 28, 0, 0, tzinfo=timezone.utc),
            window_end=datetime(2026, 3, 28, 12, 0, tzinfo=timezone.utc),
            window_label="this morning",
            resolved_lab_name="concrete_lab",
            compared_spaces=[],
            explicit_metrics=["humidity"],
            hinted_metrics=[],
        )
        # Handler should correctly route to baseline comparison regardless of data availability
        self.assertIn(
            result.get("operation_type"),
            ("baseline_reference_comparison", "comparison_multi_metric", "comparison"),
        )

    def test_comparison_multi_uses_single_lab_path_for_metric_vs_metric_questions(self):
        class _Cursor:
            def execute(self, _sql, _params):
                return None

            def fetchall(self):
                return []

            def fetchone(self):
                return {
                    "lab_space": "smart_lab",
                    "co2": 430.0,
                    "pm25": 3.2,
                }

        result = execute_intent_query(
            cur=_Cursor(),
            question="Is PM2.5 or CO2 the bigger issue in smart_lab this month?",
            intent=IntentType.COMPARISON_DB,
            metric_alias="pm25",
            metric_column="pm25_avg",
            unit="ug/m3",
            window_start=datetime(2026, 3, 1, 0, 0, tzinfo=timezone.utc),
            window_end=datetime(2026, 3, 31, 0, 0, tzinfo=timezone.utc),
            window_label="this month",
            resolved_lab_name="smart_lab",
            compared_spaces=[],
            explicit_metrics=["pm25", "co2"],
            hinted_metrics=[],
        )
        self.assertEqual(result.get("operation_type"), "comparison_multi_metric")
        self.assertNotIn("need two explicit spaces", str(result.get("fallback_answer") or "").lower())
        self.assertEqual((result.get("rows") or [{}])[0].get("lab_space"), "smart_lab")


if __name__ == "__main__":
    unittest.main()
