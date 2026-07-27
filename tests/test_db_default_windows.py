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

from fake_sensor_api import FakeSensorApiMixin
from query_routing.intent_classifier import IntentType
from executors.db_support.response_helpers import build_db_payload as _build_db_payload
from executors.db_support.query_parsing import default_window_hours_for_intent as _default_window_hours_for_intent
from executors.db_support.query_handlers import execute_intent_query
from executors.db_support.query_parsing import (
    extract_metric_aliases,
    extract_time_window,
    has_explicit_time_hint,
    pick_metric,
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

    def test_short_span_is_hourly(self):
        self.assertEqual(self._gran(1), 1)
        self.assertEqual(self._gran(24), 1)
        self.assertEqual(self._gran(6 * 24), 1)

    def test_a_week_up_to_a_month_is_still_hourly(self):
        # Aggregation is always 1h — wide ranges are no longer coarsened to 6h.
        self.assertEqual(self._gran(7 * 24), 1)
        self.assertEqual(self._gran(20 * 24), 1)

    def test_a_month_or_more_is_still_hourly(self):
        # Aggregation is always 1h — wide ranges are no longer coarsened to 12h.
        self.assertEqual(self._gran(28 * 24), 1)
        self.assertEqual(self._gran(31 * 24), 1)
        self.assertEqual(self._gran(120 * 24), 1)

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

    def test_spelled_out_ordinal_day_range(self):
        # "the second of June until the fourth of June" must not collapse to June 1.
        start, end, _ = extract_time_window(
            "Give me the pm2.5 from the second of june until the fourth of june 2026"
        )
        self.assertEqual((start.month, start.day), (6, 2))
        # End day inclusive → covers through all of June 4.
        self.assertEqual((end.month, end.day), (6, 5))

    def test_spelled_out_ordinal_month_first_form(self):
        start, end, _ = extract_time_window("pm2.5 from june second to june fourth 2026")
        self.assertEqual((start.month, start.day), (6, 2))
        self.assertEqual((end.month, end.day), (6, 5))

    def test_spelled_out_compound_ordinal_single_day(self):
        start, end, _ = extract_time_window("humidity on the twenty-first of june 2026")
        self.assertEqual((start.month, start.day), (6, 21))
        self.assertEqual((end - start).days, 1)

    def test_spelled_out_ordinal_cross_month_range(self):
        start, end, _ = extract_time_window(
            "co2 from the twenty-eighth of may until the second of june 2026"
        )
        self.assertEqual((start.month, start.day), (5, 28))
        self.assertEqual((end.month, end.day), (6, 3))


class DbDefaultWindowTests(FakeSensorApiMixin, unittest.TestCase):
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
        from executors.db_support.metric_planning import plan_metrics

        metrics = plan_metrics(
            question="How does humidity compare with comfort levels today?",
            explicit_metrics=["humidity"],
            hinted_metrics=[],
            intent=IntentType.COMPARISON_DB,
        ).selected
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


class IeqIndexQueryTests(FakeSensorApiMixin, unittest.TestCase):
    """An explicit IEQ-index ask reports the IEQ composite + sub-indices, not CO2."""

    def test_plan_returns_ieq_family_not_pollutant_pack(self):
        from executors.db_support.metric_planning import plan_metrics

        metrics = plan_metrics(
            question="Give me the IEQ data from May 1st to May 8th",
            explicit_metrics=["ieq"],
            hinted_metrics=["ieq", "co2", "pm25"],
            intent=IntentType.AGGREGATION_DB,
        ).selected
        # IEQ index leads, with its sub-indices — no CO2/PM2.5 pollutant pack.
        self.assertEqual(metrics[0], "ieq")
        for sub in ("iaq", "itc", "iac", "iil"):
            self.assertIn(sub, metrics)
        self.assertNotIn("co2", metrics)
        self.assertNotIn("pm25", metrics)

    def test_air_quality_query_is_unaffected(self):
        from executors.db_support.metric_planning import plan_metrics

        metrics = plan_metrics(
            question="How is the air quality from May 1st to May 8th?",
            explicit_metrics=[],
            hinted_metrics=[],
            intent=IntentType.AGGREGATION_DB,
        ).selected
        self.assertIn("co2", metrics)
        self.assertIn("ieq", metrics)

    def test_ieq_query_is_not_treated_as_air_quality(self):
        from executors.db_support.response_helpers import (
            is_air_quality_query_text,
            is_ieq_index_query_text,
        )

        self.assertTrue(is_ieq_index_query_text("Give me the IEQ data from May 1st to May 8th"))
        self.assertFalse(is_air_quality_query_text("Give me the IEQ data from May 1st to May 8th"))
        # "air quality" remains a pollutant air-quality query.
        self.assertTrue(is_air_quality_query_text("How is the air quality today?"))

    def test_ieq_directive_leads_with_ieq_not_co2(self):
        from executors.db_support.response_helpers import db_response_directive

        directive = db_response_directive(
            IntentType.AGGREGATION_DB,
            question="Give me the IEQ data from May 1st to May 8th",
        )
        self.assertIn("LEAD WITH THE IEQ INDEX SCORE", directive)
        self.assertIn("Do NOT lead with CO2", directive)


class DiagnosticSignalTests(FakeSensorApiMixin, unittest.TestCase):
    """The LLM router's analysis_mode=diagnostic drives root-cause decomposition,
    independent of question phrasing — so 'main driver making the IEQ bad' (which
    the keyword heuristic misses) still pulls every contributing metric."""

    def test_diagnostic_hint_pulls_full_pack_for_unmatched_phrasing(self):
        from executors.db_support.metric_planning import plan_metrics

        # This phrasing is NOT caught by the is_diagnostic_query_text heuristic.
        question = "What is the main driver making the IEQ bad?"
        from executors.db_support.response_helpers import is_diagnostic_query_text

        self.assertFalse(is_diagnostic_query_text(question))

        # Without the LLM hint it collapses to the IEQ-index family only.
        without_hint = plan_metrics(
            question=question,
            explicit_metrics=[],
            hinted_metrics=["ieq"],
            intent=IntentType.CURRENT_STATUS_DB,
        ).selected
        self.assertNotIn("co2", without_hint)

        # With the LLM diagnostic hint it pulls the full contributing pack.
        with_hint = plan_metrics(
            question=question,
            explicit_metrics=[],
            hinted_metrics=["ieq"],
            intent=IntentType.CURRENT_STATUS_DB,
            is_diagnostic=True,
        ).selected
        for metric in ("co2", "pm25", "voc", "humidity", "temperature", "ieq"):
            self.assertIn(metric, with_hint)

    def test_diagnostic_hint_routes_to_diagnostic_handler(self):
        from executors.db_support import query_handlers as qh

        captured = {}
        orig = qh._handle_diagnostic

        def _spy(**kwargs):
            captured["is_diagnostic"] = kwargs.get("is_diagnostic")
            return None  # let the chain continue; we only assert the gate value

        qh._handle_diagnostic = _spy
        try:
            qh.execute_intent_query(
                question="What is the main driver making the IEQ bad?",
                intent=IntentType.CURRENT_STATUS_DB,
                metric_alias="ieq",
                metric_column="ieq_index",
                unit="index",
                window_start=datetime(2025, 1, 1),
                window_end=datetime(2025, 1, 2),
                window_label="last 24 hours",
                resolved_lab_name=None,
                compared_spaces=[],
                explicit_metrics=[],
                hinted_metrics=["ieq"],
                diagnostic_hint=True,
            )
        finally:
            qh._handle_diagnostic = orig
        self.assertTrue(captured.get("is_diagnostic"))

    def test_directive_follows_executed_diagnostic_operation(self):
        from executors.db_support.response_helpers import db_response_directive

        # Even when the question text would not trip the heuristic, an explicit
        # diagnostic flag yields the diagnostic directive.
        directive = db_response_directive(
            IntentType.CURRENT_STATUS_DB,
            question="What is the main driver making the IEQ bad?",
            diagnostic=True,
        )
        from prompting.db_prompts import DB_TOOL_RESPONSE_DIRECTIVE_DIAGNOSTIC

        self.assertEqual(directive, DB_TOOL_RESPONSE_DIRECTIVE_DIAGNOSTIC)


class IeqCompositeFanoutTests(FakeSensorApiMixin, unittest.TestCase):
    """A plain IEQ question (no snapshot noun, not diagnostic) must still fan out to
    the IEQ composite's sub-indices rather than collapsing to a single IEQ value."""

    def test_plain_ieq_question_pulls_sub_indices(self):
        from executors.db_support import query_handlers as qh
        from executors.db_support import api_client

        captured = {}

        def _fake_point_row(slug, metrics):
            captured["metrics"] = list(metrics)
            return {m: 50.0 for m in metrics}

        orig = api_client.fetch_multi_metric_point_row
        api_client.fetch_multi_metric_point_row = _fake_point_row
        try:
            result = qh.execute_intent_query(
                question="Is the IEQ good?",  # no "current/now/reading/value/level" snapshot noun
                intent=IntentType.CURRENT_STATUS_DB,
                metric_alias="ieq",
                metric_column="index_value",
                unit="index",
                window_start=datetime(2025, 1, 1, tzinfo=timezone.utc),
                window_end=datetime(2025, 1, 2, tzinfo=timezone.utc),
                window_label="last 24 hours",
                resolved_lab_name="smart_lab",
                compared_spaces=[],
                explicit_metrics=["ieq"],
                hinted_metrics=["ieq"],
            )
        finally:
            api_client.fetch_multi_metric_point_row = orig

        self.assertEqual(result.get("operation_type"), "point_lookup_multi_metric")
        metrics_used = list(result.get("metrics_used") or [])
        self.assertEqual(metrics_used[0], "ieq")
        for sub in ("iaq", "itc", "iac", "iil"):
            self.assertIn(sub, metrics_used)


class RouterAnalysisModeTests(unittest.TestCase):
    """The router carries the LLM's analysis_mode through to the RoutePlan, and the
    emergency keyword fallback recovers it only when the LLM is unreachable."""

    def test_parser_extracts_diagnostic_mode(self):
        import json
        from query_routing.llm_router_planner import _parse_llm_response

        raw = json.dumps(
            {"intent": "current_status_db", "metrics": ["ieq"], "confidence": 0.9, "analysis_mode": "diagnostic"}
        )
        plan = _parse_llm_response(raw, "what is the main driver making the IEQ bad?", None)
        self.assertEqual(plan.analysis_mode, "diagnostic")

    def test_parser_defaults_to_none(self):
        import json
        from query_routing.llm_router_planner import _parse_llm_response

        for value in ({}, {"analysis_mode": "bogus"}, {"analysis_mode": None}):
            payload = {"intent": "current_status_db", "metrics": ["ieq"], "confidence": 0.9, **value}
            plan = _parse_llm_response(json.dumps(payload), "what is the IEQ?", None)
            self.assertIsNone(plan.analysis_mode)

    def test_fallback_recovers_diagnostic_via_keyword_heuristic(self):
        from query_routing.llm_router_planner import _fallback_plan

        self.assertEqual(_fallback_plan("why is the IEQ low?", None).analysis_mode, "diagnostic")
        self.assertIsNone(_fallback_plan("what is the IEQ?", None).analysis_mode)


if __name__ == "__main__":
    unittest.main()
