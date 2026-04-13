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
from executors.db_query_executor import _build_db_payload, _default_window_hours_for_intent, prepare_db_query
from executors.db_support.charts import build_forecast_chart
from executors.db_support.query_handlers import execute_intent_query
from executors.db_support.query_parsing import (
    extract_metric_aliases,
    extract_time_window,
    pick_metric,
    strip_conversation_context,
    validate_db_execution_invariants,
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

    def test_extract_time_window_uses_current_question_scope(self):
        start, end, label = extract_time_window("pm2.5 in smart lab for the last hour", default_hours=24)
        self.assertEqual(label, "last 1 hour")
        self.assertAlmostEqual((end - start).total_seconds(), 3600.0, delta=2.0)

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

    def test_invariants_block_comparison_without_explicit_second_space(self):
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
        self.assertFalse(result["allowed"])
        self.assertIn("comparison_second_space_not_justified", result["violations"])

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

    def test_invariants_allow_forecast_with_default_metric_and_window(self):
        result = validate_db_execution_invariants(
            question="Predict next day in smart_lab",
            intent=IntentType.FORECAST_DB,
            selected_metric="pm25",
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
                "Compare smart_lab and concrete_lab for CO2, PM2.5, TVOC, "
                "temperature, and humidity for the last 7 days."
            ),
            intent=IntentType.COMPARISON_DB,
            selected_metric="co2",
            resolved_lab_name="smart_lab",
            request_lab_name=None,
            explicit_metrics=["co2", "pm25", "tvoc", "temperature", "humidity"],
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

    def test_forecast_handler_keeps_requested_window_for_chart_and_metadata(self):
        class _Cursor:
            def __init__(self):
                self.calls = 0
                self._rows = [
                    [  # model history query rows
                        {"bucket": datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc), "value": 10.0},
                        {"bucket": datetime(2026, 1, 2, 0, 0, tzinfo=timezone.utc), "value": 12.0},
                    ],
                    [  # requested window rows
                        {"bucket": datetime(2026, 3, 27, 0, 0, tzinfo=timezone.utc), "value": 15.0},
                        {"bucket": datetime(2026, 3, 28, 0, 0, tzinfo=timezone.utc), "value": 14.0},
                    ],
                ]

            def execute(self, _sql, _params):
                self.calls += 1

            def fetchall(self):
                idx = max(0, min(self.calls - 1, len(self._rows) - 1))
                return self._rows[idx]

            def fetchone(self):
                return None

        cur = _Cursor()
        start = datetime(2026, 3, 27, 0, 0, tzinfo=timezone.utc)
        end = datetime(2026, 3, 29, 0, 0, tzinfo=timezone.utc)
        result = execute_intent_query(
            cur=cur,
            question="Forecast PM2.5 next day in smart_lab",
            intent=IntentType.FORECAST_DB,
            metric_alias="pm25",
            metric_column="pm25_avg",
            unit="ug/m3",
            window_start=start,
            window_end=end,
            window_label="last 24 hours",
            resolved_lab_name="smart_lab",
            compared_spaces=[],
            explicit_metrics=["pm25"],
            hinted_metrics=[],
            max_chart_lookback_points=72,
        )
        chart_title = (((result.get("chart_payload") or {}).get("chart") or {}).get("title") or "")
        self.assertIn("last 24 hours", chart_title)
        # Forecast handler should keep requested external window fields.
        self.assertEqual(result.get("window_start"), start)
        self.assertEqual(result.get("window_end"), end)

    def test_forecast_handler_queries_latest_points_not_oldest(self):
        class _Cursor:
            def __init__(self):
                self.calls = []

            def execute(self, sql, _params):
                self.calls.append(str(sql))

            def fetchall(self):
                return []

            def fetchone(self):
                return None

        cur = _Cursor()
        start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
        end = datetime(2026, 3, 29, 0, 0, tzinfo=timezone.utc)
        execute_intent_query(
            cur=cur,
            question="Forecast PM2.5 in smart_lab next day",
            intent=IntentType.FORECAST_DB,
            metric_alias="pm25",
            metric_column="pm25_avg",
            unit="ug/m3",
            window_start=start,
            window_end=end,
            window_label="last 24 hours",
            resolved_lab_name="smart_lab",
            compared_spaces=[],
            explicit_metrics=["pm25"],
            hinted_metrics=[],
            max_chart_lookback_points=72,
        )
        self.assertGreaterEqual(len(cur.calls), 1)
        forecast_sql = cur.calls[0].lower()
        self.assertIn("order by bucket desc", forecast_sql)

    def test_forecast_chart_uses_target_timezone_for_history_and_prediction(self):
        history_rows = [
            {"bucket": datetime(2026, 3, 29, 9, 0, tzinfo=timezone.utc), "value": 1.0},
        ]
        forecast = {
            "forecast_points": [
                {"bucket": datetime(2026, 3, 29, 14, 0, tzinfo=timezone(timedelta(hours=4))), "value": 2.0}
            ]
        }
        chart = build_forecast_chart(
            metric_alias="pm25",
            unit="ug/m3",
            window_label="last 24 hours + next 24 hour(s)",
            history_rows=history_rows,
            forecast=forecast,
            series_name="smart_lab",
            lookback_points=0,
        )
        series = ((chart.get("chart") or {}).get("series") or [])
        history_x = series[0]["points"][0]["x"]
        forecast_x = series[1]["points"][0]["x"]
        self.assertTrue(history_x.endswith("+04:00"))
        self.assertTrue(forecast_x.endswith("+04:00"))

    def test_forecast_chart_with_zero_lookback_keeps_full_requested_history(self):
        start = datetime(2026, 3, 25, 0, 0, tzinfo=timezone.utc)
        history_rows = [
            {"bucket": start + timedelta(hours=hour), "value": float(hour)}
            for hour in range(100)
        ]
        chart = build_forecast_chart(
            metric_alias="pm25",
            unit="ug/m3",
            window_label="last 100 hours + next 12 hour(s)",
            history_rows=history_rows,
            forecast={"forecast_points": []},
            series_name="smart_lab",
            lookback_points=0,
        )
        series = ((chart.get("chart") or {}).get("series") or [])
        self.assertEqual(len(series[0]["points"]), 100)

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
                        "tvoc": 0.08,
                        "humidity": 44.2,
                    },
                    {
                        "lab_space": "concrete_lab",
                        "co2": 418.1,
                        "pm25": 1.4,
                        "tvoc": 0.09,
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
            max_chart_lookback_points=0,
        )

        self.assertEqual(result.get("operation_type"), "comparison_multi_metric")
        metrics_used = list(result.get("metrics_used") or [])
        self.assertIn("co2", metrics_used)
        self.assertIn("pm25", metrics_used)
        self.assertIn("tvoc", metrics_used)

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
                    "tvoc": 0.09,
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
            max_chart_lookback_points=0,
        )
        self.assertEqual(result.get("operation_type"), "aggregation_multi_metric")
        metrics_used = list(result.get("metrics_used") or [])
        self.assertIn("co2", metrics_used)
        self.assertIn("pm25", metrics_used)
        self.assertIn("tvoc", metrics_used)

    def test_current_status_retries_last_six_hours_when_last_hour_empty(self):
        class _Cursor:
            def __init__(self):
                self.fetchone_calls = 0

            def execute(self, _sql, _params):
                return None

            def fetchall(self):
                return []

            def fetchone(self):
                self.fetchone_calls += 1
                if self.fetchone_calls == 1:
                    return None
                return {
                    "lab_space": "smart_lab",
                    "bucket": datetime(2026, 3, 29, 11, 15, tzinfo=timezone.utc),
                    "value": 418.0,
                }

        end = datetime(2026, 3, 29, 12, 0, tzinfo=timezone.utc)
        start = end - timedelta(hours=1)
        result = execute_intent_query(
            cur=_Cursor(),
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
            max_chart_lookback_points=0,
        )
        self.assertEqual(result.get("operation_type"), "point_lookup")
        self.assertEqual(result.get("window_label"), "last 6 hours")
        self.assertIn("hour(s) old", str(result.get("fallback_answer") or ""))

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
                    "tvoc": 0.08,
                }

        end = datetime(2026, 3, 29, 12, 0, tzinfo=timezone.utc)
        start = end - timedelta(hours=1)
        result = execute_intent_query(
            cur=_Cursor(),
            question="What are the latest readings for CO2, PM2.5, and TVOC in smart_lab?",
            intent=IntentType.POINT_LOOKUP_DB,
            metric_alias="co2",
            metric_column="co2_avg",
            unit="ppm",
            window_start=start,
            window_end=end,
            window_label="last 1 hour",
            resolved_lab_name="smart_lab",
            compared_spaces=[],
            explicit_metrics=["co2", "pm25", "tvoc"],
            hinted_metrics=[],
            max_chart_lookback_points=0,
        )
        self.assertEqual(result.get("operation_type"), "point_lookup_multi_metric")
        metrics_used = list(result.get("metrics_used") or [])
        self.assertIn("co2", metrics_used)
        self.assertIn("pm25", metrics_used)
        self.assertIn("tvoc", metrics_used)

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
            max_chart_lookback_points=0,
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
                    "tvoc": 0.09,
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
            max_chart_lookback_points=0,
        )
        self.assertEqual(result.get("operation_type"), "aggregation_multi_metric")
        metrics_used = list(result.get("metrics_used") or [])
        self.assertEqual(
            metrics_used[:8],
            ["ieq", "co2", "pm25", "tvoc", "humidity", "temperature", "sound", "light"],
        )

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
                    "tvoc": 0.09,
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
            max_chart_lookback_points=0,
        )
        self.assertEqual(result.get("operation_type"), "aggregation_multi_metric")
        metrics_used = list(result.get("metrics_used") or [])
        self.assertIn("sound", metrics_used)
        self.assertIn("light", metrics_used)

    def test_comparison_handler_never_falls_back_to_global_two_labs(self):
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
            max_chart_lookback_points=0,
        )
        self.assertIn("need two explicit spaces", str(result.get("fallback_answer") or "").lower())

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
            max_chart_lookback_points=0,
        )
        self.assertEqual(result.get("operation_type"), "baseline_reference_comparison")
        self.assertEqual((result.get("rows") or [{}])[0].get("lab_space"), "concrete_lab")

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
            max_chart_lookback_points=0,
        )
        self.assertEqual(result.get("operation_type"), "comparison_multi_metric")
        self.assertNotIn("need two explicit spaces", str(result.get("fallback_answer") or "").lower())
        self.assertEqual((result.get("rows") or [{}])[0].get("lab_space"), "smart_lab")


if __name__ == "__main__":
    unittest.main()
