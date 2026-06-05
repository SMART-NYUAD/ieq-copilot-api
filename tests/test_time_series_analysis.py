"""Tests for time-series analysis helpers."""

import unittest

from executors.db_support.response_helpers import (
    build_backend_semantic_state,
    build_time_series_analysis,
    enrich_backend_semantic_state,
    normalize_series_rows,
)


def _hourly_rows(values):
    return [
        {"lab_space": "smart_lab", "bucket": f"2026-06-01T{hour:02d}:00:00+04:00", "value": value}
        for hour, value in enumerate(values)
    ]


class TimeSeriesAnalysisTests(unittest.TestCase):
    def test_normalize_series_rows_filters_metric(self):
        rows = [
            {"metric": "co2", "bucket": "2026-06-01T08:00:00+04:00", "value": 400.0},
            {"metric": "pm25", "bucket": "2026-06-01T08:00:00+04:00", "value": 2.0},
        ]
        points = normalize_series_rows(rows, "co2")
        self.assertEqual(len(points), 1)
        self.assertEqual(points[0]["value"], 400.0)

    def test_build_time_series_analysis_detects_spike(self):
        values = [400.0, 405.0, 402.0, 404.0, 403.0, 401.0, 406.0, 500.0]
        analysis = build_time_series_analysis(
            series_rows=_hourly_rows(values),
            metric_alias="co2",
            unit="ppm",
            api_trend_pct=9.5,
        )
        self.assertIn("window_stats", analysis)
        self.assertEqual(analysis["window_stats"]["max"], 500.0)
        self.assertIn("change_analysis", analysis)
        self.assertEqual(analysis["change_analysis"]["direction"], "rising")
        self.assertEqual(analysis["change_analysis"]["api_trend_pct"], 9.5)
        self.assertGreaterEqual(len(analysis["time_series"]["points"]), 8)
        self.assertTrue(analysis["notable_events"])

    def test_build_backend_semantic_state_omits_empty(self):
        self.assertIsNone(build_backend_semantic_state({}))
        state = build_backend_semantic_state(
            {
                "window_stats": {"mean": 1.0},
                "change_analysis": {"direction": "stable"},
                "notable_events": [],
                "time_series": {"points": []},
            }
        )
        self.assertIn("window_stats", state)
        self.assertNotIn("time_series", state)

    def test_enrich_skips_authoritative_bounds_for_multi_metric_ops(self):
        analysis = build_time_series_analysis(
            series_rows=_hourly_rows([50.0, 51.0, 52.0, 73.0]),
            metric_alias="ieq",
            unit="score",
        )
        state = enrich_backend_semantic_state(
            analysis,
            operation_type="comparison_multi_metric",
            metrics_used=["humidity", "ieq"],
        )
        self.assertIsNotNone(state)
        self.assertNotIn("authoritative_bounds", state)

    def test_enrich_adds_authoritative_bounds_for_single_metric_timeseries(self):
        analysis = build_time_series_analysis(
            series_rows=_hourly_rows([50.0, 51.0, 52.0, 49.0]),
            metric_alias="humidity",
            unit="%RH",
        )
        state = enrich_backend_semantic_state(
            analysis,
            operation_type="timeseries",
            metrics_used=["humidity"],
        )
        self.assertIn("authoritative_bounds", state)
        self.assertEqual(state["authoritative_bounds"]["metric"], "humidity")


if __name__ == "__main__":
    unittest.main()
