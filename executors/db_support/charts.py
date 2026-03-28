"""Chart payload builders for DB executor visualizations."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional


def _serialize_timestamp_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, list):
        return [_serialize_timestamp_value(item) for item in value]
    if isinstance(value, dict):
        return {k: _serialize_timestamp_value(v) for k, v in value.items()}
    return value


def build_line_chart(
    *,
    metric_alias: str,
    unit: str,
    window_label: str,
    series_rows: List[Dict[str, Any]],
    series_name: str,
    lookback_points: int = 72,
) -> Dict[str, Any]:
    recent_rows = (series_rows or [])[-lookback_points:]
    points = [
        {"x": _serialize_timestamp_value(row.get("bucket")), "y": float(row.get("value") or 0.0)}
        for row in recent_rows
        if row.get("bucket") is not None and row.get("value") is not None
    ]
    return {
        "visualization_type": "line",
        "chart": {
            "title": f"{metric_alias.upper()} trend ({window_label})",
            "x_label": "time",
            "y_label": f"{metric_alias.upper()} ({unit})",
            "unit": unit,
            "series": [{"name": series_name, "points": points}],
        },
    }


def build_anomaly_chart(
    *,
    metric_alias: str,
    unit: str,
    window_label: str,
    series_rows: List[Dict[str, Any]],
    anomalies: List[Dict[str, Any]],
    series_name: str,
    lookback_points: int = 72,
) -> Dict[str, Any]:
    recent_rows = (series_rows or [])[-lookback_points:]
    row_keys = {
        (_serialize_timestamp_value(row.get("bucket")), float(row.get("value")))
        for row in recent_rows
        if row.get("bucket") is not None and row.get("value") is not None
    }
    anomaly_keys = {
        (_serialize_timestamp_value(item.get("bucket")), float(item.get("value")))
        for item in anomalies
        if item.get("bucket") is not None and item.get("value") is not None
    }
    base_points = [
        {"x": _serialize_timestamp_value(row.get("bucket")), "y": float(row.get("value"))}
        for row in recent_rows
        if row.get("bucket") is not None and row.get("value") is not None
    ]
    anomaly_points = [{"x": key[0], "y": key[1]} for key in row_keys if key in anomaly_keys]
    anomaly_points.sort(key=lambda point: point["x"])
    return {
        "visualization_type": "line",
        "chart": {
            "title": f"{metric_alias.upper()} anomaly scan ({window_label})",
            "x_label": "time",
            "y_label": f"{metric_alias.upper()} ({unit})",
            "unit": unit,
            "series": [
                {"name": f"{series_name} (all)", "points": base_points},
                {"name": f"{series_name} (anomalies)", "points": anomaly_points},
            ],
            "anomaly_count": len(anomalies),
        },
    }


def build_forecast_chart(
    *,
    metric_alias: str,
    unit: str,
    window_label: str,
    history_rows: List[Dict[str, Any]],
    forecast: Optional[Dict[str, Any]],
    series_name: str,
    lookback_points: int = 72,
) -> Dict[str, Any]:
    recent_history_rows = (history_rows or [])[-lookback_points:]
    history_points = [
        {"x": _serialize_timestamp_value(row.get("bucket")), "y": float(row.get("value") or 0.0)}
        for row in recent_history_rows
        if row.get("bucket") is not None and row.get("value") is not None
    ]
    forecast_points = [
        {"x": _serialize_timestamp_value(row.get("bucket")), "y": float(row.get("value") or 0.0)}
        for row in (forecast or {}).get("forecast_points", [])
        if row.get("bucket") is not None and row.get("value") is not None
    ]
    return {
        "visualization_type": "line",
        "chart": {
            "title": f"{metric_alias.upper()} trend + forecast ({window_label})",
            "x_label": "time",
            "y_label": f"{metric_alias.upper()} ({unit})",
            "unit": unit,
            "series": [
                {"name": f"{series_name} (history)", "points": history_points},
                {"name": f"{series_name} (prediction)", "points": forecast_points},
            ],
        },
    }


def build_bar_chart(
    *,
    metric_alias: str,
    unit: str,
    window_label: str,
    rows: List[Dict[str, Any]],
    value_key: str = "avg_value",
) -> Dict[str, Any]:
    points = [
        {"x": str(row.get("lab_space", "unknown")), "y": float(row.get(value_key) or 0.0)}
        for row in rows
        if row.get(value_key) is not None
    ]
    return {
        "visualization_type": "bar",
        "chart": {
            "title": f"{metric_alias.upper()} comparison ({window_label})",
            "x_label": "space",
            "y_label": f"{metric_alias.upper()} ({unit})",
            "unit": unit,
            "series": [{"name": metric_alias.upper(), "points": points}],
        },
    }


def build_scatter_chart(
    *,
    metric_x: str,
    metric_y: str,
    unit_x: str,
    unit_y: str,
    window_label: str,
    rows: List[Dict[str, Any]],
    correlation: Optional[float],
    lookback_points: int = 72,
) -> Dict[str, Any]:
    points = [
        {"x": float(row.get("x_value")), "y": float(row.get("y_value")), "t": str(row.get("bucket"))}
        for row in rows[-lookback_points:]
        if row.get("x_value") is not None and row.get("y_value") is not None
    ]
    corr_label = "n/a" if correlation is None else f"{correlation:.3f}"
    return {
        "visualization_type": "scatter",
        "chart": {
            "title": f"{metric_x.upper()} vs {metric_y.upper()} ({window_label}, r={corr_label})",
            "x_label": f"{metric_x.upper()} ({unit_x})",
            "y_label": f"{metric_y.upper()} ({unit_y})",
            "series": [{"name": f"{metric_x}_vs_{metric_y}", "points": points}],
            "correlation": correlation,
        },
    }


def empty_chart() -> Dict[str, Any]:
    return {"visualization_type": "none", "chart": None}

