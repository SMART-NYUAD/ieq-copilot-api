"""Latest sensor readings for a single space."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from executors import metric_registry
from executors.db_support import api_client

_SENSOR_METRICS = ["co2", "pm25", "voc", "humidity", "temperature", "light", "sound"]

_UNIT_OVERRIDES: Dict[str, str] = {
    "pm25": "ug/m3",
    "temperature": "degC",
}

_CORE_METRICS = _SENSOR_METRICS + ["ieq"]


def _metric_unit(metric: str) -> str:
    return _UNIT_OVERRIDES.get(metric, metric_registry.metric_unit(metric))


def _metric_status(metric: str, value: float) -> str:
    entry = metric_registry.METRICS.get(metric, {})
    if "good_below" in entry and "warning_below" in entry:
        if value < entry["good_below"]:
            return "good"
        if value < entry["warning_below"]:
            return "warning"
        return "critical"
    if "good_min" in entry and "good_max" in entry:
        if entry["good_min"] <= value <= entry["good_max"]:
            return "good"
        return "warning"
    if "good_above" in entry and "warning_above" in entry:
        if value >= entry["good_above"]:
            return "good"
        if value >= entry["warning_above"]:
            return "warning"
        return "critical"
    return "good"


def _reading(metric: str, value: Optional[float]) -> Dict[str, Any]:
    v = float(value) if value is not None else 0.0
    return {
        "value": v,
        "unit": _metric_unit(metric),
        "status": _metric_status(metric, v),
    }


def get_sensor_latest(space: str) -> Dict[str, Any]:
    space_data = api_client.fetch_space_metrics(space) or {}
    last_updated = space_data.get("last_updated")
    ieq_value = (space_data.get("ieq") or {}).get("score")

    window_end = datetime.now(tz=timezone.utc)
    window_start = window_end - timedelta(hours=1)
    sensor_values: Dict[str, Optional[float]] = {}
    for metric in _SENSOR_METRICS:
        agg = api_client.fetch_metric_agg_summary(space, metric, window_start, window_end, interval_hours=1)
        sensor_values[metric] = (agg or {}).get("avg_agg_value")

    readings: Dict[str, Any] = {}
    for metric in _SENSOR_METRICS:
        readings[metric] = _reading(metric, sensor_values.get(metric))
    readings["ieq"] = _reading("ieq", ieq_value)

    return {
        "space": space,
        "timestamp": last_updated,
        "readings": readings,
    }
