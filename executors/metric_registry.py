"""Metric registry: single source of truth for all supported metrics.

Each entry maps a canonical metric name to its DB column, unit, display name,
and optional threshold context. Aliases map non-canonical names to canonical ones.
"""

from typing import Dict, Any

METRICS: Dict[str, Dict[str, Any]] = {
    "co2": {
        "column": "co2_avg",
        "unit": "ppm",
        "display": "CO2",
        "good_below": 800,
        "warning_below": 1000,
    },
    "pm25": {
        "column": "pm25_avg",
        "unit": "μg/m³",
        "display": "PM2.5",
        "good_below": 12,
        "warning_below": 35,
    },
    "voc": {
        "column": "voc_avg",
        "unit": "ppm",
        "display": "VOC",
        "good_below": 0.5,
        "warning_below": 1.0,
    },
    "temperature": {
        "column": "temp_avg",
        "unit": "°C",
        "display": "Temperature",
        "good_min": 20,
        "good_max": 24,
    },
    "humidity": {
        "column": "humidity_avg",
        "unit": "%",
        "display": "Humidity",
        "good_min": 30,
        "good_max": 60,
    },
    "light": {
        "column": "light_avg",
        "unit": "lux",
        "display": "Light",
    },
    "sound": {
        "column": "sound_avg",
        "unit": "dB",
        "display": "Sound",
        "good_below": 50,
        "warning_below": 65,
    },
    "ieq": {
        "column": "index_value",
        "unit": "index",
        "display": "IEQ",
        "good_above": 75,
        "warning_above": 50,
    },
    "air_contribution": {
        "column": "contri_air",
        "unit": "%",
        "display": "Air Contribution",
    },
    "iaq": {
        "column": "iaq_value",
        "unit": "index",
        "display": "IAQ Sub-index",
        "good_above": 75,
        "warning_above": 50,
    },
    "itc": {
        "column": "itc_value",
        "unit": "index",
        "display": "ITC Sub-index",
        "good_above": 75,
        "warning_above": 50,
    },
    "iac": {
        "column": "iac_value",
        "unit": "index",
        "display": "IAC Sub-index",
        "good_above": 75,
        "warning_above": 50,
    },
    "iil": {
        "column": "iil_value",
        "unit": "index",
        "display": "IIL Sub-index",
        "good_above": 75,
        "warning_above": 50,
    },
}

# Aliases: non-canonical names → canonical names
ALIASES: Dict[str, str] = {
    "pm2.5": "pm25",
    "pm 2.5": "pm25",
    "pm 25": "pm25",
    "tvoc": "voc",
    "temp": "temperature",
    "lux": "light",
    "noise": "sound",
    "index": "ieq",
}


def resolve_metric(name: str) -> str:
    """Resolve a metric name (including aliases) to its canonical form."""
    key = str(name or "").strip().lower()
    return ALIASES.get(key, key)


def metric_column(name: str) -> str | None:
    """Return the DB column for a metric name, or None if unknown."""
    canonical = resolve_metric(name)
    entry = METRICS.get(canonical)
    return entry["column"] if entry else None


def metric_unit(name: str) -> str:
    """Return the display unit for a metric."""
    canonical = resolve_metric(name)
    entry = METRICS.get(canonical, {})
    return entry.get("unit", "")


def metric_display(name: str) -> str:
    """Return the human-readable display name for a metric."""
    canonical = resolve_metric(name)
    entry = METRICS.get(canonical, {})
    return entry.get("display", canonical.upper())


# Backwards-compatible column maps used by query_parsing.py
METRIC_COLUMN_MAP: Dict[str, str] = {}
for _name, _info in METRICS.items():
    METRIC_COLUMN_MAP[_name] = _info["column"]
for _alias, _canonical in ALIASES.items():
    if _canonical in METRICS:
        METRIC_COLUMN_MAP[_alias] = METRICS[_canonical]["column"]

CANONICAL_METRIC_COLUMN_MAP: Dict[str, str] = {
    name: info["column"] for name, info in METRICS.items()
}
