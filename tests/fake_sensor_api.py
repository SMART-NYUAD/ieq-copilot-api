"""Deterministic stand-in for the Smart CRG sensor REST API.

Tests must not depend on a reachable sensor host: a test that talks to the live API
fails off-network, and quietly changes meaning whenever the real readings change.

This stubs the HTTP layer (``api_client._get_client``) rather than the individual
``fetch_*`` helpers, so the row-conversion, merging, and aggregation logic in
``api_client`` still runs for real — only the network is replaced.

Usage::

    class MyTests(FakeSensorApiMixin, unittest.TestCase):
        ...

``FakeSensorApiMixin`` also blocks outbound sockets for the duration of each test, so a
call path that bypasses the stub fails loudly instead of silently reaching the network.
"""

from __future__ import annotations

import socket
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import patch

from executors.db_support import api_client


_BASE_TIME = datetime(2026, 6, 1, tzinfo=timezone.utc)

# Deterministic per-metric values. Chosen to be plainly synthetic so a test asserting on
# them reads as "the stub said so", never as a claim about the real building.
_METRIC_VALUES: Dict[str, float] = {
    "co2": 430.0,
    "pm25": 3.2,
    "temperature": 22.5,
    "humidity": 45.0,
    "voc": 120.0,
    "light": 300.0,
    "noise": 38.0,
}
_SCORE_VALUES: Dict[str, float] = {"IEQ": 82.0, "IAQ": 78.0, "ITC": 90.0, "IAC": 85.0, "IIL": 70.0}

SPACES: List[Dict[str, Any]] = [
    {"slug": "smart_lab", "name": "Smart Lab", "metrics": {"itc": 90, "iil": 70, "iaq": 78, "iac": 85}},
    {"slug": "concrete_lab", "name": "Concrete Lab", "metrics": {"itc": 80, "iil": 65, "iaq": 70, "iac": 75}},
]


def _hourly_series(value: float, hours: int = 24, key: str = "timestamp", value_key: str = "agg_value"):
    return [
        {
            key: (_BASE_TIME + timedelta(hours=offset)).isoformat(),
            value_key: round(value + (offset % 3) * 0.5, 3),
        }
        for offset in range(hours)
    ]


class _FakeResponse:
    def __init__(self, payload: Dict[str, Any]):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Dict[str, Any]:
        return self._payload


class FakeSensorApiClient:
    """Routes ``GET`` calls by URL shape and returns canned, well-formed payloads."""

    def __init__(self, empty: bool = False):
        self.empty = empty
        self.requests: List[Dict[str, Any]] = []

    def get(self, url: str, params: Optional[Dict[str, Any]] = None, headers=None) -> _FakeResponse:
        self.requests.append({"url": url, "params": dict(params or {})})
        return _FakeResponse(self._payload_for(url, params or {}))

    def _payload_for(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if url.rstrip("/").endswith("/spaces"):
            return {"spaces": [] if self.empty else SPACES}

        if url.endswith("/heatmap/metrics"):
            if self.empty:
                return {"success": True, "devices": []}
            return {
                "success": True,
                "devices": [
                    {
                        "device_id": "sensor-1",
                        "name": "Sensor 1",
                        "metrics": [
                            {
                                "type": metric,
                                "latest_value": value,
                                "unit": "ppm" if metric == "co2" else "",
                                "latest_timestamp": datetime.now(tz=timezone.utc).isoformat(),
                            }
                            for metric, value in (("co2", 430.0), ("temperature", 22.5))
                        ],
                    },
                    {
                        "device_id": "sensor-2",
                        "name": "Sensor 2",
                        "metrics": [
                            {
                                "type": "temperature",
                                "latest_value": 25.5,
                                "unit": "",
                                # Deliberately stale, so offline/faulty paths have data.
                                "latest_timestamp": (
                                    datetime.now(tz=timezone.utc) - timedelta(days=3)
                                ).isoformat(),
                            }
                        ],
                    },
                ],
            }

        if url.endswith("/metrics"):
            if self.empty:
                return {"success": True, "data": {"space": None}}
            slug = url.split("/spaces/")[1].split("/")[0]
            space = next((s for s in SPACES if s["slug"] == slug), SPACES[0])
            return {
                "success": True,
                "data": {
                    "space": {
                        **space,
                        "last_updated": _BASE_TIME.isoformat(),
                        "ieq": {"score": _SCORE_VALUES["IEQ"]},
                        "avg_metrics": [
                            {"type": metric, "avg_value": value}
                            for metric, value in _METRIC_VALUES.items()
                        ],
                    }
                },
            }

        if url.endswith("/agg-summary"):
            metric = url.split("/metrics/")[1].split("/")[0]
            value = _METRIC_VALUES.get(metric, 1.0)
            readings = [] if self.empty else _hourly_series(value)
            return {
                "success": True,
                "data": {
                    "aggregate_readings": readings,
                    "avg_agg_value": None if self.empty else value,
                    "min_agg_value": None if self.empty else value - 1,
                    "max_agg_value": None if self.empty else value + 1,
                },
            }

        if url.endswith("/indoor-data"):
            score = str(params.get("type") or "IEQ")
            value = _SCORE_VALUES.get(score, 75.0)
            readings = [] if self.empty else _hourly_series(value, key="bucket", value_key="value")
            return {"success": True, "data": {"readings": readings}}

        if url.endswith("/predictions"):
            metric = url.split("/metrics/")[1].split("/")[0]
            value = _METRIC_VALUES.get(metric, 1.0)
            if self.empty:
                return {"success": True, "data": {"predictions": []}}
            return {
                "success": True,
                "data": {
                    "predictions": [
                        {
                            "timestamp": (_BASE_TIME + timedelta(hours=offset)).isoformat(),
                            "predicted_value": value + offset,
                        }
                        for offset in range(1, 7)
                    ]
                },
            }

        return {"success": False}


def _blocked_connect(*_args, **_kwargs):
    raise AssertionError(
        "test attempted a real network connection — extend FakeSensorApiClient instead"
    )


class FakeSensorApiMixin:
    """TestCase mixin: stub the sensor API and fail on any real network call."""

    fake_api_empty = False

    def setUp(self):  # noqa: D102 - unittest hook
        super().setUp()
        api_client._RESPONSE_CACHE.clear()
        self.fake_api = FakeSensorApiClient(empty=self.fake_api_empty)
        patches = [
            patch.object(api_client, "_get_client", return_value=self.fake_api),
            patch.object(socket.socket, "connect", _blocked_connect),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(api_client._RESPONSE_CACHE.clear)
