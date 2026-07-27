"""HTTP client for the Smart CRG REST API — replaces direct DB queries."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

import httpx

from core_settings import predictions_api_base_url, sensor_api_base_url
from executors.db_support.time_windows import granularity_hours_for_window

# Host URLs resolve through core_settings (``SENSOR_API_BASE_URL`` for latest readings /
# agg-summaries, ``PREDICTIONS_API_BASE_URL`` for forecasts) so they are configurable per
# environment instead of hardcoded.
_MAX_CONCURRENT_API_CALLS = 6
_MAX_CACHE_ENTRIES = 512
_T = TypeVar("_T")

_METRICS_CACHE_TTL_SECONDS = 45.0
_SPACES_CACHE_TTL_SECONDS = 300.0
_AGG_CACHE_TTL_SECONDS = 45.0
_INDOOR_CACHE_TTL_SECONDS = 45.0
_PREDICTIONS_CACHE_TTL_SECONDS = 120.0
_HEATMAP_CACHE_TTL_SECONDS = 30.0

_CLIENT: Optional[httpx.Client] = None
_RESPONSE_CACHE: Dict[str, Tuple[float, Any]] = {}

# Sensor metrics available via /metrics/{type}/agg-summary
_SENSOR_METRICS = {"co2", "humidity", "light", "pm25", "temperature", "voc"}
# "sound" in this codebase == "noise" in the API
_SOUND_ALIASES = {"sound", "noise"}
# Legacy alias "tvoc" maps to API slug "voc"
_VOC_LEGACY_ALIASES = {"tvoc"}
# Score metrics available via /indoor-data?type=...
_SCORE_METRIC_MAP: Dict[str, str] = {
    "ieq": "IEQ",
    "iaq": "IAQ",
    "itc": "ITC",
    "iac": "IAC",
    "iil": "IIL",
}


def _get_client() -> httpx.Client:
    global _CLIENT
    if _CLIENT is None or _CLIENT.is_closed:
        _CLIENT = httpx.Client(
            timeout=httpx.Timeout(15.0, connect=10.0),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )
    return _CLIENT


def _cache_get(key: str, ttl_seconds: float) -> Optional[Any]:
    entry = _RESPONSE_CACHE.get(key)
    if not entry:
        return None
    cached_at, value = entry
    if (time.time() - cached_at) >= ttl_seconds:
        return None
    return value


def _cache_set(key: str, value: Any) -> None:
    now = time.time()
    _RESPONSE_CACHE[key] = (now, value)
    # Bound the cache so a long-running process can't grow it without limit. Evict the
    # oldest entries (by insertion/refresh time) once over the cap. Cheap and only runs
    # when the cap is exceeded.
    if len(_RESPONSE_CACHE) > _MAX_CACHE_ENTRIES:
        overflow = len(_RESPONSE_CACHE) - _MAX_CACHE_ENTRIES
        for old_key in sorted(_RESPONSE_CACHE, key=lambda k: _RESPONSE_CACHE[k][0])[:overflow]:
            _RESPONSE_CACHE.pop(old_key, None)


def warm_client() -> None:
    """Open the shared connection pool (optionally prefetch spaces list)."""
    try:
        fetch_spaces()
    except Exception:
        pass


def close_client() -> None:
    global _CLIENT
    if _CLIENT is not None and not _CLIENT.is_closed:
        _CLIENT.close()
    _CLIENT = None


def _api_sensor_slug(metric: str) -> Optional[str]:
    """Return the API sensor slug or None if this metric is not a sensor type."""
    m = metric.lower()
    if m in _VOC_LEGACY_ALIASES:
        return "voc"
    if m in _SENSOR_METRICS:
        return m
    if m in _SOUND_ALIASES:
        return "noise"
    return None


def _score_type(metric: str) -> Optional[str]:
    """Return the API score type string (IEQ, IAQ, …) or None."""
    return _SCORE_METRIC_MAP.get(metric.lower())


def _run_parallel_tasks(tasks: Dict[str, Callable[[], _T]]) -> Dict[str, Optional[_T]]:
    """Run independent API tasks concurrently with bounded worker count."""
    if not tasks:
        return {}
    if len(tasks) == 1:
        key, task = next(iter(tasks.items()))
        try:
            return {key: task()}
        except Exception:
            return {key: None}

    results: Dict[str, Optional[_T]] = {}
    max_workers = min(_MAX_CONCURRENT_API_CALLS, len(tasks))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_key = {executor.submit(task): key for key, task in tasks.items()}
        for future in as_completed(future_to_key):
            key = future_to_key[future]
            try:
                results[key] = future.result()
            except Exception:
                results[key] = None
    return results


def _iso_param(dt: Optional[datetime]) -> Optional[str]:
    """Serialize a datetime to an ISO-8601 string for an API query parameter."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def _resolve_interval_hours(
    window_start: Optional[datetime],
    window_end: Optional[datetime],
    interval_hours: Optional[int],
) -> int:
    """Use the explicit interval if given, else derive granularity from the window."""
    if interval_hours is not None:
        try:
            return max(1, int(interval_hours))
        except (TypeError, ValueError):
            pass
    if window_start is not None and window_end is not None:
        return granularity_hours_for_window(window_start, window_end)
    return 1


def _default_window(window_start: Optional[datetime], window_end: Optional[datetime]) -> Tuple[datetime, datetime]:
    """Fill in a sensible default window (last 24h) when bounds are missing."""
    end = window_end or datetime.now(tz=timezone.utc)
    start = window_start or (end - timedelta(hours=24))
    return start, end


# ---------------------------------------------------------------------------
# Low-level API calls
# ---------------------------------------------------------------------------

def fetch_spaces() -> List[Dict[str, Any]]:
    """GET /spaces/ — returns list of space dicts."""
    cache_key = "spaces"
    cached = _cache_get(cache_key, _SPACES_CACHE_TTL_SECONDS)
    if cached is not None:
        return cached
    try:
        resp = _get_client().get(f"{sensor_api_base_url()}/spaces/", headers={"accept": "application/json"})
        resp.raise_for_status()
        spaces = list(resp.json().get("spaces") or [])
        _cache_set(cache_key, spaces)
        return spaces
    except Exception:
        stale = _RESPONSE_CACHE.get(cache_key)
        if stale is not None:
            return stale[1]
        return []


def fetch_space_metrics(slug: str) -> Optional[Dict[str, Any]]:
    """GET /spaces/{slug}/metrics — returns the space dict with avg_metrics and ieq."""
    cache_key = f"space_metrics:{slug}"
    cached = _cache_get(cache_key, _METRICS_CACHE_TTL_SECONDS)
    if cached is not None:
        return cached
    try:
        resp = _get_client().get(
            f"{sensor_api_base_url()}/spaces/{slug}/metrics",
            headers={"accept": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("success"):
            space = (data.get("data") or {}).get("space")
            if space is not None:
                _cache_set(cache_key, space)
            return space
    except Exception:
        pass
    stale = _RESPONSE_CACHE.get(cache_key)
    if stale is not None:
        return stale[1]
    return None


def fetch_heatmap_metrics(slug: str) -> List[Dict[str, Any]]:
    """GET /spaces/{slug}/heatmap/metrics — per-device latest readings.

    Returns the ``devices`` list, where each device carries its ``metrics`` with
    ``latest_value``/``unit``/``latest_timestamp``. Used by the sensor-inspection
    executor to rank individual sensors and flag stale/offline ones. Returns ``[]``
    on failure (falling back to the last cached value when one exists).
    """
    cache_key = f"heatmap_metrics:{slug}"
    cached = _cache_get(cache_key, _HEATMAP_CACHE_TTL_SECONDS)
    if cached is not None:
        return cached
    try:
        resp = _get_client().get(
            f"{sensor_api_base_url()}/spaces/{slug}/heatmap/metrics",
            headers={"accept": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("success"):
            devices = list(data.get("devices") or [])
            _cache_set(cache_key, devices)
            return devices
    except Exception:
        pass
    stale = _RESPONSE_CACHE.get(cache_key)
    if stale is not None:
        return stale[1]
    return []


def fetch_metric_agg_summary(
    slug: str,
    metric: str,
    window_start: Optional[datetime] = None,
    window_end: Optional[datetime] = None,
    interval_hours: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """GET /spaces/{slug}/metrics/{metric}/agg-summary — returns data dict.

    Sends the explicit ``window_start``/``window_end`` bounds and an
    ``interval_hours`` granularity (derived from the window span when not given).
    """
    api_slug = _api_sensor_slug(metric)
    if not api_slug:
        return None
    start, end = _default_window(window_start, window_end)
    resolved_interval = _resolve_interval_hours(start, end, interval_hours)
    start_iso, end_iso = _iso_param(start), _iso_param(end)
    cache_key = f"agg:{slug}:{api_slug}:{start_iso}:{end_iso}:{resolved_interval}"
    cached = _cache_get(cache_key, _AGG_CACHE_TTL_SECONDS)
    if cached is not None:
        return cached
    try:
        resp = _get_client().get(
            f"{sensor_api_base_url()}/spaces/{slug}/metrics/{api_slug}/agg-summary",
            params={
                "window_start": start_iso,
                "window_end": end_iso,
                "interval_hours": resolved_interval,
            },
            headers={"accept": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("success"):
            payload = data.get("data")
            if payload is not None:
                _cache_set(cache_key, payload)
            return payload
    except Exception:
        pass
    stale = _RESPONSE_CACHE.get(cache_key)
    if stale is not None:
        return stale[1]
    return None


def fetch_indoor_data(
    slug: str,
    score_type: str,
    window_start: Optional[datetime] = None,
    window_end: Optional[datetime] = None,
    interval: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """GET /spaces/{slug}/indoor-data — returns data dict.

    Sends the explicit ``window_start``/``window_end`` bounds and an ``interval``
    granularity (hours per bucket, derived from the window span when not given).
    """
    start, end = _default_window(window_start, window_end)
    resolved_interval = _resolve_interval_hours(start, end, interval)
    start_iso, end_iso = _iso_param(start), _iso_param(end)
    cache_key = f"indoor:{slug}:{score_type}:{start_iso}:{end_iso}:{resolved_interval}"
    cached = _cache_get(cache_key, _INDOOR_CACHE_TTL_SECONDS)
    if cached is not None:
        return cached
    try:
        resp = _get_client().get(
            f"{sensor_api_base_url()}/spaces/{slug}/indoor-data",
            params={
                "type": score_type,
                "interval": resolved_interval,
                "window_start": start_iso,
                "window_end": end_iso,
            },
            headers={"accept": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("success"):
            payload = data.get("data")
            if payload is not None:
                _cache_set(cache_key, payload)
            return payload
    except Exception:
        pass
    stale = _RESPONSE_CACHE.get(cache_key)
    if stale is not None:
        return stale[1]
    return None


# ---------------------------------------------------------------------------
# Row-format conversion helpers
# ---------------------------------------------------------------------------

def fetch_timeseries_rows(
    slug: str,
    metric: str,
    window_start: datetime,
    window_end: datetime,
    interval_hours: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Fetch hourly rows as [{"lab_space", "bucket", "value"}, …] newest-first → oldest-first."""
    api_slug = _api_sensor_slug(metric)
    score = _score_type(metric)
    rows: List[Dict[str, Any]] = []
    if api_slug:
        data = fetch_metric_agg_summary(slug, metric, window_start, window_end, interval_hours)
        if data:
            rows = [
                {"lab_space": slug, "bucket": item["timestamp"], "value": item["agg_value"]}
                for item in (data.get("aggregate_readings") or [])
            ]
    elif score:
        data = fetch_indoor_data(slug, score, window_start, window_end, interval_hours)
        if data:
            rows = [
                {"lab_space": slug, "bucket": item["bucket"], "value": item["value"]}
                for item in (data.get("readings") or [])
            ]
    # Ensure ascending bucket order (API returns newest-first for some endpoints)
    rows.sort(key=lambda r: str(r.get("bucket") or ""))
    return rows


def fetch_aggregation_row(
    slug: str,
    metric: str,
    window_start: datetime,
    window_end: datetime,
) -> Optional[Dict[str, Any]]:
    """Fetch aggregated {avg_value, min_value, max_value, reading_count} for one metric."""
    api_slug = _api_sensor_slug(metric)
    score = _score_type(metric)
    if api_slug:
        data = fetch_metric_agg_summary(slug, metric, window_start, window_end)
        if not data:
            return None
        readings = data.get("aggregate_readings") or []
        return {
            "lab_space": slug,
            "avg_value": data.get("avg_agg_value"),
            "min_value": data.get("min_agg_value"),
            "max_value": data.get("max_agg_value"),
            "reading_count": len(readings),
        }
    elif score:
        data = fetch_indoor_data(slug, score, window_start, window_end)
        if not data:
            return None
        values = [r["value"] for r in (data.get("readings") or []) if r.get("value") is not None]
        if not values:
            return None
        return {
            "lab_space": slug,
            "avg_value": sum(values) / len(values),
            "min_value": min(values),
            "max_value": max(values),
            "reading_count": len(values),
        }
    return None


def fetch_ieq_latest_with_subindices(slug: str, lookback_hours: int = 48) -> Dict[str, Any]:
    """Fetch latest value for each IEQ sub-index via separate indoor-data calls.

    The API returns newest-first; we pick the reading with the latest bucket.
    """
    score_types = {"iaq": "IAQ", "itc": "ITC", "iac": "IAC", "iil": "IIL"}
    window_end = datetime.now(tz=timezone.utc)
    window_start = window_end - timedelta(hours=lookback_hours)
    responses = _run_parallel_tasks(
        {
            key: (
                lambda score_type=score_type: fetch_indoor_data(
                    slug,
                    score_type,
                    window_start,
                    window_end,
                    interval=1,
                )
            )
            for key, score_type in score_types.items()
        }
    )
    result: Dict[str, Any] = {}
    for key in ("iaq", "itc", "iac", "iil"):
        data = responses.get(key)
        if not data:
            continue
        readings = data.get("readings") or []
        if not readings:
            continue
        latest = max(readings, key=lambda r: str(r.get("bucket") or ""))
        val = latest.get("value")
        if val is not None:
            result[key] = val
    return result


def fetch_multi_metric_point_row(slug: str, metrics: List[str]) -> Dict[str, Any]:
    """Return a single row with the latest average for each metric using /metrics endpoint."""
    row: Dict[str, Any] = {"lab_space": slug}
    space = fetch_space_metrics(slug)
    if not space:
        return row
    avg_by_type = {m["type"]: m["avg_value"] for m in (space.get("avg_metrics") or [])}
    row["bucket"] = space.get("last_updated")
    row["ieq"] = (space.get("ieq") or {}).get("score")
    for metric in metrics:
        api_slug = _api_sensor_slug(metric)
        score = _score_type(metric)
        if api_slug:
            row[metric] = avg_by_type.get(api_slug)
        elif score and metric == "ieq":
            pass  # already set above
        elif score:
            row[metric] = None
    if "ieq" in metrics:
        sub = fetch_ieq_latest_with_subindices(slug)
        if sub.get("ieq") is not None and row.get("ieq") is None:
            row["ieq"] = sub["ieq"]
        for key in ("iaq", "itc", "iac", "iil"):
            if sub.get(key) is not None:
                row[key] = sub[key]
    return row


def fetch_multi_metric_agg_row(
    slug: str,
    metrics: List[str],
    window_start: datetime,
    window_end: datetime,
) -> Dict[str, Any]:
    """Return a row with avg/min/max/stddev_placeholder for each metric."""
    row: Dict[str, Any] = {"lab_space": slug, "reading_count": 0}
    agg_by_metric = _run_parallel_tasks(
        {
            metric: (
                lambda metric_name=metric: fetch_aggregation_row(
                    slug,
                    metric_name,
                    window_start,
                    window_end,
                )
            )
            for metric in metrics
        }
    )
    for metric in metrics:
        agg = agg_by_metric.get(metric)
        if agg:
            row[metric] = agg.get("avg_value")
            row[f"{metric}_min"] = agg.get("min_value")
            row[f"{metric}_max"] = agg.get("max_value")
            row[f"{metric}_stddev"] = None
            if not row["reading_count"]:
                row["reading_count"] = agg.get("reading_count", 0)
        else:
            row[metric] = None
            row[f"{metric}_min"] = None
            row[f"{metric}_max"] = None
            row[f"{metric}_stddev"] = None
    return row


def fetch_all_spaces_point_row(metrics: List[str]) -> Dict[str, Any]:
    """Average the *latest* reading of each metric across every known space.

    The point-lookup counterpart to :func:`fetch_all_spaces_avg_row`, for a current-status
    question that names no space. Returning one arbitrary space's readings instead would
    present a single room's air as the whole site's.
    """
    slugs = [s.get("slug") for s in fetch_spaces() if s.get("slug")]
    row: Dict[str, Any] = {"lab_space": "all_labs"}
    if not slugs:
        for metric in metrics:
            row[metric] = None
        return row

    per_space = [
        value
        for value in _run_parallel_tasks(
            {slug: (lambda s=slug: fetch_multi_metric_point_row(s, metrics)) for slug in slugs}
        ).values()
        if value
    ]
    buckets = [str(r.get("bucket")) for r in per_space if r.get("bucket")]
    if buckets:
        row["bucket"] = max(buckets)
    # "ieq" is set by fetch_multi_metric_point_row outside the requested metric list, and
    # the sub-indices come along with it, so average whatever keys came back.
    value_keys = set(metrics) | {"ieq", "iaq", "itc", "iac", "iil"}
    for key in value_keys:
        values = [r[key] for r in per_space if isinstance(r.get(key), (int, float))]
        row[key] = (sum(values) / len(values)) if values else None
    return row


def fetch_all_spaces_avg_row(
    metrics: List[str],
    window_start: datetime,
    window_end: datetime,
) -> Dict[str, Any]:
    """Aggregate metric values across all known spaces into a single 'all_labs' row."""
    spaces = fetch_spaces()
    slugs = [s["slug"] for s in spaces if s.get("slug")]
    if not slugs:
        row: Dict[str, Any] = {"lab_space": "all_labs", "reading_count": 0}
        for m in metrics:
            row[m] = None
            row[f"{m}_min"] = None
            row[f"{m}_max"] = None
            row[f"{m}_stddev"] = None
        return row

    per_space = list(
        _run_parallel_tasks(
            {
                slug: (lambda space_slug=slug: fetch_multi_metric_agg_row(space_slug, metrics, window_start, window_end))
                for slug in slugs
            }
        ).values()
    )
    row = {"lab_space": "all_labs", "reading_count": 0}
    for metric in metrics:
        vals = [r[metric] for r in per_space if r and r.get(metric) is not None]
        mins = [r.get(f"{metric}_min") for r in per_space if r and r.get(f"{metric}_min") is not None]
        maxs = [r.get(f"{metric}_max") for r in per_space if r and r.get(f"{metric}_max") is not None]
        row[metric] = sum(vals) / len(vals) if vals else None
        row[f"{metric}_min"] = min(mins) if mins else None
        row[f"{metric}_max"] = max(maxs) if maxs else None
        row[f"{metric}_stddev"] = None
    row["reading_count"] = sum((r or {}).get("reading_count", 0) for r in per_space)
    return row


def fetch_predictions(slug: str, metric: str) -> Optional[Dict[str, Any]]:
    """GET /spaces/{slug}/metrics/{metric}/predictions — returns next-6-hour forecast dict.

    Tries the API slug first (e.g. 'voc'), falls back to the canonical
    name so the caller does not need to normalise before calling.
    Returns the raw payload dict or None on failure.
    """
    api_slug = _api_sensor_slug(metric) or metric.lower()
    cache_key = f"predictions:{slug}:{api_slug}"
    cached = _cache_get(cache_key, _PREDICTIONS_CACHE_TTL_SECONDS)
    if cached is not None:
        return cached
    try:
        resp = _get_client().get(
            f"{predictions_api_base_url()}/spaces/{slug}/metrics/{api_slug}/predictions",
            headers={"accept": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict):
            payload = data.get("data") if data.get("success") else data
            if payload is not None:
                _cache_set(cache_key, payload)
                return payload
    except Exception:
        pass
    stale = _RESPONSE_CACHE.get(cache_key)
    if stale is not None:
        return stale[1]
    return None


def fetch_all_spaces_agg_rows_for_metric(
    metric: str,
    window_start: datetime,
    window_end: datetime,
) -> List[Dict[str, Any]]:
    """Return per-space aggregation rows for a single metric (for ranking/comparison)."""
    spaces = fetch_spaces()
    slugs = [s.get("slug") for s in spaces if s.get("slug")]
    agg_by_slug = _run_parallel_tasks(
        {
            slug: (lambda space_slug=slug: fetch_aggregation_row(space_slug, metric, window_start, window_end))
            for slug in slugs
        }
    )
    rows: List[Dict[str, Any]] = [agg for agg in agg_by_slug.values() if agg]
    rows.sort(key=lambda r: (r.get("avg_value") or 0), reverse=True)
    return rows


def fetch_merged_timeseries(
    slug: str,
    metrics: List[str],
    window_start: datetime,
    window_end: datetime,
    interval_hours: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Fetch per-bucket time series for multiple metrics, merged by timestamp."""
    series_by_metric = _run_parallel_tasks(
        {
            metric: (
                lambda metric_name=metric: fetch_timeseries_rows(
                    slug,
                    metric_name,
                    window_start,
                    window_end,
                    interval_hours,
                )
            )
            for metric in metrics
        }
    )
    by_bucket: Dict[str, Dict[str, Any]] = {}
    for metric in metrics:
        series = series_by_metric.get(metric) or []
        for row in series:
            bucket = str(row.get("bucket") or "")
            if not bucket:
                continue
            if bucket not in by_bucket:
                by_bucket[bucket] = {"lab_space": slug, "bucket": bucket}
            by_bucket[bucket][metric] = row.get("value")
    return sorted(by_bucket.values(), key=lambda r: r["bucket"])
