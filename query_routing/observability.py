"""In-process observability counters for routing rollout safety and API KPIs."""

from __future__ import annotations

from collections import Counter, deque
from datetime import datetime, timezone
import statistics
from threading import Lock
import time
from typing import Any, Deque, Dict, Optional


_LOCK = Lock()
_STARTED_AT_UTC = datetime.now(timezone.utc)
_STARTED_MONOTONIC = time.monotonic()
_COUNTERS = {
    "planner_total": 0,
    "planner_fallback_total": 0,
    "critic_total": 0,
    "critic_failure_total": 0,
    "shadow_total": 0,
    "shadow_diff_total": 0,
    "rollout_policy_total": 0,
    "sync_stream_total": 0,
    "sync_stream_flip_total": 0,
}
_FALLBACK_REASON_COUNTS: Counter[str] = Counter()
_TEMPLATE_COUNTS: Counter[str] = Counter()
_SHADOW_DIFF_COUNTS: Counter[str] = Counter()
_ENDPOINT_EXECUTION_CACHE: Dict[str, str] = {}
_HTTP_COUNTERS = {
    "http_requests_total": 0,
    "http_errors_total": 0,
    "http_inflight": 0,
}
_HTTP_METHOD_COUNTS: Counter[str] = Counter()
_HTTP_PATH_COUNTS: Counter[str] = Counter()
_HTTP_STATUS_COUNTS: Counter[str] = Counter()
_HTTP_ENDPOINT_COUNTS: Counter[str] = Counter()
_HTTP_LATENCY_MS: Deque[float] = deque(maxlen=5000)
_HTTP_APP_LATENCY_MS: Deque[float] = deque(maxlen=5000)
_HTTP_ENDPOINT_LATENCY_MS: Dict[str, Deque[float]] = {}
_HTTP_REQUEST_EVENTS: Deque[tuple[float, str, int]] = deque(maxlen=25000)
_RUNTIME_ERROR_CODE_COUNTS: Counter[str] = Counter()
_RUNTIME_ERROR_SCOPE_COUNTS: Counter[str] = Counter()


def _latency_percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    position = (len(values) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    if lower == upper:
        return float(values[lower])
    weight = position - lower
    return float((values[lower] * (1.0 - weight)) + (values[upper] * weight))


def _is_internal_path(path: str) -> bool:
    text = str(path or "").strip().lower()
    return (
        text == "/"
        or text.startswith("/health")
        or text.startswith("/observability")
        or text.startswith("/docs")
        or text.startswith("/redoc")
        or text.startswith("/openapi")
    )


def _latency_summary(values: list[float]) -> Dict[str, float]:
    if not values:
        return {
            "count": 0.0,
            "mean_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "max_ms": 0.0,
        }
    ordered = sorted(values)
    return {
        "count": float(len(ordered)),
        "mean_ms": float(statistics.mean(ordered)),
        "p50_ms": _latency_percentile(ordered, 0.50),
        "p95_ms": _latency_percentile(ordered, 0.95),
        "p99_ms": _latency_percentile(ordered, 0.99),
        "max_ms": float(ordered[-1]),
    }


def _build_status_timeline(
    *,
    events: list[tuple[float, str, int]],
    now_monotonic: float,
    bucket_seconds: int,
    bucket_count: int,
    path_filter: Optional[str] = None,
) -> Dict[str, Any]:
    totals = [0 for _ in range(bucket_count)]
    errors = [0 for _ in range(bucket_count)]
    window_seconds = bucket_seconds * bucket_count
    window_start = now_monotonic - float(window_seconds)
    for event_time, path, status_code in events:
        if event_time < window_start:
            continue
        if path_filter and path != path_filter:
            continue
        index = int((event_time - window_start) // float(bucket_seconds))
        if index < 0 or index >= bucket_count:
            continue
        totals[index] += 1
        if int(status_code) >= 400:
            errors[index] += 1
    bars: list[Dict[str, Any]] = []
    for idx in range(bucket_count):
        total = totals[idx]
        error_total = errors[idx]
        error_rate = (error_total / total) if total else 0.0
        if total == 0:
            status = "no_data"
        elif error_rate == 0.0:
            status = "ok"
        elif error_rate <= 0.10:
            status = "warn"
        else:
            status = "bad"
        bars.append(
            {
                "bucket_index": idx,
                "total": total,
                "errors": error_total,
                "error_rate": error_rate,
                "status": status,
            }
        )
    total_requests = int(sum(totals))
    total_errors = int(sum(errors))
    uptime = 1.0 - (total_errors / total_requests) if total_requests else 1.0
    return {
        "bucket_seconds": bucket_seconds,
        "bucket_count": bucket_count,
        "window_seconds": window_seconds,
        "total_requests": total_requests,
        "total_errors": total_errors,
        "uptime_rate": uptime,
        "bars": bars,
    }


def reset_observability_metrics() -> None:
    with _LOCK:
        _COUNTERS["planner_total"] = 0
        _COUNTERS["planner_fallback_total"] = 0
        _COUNTERS["critic_total"] = 0
        _COUNTERS["critic_failure_total"] = 0
        _COUNTERS["shadow_total"] = 0
        _COUNTERS["shadow_diff_total"] = 0
        _COUNTERS["rollout_policy_total"] = 0
        _COUNTERS["sync_stream_total"] = 0
        _COUNTERS["sync_stream_flip_total"] = 0
        _FALLBACK_REASON_COUNTS.clear()
        _TEMPLATE_COUNTS.clear()
        _SHADOW_DIFF_COUNTS.clear()
        _ENDPOINT_EXECUTION_CACHE.clear()
        _HTTP_COUNTERS["http_requests_total"] = 0
        _HTTP_COUNTERS["http_errors_total"] = 0
        _HTTP_COUNTERS["http_inflight"] = 0
        _HTTP_METHOD_COUNTS.clear()
        _HTTP_PATH_COUNTS.clear()
        _HTTP_STATUS_COUNTS.clear()
        _HTTP_ENDPOINT_COUNTS.clear()
        _HTTP_LATENCY_MS.clear()
        _HTTP_APP_LATENCY_MS.clear()
        _HTTP_ENDPOINT_LATENCY_MS.clear()
        _HTTP_REQUEST_EVENTS.clear()
        _RUNTIME_ERROR_CODE_COUNTS.clear()
        _RUNTIME_ERROR_SCOPE_COUNTS.clear()


def record_http_request_start() -> None:
    with _LOCK:
        _HTTP_COUNTERS["http_inflight"] += 1


def record_http_request_end(
    *,
    method: str,
    path: str,
    status_code: int,
    duration_ms: float,
) -> None:
    normalized_method = str(method or "UNKNOWN").upper()
    normalized_path = str(path or "/unknown")
    status = str(int(status_code)) if status_code else "0"
    endpoint_key = f"{normalized_method} {normalized_path}"
    with _LOCK:
        _HTTP_COUNTERS["http_requests_total"] += 1
        if int(status_code or 0) >= 400:
            _HTTP_COUNTERS["http_errors_total"] += 1
        _HTTP_COUNTERS["http_inflight"] = max(0, _HTTP_COUNTERS["http_inflight"] - 1)
        _HTTP_METHOD_COUNTS[normalized_method] += 1
        _HTTP_PATH_COUNTS[normalized_path] += 1
        _HTTP_STATUS_COUNTS[status] += 1
        _HTTP_ENDPOINT_COUNTS[endpoint_key] += 1
        _HTTP_LATENCY_MS.append(float(max(0.0, duration_ms)))
        if not _is_internal_path(normalized_path):
            _HTTP_APP_LATENCY_MS.append(float(max(0.0, duration_ms)))
            _HTTP_REQUEST_EVENTS.append((time.monotonic(), normalized_path, int(status_code or 0)))
        if endpoint_key not in _HTTP_ENDPOINT_LATENCY_MS:
            _HTTP_ENDPOINT_LATENCY_MS[endpoint_key] = deque(maxlen=1200)
        _HTTP_ENDPOINT_LATENCY_MS[endpoint_key].append(float(max(0.0, duration_ms)))


def record_runtime_error(*, code: str, scope: str) -> None:
    normalized_code = str(code or "unknown").strip().lower() or "unknown"
    normalized_scope = str(scope or "unknown").strip().lower() or "unknown"
    with _LOCK:
        _RUNTIME_ERROR_CODE_COUNTS[normalized_code] += 1
        _RUNTIME_ERROR_SCOPE_COUNTS[normalized_scope] += 1


def record_route_plan(route_plan: Any) -> None:
    with _LOCK:
        _COUNTERS["planner_total"] += 1

        fallback_used = bool(getattr(route_plan, "planner_fallback_used", False))
        if fallback_used:
            _COUNTERS["planner_fallback_total"] += 1
            reason = str(getattr(route_plan, "planner_fallback_reason", "unknown") or "unknown")
            _FALLBACK_REASON_COUNTS[reason] += 1

        template = getattr(route_plan, "decomposition_template", None)
        if template is not None:
            template_id = str(getattr(template, "value", template) or "unknown")
            if template_id:
                _TEMPLATE_COUNTS[template_id] += 1


def record_critic_outcome(status: Optional[str]) -> None:
    value = str(status or "").strip().lower()
    if not value:
        return
    with _LOCK:
        _COUNTERS["critic_total"] += 1
        if value != "pass":
            _COUNTERS["critic_failure_total"] += 1


def record_rollout_selection(selected_policy: str) -> None:
    with _LOCK:
        _COUNTERS["rollout_policy_total"] += 1


def record_shadow_comparison(
    *,
    sampled: bool,
    active_executor: str,
    shadow_executor: str,
) -> None:
    if not sampled:
        return
    active = str(active_executor or "").strip().lower() or "unknown"
    shadow = str(shadow_executor or "").strip().lower() or "unknown"
    with _LOCK:
        _COUNTERS["shadow_total"] += 1
        if active != shadow:
            _COUNTERS["shadow_diff_total"] += 1
            _SHADOW_DIFF_COUNTS[f"{active}->{shadow}"] += 1


def record_endpoint_executor(
    *,
    latest_question_hash: str,
    endpoint_key: str,
    executor: str,
) -> None:
    question_hash = str(latest_question_hash or "").strip().lower()
    endpoint = str(endpoint_key or "").strip().lower()
    exec_value = str(executor or "").strip().lower()
    if not question_hash or not endpoint or not exec_value:
        return
    key = f"{question_hash}:{endpoint}"
    with _LOCK:
        _COUNTERS["sync_stream_total"] += 1
        prior = _ENDPOINT_EXECUTION_CACHE.get(key)
        if prior and prior != exec_value:
            _COUNTERS["sync_stream_flip_total"] += 1
        _ENDPOINT_EXECUTION_CACHE[key] = exec_value


def get_observability_snapshot() -> Dict[str, Any]:
    with _LOCK:
        planner_total = int(_COUNTERS["planner_total"])
        planner_fallback_total = int(_COUNTERS["planner_fallback_total"])
        critic_total = int(_COUNTERS["critic_total"])
        critic_failure_total = int(_COUNTERS["critic_failure_total"])
        shadow_total = int(_COUNTERS["shadow_total"])
        shadow_diff_total = int(_COUNTERS["shadow_diff_total"])
        rollout_policy_total = int(_COUNTERS["rollout_policy_total"])
        sync_stream_total = int(_COUNTERS["sync_stream_total"])
        sync_stream_flip_total = int(_COUNTERS["sync_stream_flip_total"])
        fallback_distribution = dict(_FALLBACK_REASON_COUNTS)
        template_distribution = dict(_TEMPLATE_COUNTS)
        shadow_diff_distribution = dict(_SHADOW_DIFF_COUNTS)

    fallback_rate = (planner_fallback_total / planner_total) if planner_total else 0.0
    critic_failure_rate = (critic_failure_total / critic_total) if critic_total else 0.0
    shadow_diff_rate = (shadow_diff_total / shadow_total) if shadow_total else 0.0
    rollout_policy_rate = 1.0 if rollout_policy_total else 0.0
    sync_stream_flip_rate = (
        (sync_stream_flip_total / sync_stream_total) if sync_stream_total else 0.0
    )

    return {
        "planner_total": planner_total,
        "planner_fallback_total": planner_fallback_total,
        "planner_fallback_rate": fallback_rate,
        "planner_fallback_reason_distribution": fallback_distribution,
        "critic_total": critic_total,
        "critic_failure_total": critic_failure_total,
        "critic_failure_rate": critic_failure_rate,
        "decomposition_template_usage": template_distribution,
        "shadow_total": shadow_total,
        "shadow_diff_total": shadow_diff_total,
        "shadow_diff_rate": shadow_diff_rate,
        "shadow_diff_distribution": shadow_diff_distribution,
        "rollout_policy_total": rollout_policy_total,
        "rollout_policy_rate": rollout_policy_rate,
        "sync_stream_total": sync_stream_total,
        "sync_stream_flip_total": sync_stream_flip_total,
        "sync_stream_flip_rate": sync_stream_flip_rate,
    }


def get_http_observability_snapshot(top_n: int = 10) -> Dict[str, Any]:
    with _LOCK:
        total = int(_HTTP_COUNTERS["http_requests_total"])
        errors = int(_HTTP_COUNTERS["http_errors_total"])
        inflight = int(_HTTP_COUNTERS["http_inflight"])
        method_counts = dict(_HTTP_METHOD_COUNTS)
        path_counts = dict(_HTTP_PATH_COUNTS)
        status_counts = dict(_HTTP_STATUS_COUNTS)
        endpoint_counts = dict(_HTTP_ENDPOINT_COUNTS)
        latency_samples = list(_HTTP_LATENCY_MS)
        app_latency_samples = list(_HTTP_APP_LATENCY_MS)
        events = list(_HTTP_REQUEST_EVENTS)
        endpoint_latency_samples = {
            endpoint: list(samples)
            for endpoint, samples in _HTTP_ENDPOINT_LATENCY_MS.items()
        }

    uptime_seconds = max(1e-6, float(time.monotonic() - _STARTED_MONOTONIC))
    error_rate = (errors / total) if total else 0.0
    throughput_rps = (total / uptime_seconds) if total else 0.0
    top_endpoints = sorted(
        endpoint_counts.items(),
        key=lambda item: item[1],
        reverse=True,
    )[: max(1, int(top_n))]
    endpoint_latency = {}
    for endpoint, count in top_endpoints:
        summary = _latency_summary(endpoint_latency_samples.get(endpoint, []))
        endpoint_latency[endpoint] = {
            "count": count,
            "mean_ms": summary["mean_ms"],
            "p50_ms": summary["p50_ms"],
            "p95_ms": summary["p95_ms"],
            "p99_ms": summary["p99_ms"],
            "max_ms": summary["max_ms"],
        }
    global_latency = _latency_summary(latency_samples)
    app_latency = _latency_summary(app_latency_samples)
    timeline_bucket_seconds = 60
    timeline_bucket_count = 90
    overall_timeline = _build_status_timeline(
        events=events,
        now_monotonic=time.monotonic(),
        bucket_seconds=timeline_bucket_seconds,
        bucket_count=timeline_bucket_count,
        path_filter=None,
    )
    top_paths = sorted(
        [(path, count) for path, count in path_counts.items() if not _is_internal_path(path)],
        key=lambda item: item[1],
        reverse=True,
    )[:5]
    path_timelines: Dict[str, Any] = {}
    for path, _ in top_paths:
        path_timelines[path] = _build_status_timeline(
            events=events,
            now_monotonic=time.monotonic(),
            bucket_seconds=timeline_bucket_seconds,
            bucket_count=timeline_bucket_count,
            path_filter=path,
        )
    return {
        "service_started_at": _STARTED_AT_UTC.isoformat(),
        "uptime_seconds": uptime_seconds,
        "http_requests_total": total,
        "http_errors_total": errors,
        "http_error_rate": error_rate,
        "http_inflight": inflight,
        "throughput_rps": throughput_rps,
        "method_distribution": method_counts,
        "path_distribution": path_counts,
        "status_distribution": status_counts,
        "latency_ms": {
            "count": int(global_latency["count"]),
            "mean_ms": global_latency["mean_ms"],
            "p50_ms": global_latency["p50_ms"],
            "p95_ms": global_latency["p95_ms"],
            "p99_ms": global_latency["p99_ms"],
            "max_ms": global_latency["max_ms"],
        },
        "app_latency_ms": {
            "count": int(app_latency["count"]),
            "mean_ms": app_latency["mean_ms"],
            "p50_ms": app_latency["p50_ms"],
            "p95_ms": app_latency["p95_ms"],
            "p99_ms": app_latency["p99_ms"],
            "max_ms": app_latency["max_ms"],
        },
        "status_timeline": overall_timeline,
        "path_status_timeline": path_timelines,
        "endpoint_latency_ms": endpoint_latency,
    }


def get_error_observability_snapshot(top_n: int = 10) -> Dict[str, Any]:
    with _LOCK:
        by_code = dict(_RUNTIME_ERROR_CODE_COUNTS)
        by_scope = dict(_RUNTIME_ERROR_SCOPE_COUNTS)
    top_scopes = dict(
        sorted(
            by_scope.items(),
            key=lambda item: item[1],
            reverse=True,
        )[: max(1, int(top_n))]
    )
    total_errors = int(sum(by_code.values()))
    return {
        "runtime_errors_total": total_errors,
        "runtime_error_by_code": by_code,
        "runtime_error_top_scopes": top_scopes,
    }


def evaluate_rollout_slo(snapshot: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    data = snapshot or get_observability_snapshot()
    fallback_rate = float(data.get("planner_fallback_rate") or 0.0)
    critic_failure_rate = float(data.get("critic_failure_rate") or 0.0)
    shadow_diff_rate = float(data.get("shadow_diff_rate") or 0.0)
    sync_stream_flip_rate = float(data.get("sync_stream_flip_rate") or 0.0)

    fallback_target_ok = fallback_rate <= 0.05
    fallback_max_ok = fallback_rate <= 0.10
    critic_target_ok = critic_failure_rate <= 0.02
    critic_max_ok = critic_failure_rate <= 0.05
    shadow_target_ok = shadow_diff_rate <= 0.10
    shadow_max_ok = shadow_diff_rate <= 0.20
    parity_target_ok = sync_stream_flip_rate <= 0.0
    parity_max_ok = sync_stream_flip_rate <= 0.01

    return {
        "fallback_target_ok": fallback_target_ok,
        "fallback_max_ok": fallback_max_ok,
        "critic_target_ok": critic_target_ok,
        "critic_max_ok": critic_max_ok,
        "shadow_target_ok": shadow_target_ok,
        "shadow_max_ok": shadow_max_ok,
        "parity_target_ok": parity_target_ok,
        "parity_max_ok": parity_max_ok,
        "rollout_blocked": not (fallback_max_ok and critic_max_ok and shadow_max_ok and parity_max_ok),
    }


def get_observability_kpis() -> Dict[str, Any]:
    router = get_observability_snapshot()
    http = get_http_observability_snapshot()
    errors = get_error_observability_snapshot()
    slo = evaluate_rollout_slo(router)
    request_total = int(http.get("http_requests_total") or 0)
    request_errors = int(http.get("http_errors_total") or 0)
    request_error_rate = float(http.get("http_error_rate") or 0.0)
    availability = 1.0 - request_error_rate if request_total else 1.0
    return {
        "kpis": {
            "availability_rate": availability,
            "request_error_rate": request_error_rate,
            "throughput_rps": float(http.get("throughput_rps") or 0.0),
            "latency_p95_ms": float((http.get("app_latency_ms") or {}).get("p95_ms") or 0.0),
            "latency_p99_ms": float((http.get("app_latency_ms") or {}).get("p99_ms") or 0.0),
            "router_planner_fallback_rate": float(router.get("planner_fallback_rate") or 0.0),
            "router_critic_failure_rate": float(router.get("critic_failure_rate") or 0.0),
            "router_shadow_diff_rate": float(router.get("shadow_diff_rate") or 0.0),
            "router_sync_stream_flip_rate": float(router.get("sync_stream_flip_rate") or 0.0),
            "runtime_errors_total": int(errors.get("runtime_errors_total") or 0),
            "requests_total": request_total,
            "request_errors_total": request_errors,
        },
        "router": router,
        "router_slo": slo,
        "http": http,
        "errors": errors,
    }
