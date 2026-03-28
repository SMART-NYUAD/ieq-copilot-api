"""In-process observability counters for routing rollout safety."""

from __future__ import annotations

from collections import Counter
from threading import Lock
from typing import Any, Dict, Optional


_LOCK = Lock()
_COUNTERS = {
    "planner_total": 0,
    "planner_fallback_total": 0,
    "critic_total": 0,
    "critic_failure_total": 0,
}
_FALLBACK_REASON_COUNTS: Counter[str] = Counter()
_TEMPLATE_COUNTS: Counter[str] = Counter()


def reset_observability_metrics() -> None:
    with _LOCK:
        _COUNTERS["planner_total"] = 0
        _COUNTERS["planner_fallback_total"] = 0
        _COUNTERS["critic_total"] = 0
        _COUNTERS["critic_failure_total"] = 0
        _FALLBACK_REASON_COUNTS.clear()
        _TEMPLATE_COUNTS.clear()


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


def get_observability_snapshot() -> Dict[str, Any]:
    with _LOCK:
        planner_total = int(_COUNTERS["planner_total"])
        planner_fallback_total = int(_COUNTERS["planner_fallback_total"])
        critic_total = int(_COUNTERS["critic_total"])
        critic_failure_total = int(_COUNTERS["critic_failure_total"])
        fallback_distribution = dict(_FALLBACK_REASON_COUNTS)
        template_distribution = dict(_TEMPLATE_COUNTS)

    fallback_rate = (planner_fallback_total / planner_total) if planner_total else 0.0
    critic_failure_rate = (critic_failure_total / critic_total) if critic_total else 0.0

    return {
        "planner_total": planner_total,
        "planner_fallback_total": planner_fallback_total,
        "planner_fallback_rate": fallback_rate,
        "planner_fallback_reason_distribution": fallback_distribution,
        "critic_total": critic_total,
        "critic_failure_total": critic_failure_total,
        "critic_failure_rate": critic_failure_rate,
        "decomposition_template_usage": template_distribution,
    }


def evaluate_rollout_slo(snapshot: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    data = snapshot or get_observability_snapshot()
    fallback_rate = float(data.get("planner_fallback_rate") or 0.0)
    critic_failure_rate = float(data.get("critic_failure_rate") or 0.0)

    fallback_target_ok = fallback_rate <= 0.05
    fallback_max_ok = fallback_rate <= 0.10
    critic_target_ok = critic_failure_rate <= 0.02
    critic_max_ok = critic_failure_rate <= 0.05

    return {
        "fallback_target_ok": fallback_target_ok,
        "fallback_max_ok": fallback_max_ok,
        "critic_target_ok": critic_target_ok,
        "critic_max_ok": critic_max_ok,
        "rollout_blocked": not (fallback_max_ok and critic_max_ok),
    }
